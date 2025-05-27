from collections import defaultdict
import re
from pathlib import Path
import logging

import torch
import safetensors.torch

from .utils import cached, JsonCache
from .num_utils import fast_decompose
from .sdxl_mapper import get_sdxl_lora_keys, get_multi_format_lora_keys


__all__ = ["BaseCheckpoint", "PairedLoraModel", "JsonCache"]

RE_NAME_SPLIT = re.compile(r"[\-_ ]").split


class BaseCheckpoint:
    """A class for indexing checkpoints weights by LoRA layer names"""

    def __init__(self, path, key_mapper=get_multi_format_lora_keys, cache=None):
        self.path = path = Path(path)
        self.fd = fd = safetensors.safe_open(path, framework="pt")
        self._cache = {} if cache is None else cache[str(path.resolve())]
        self._cached_weights_name = None

        base_keys_from_file = fd.keys()
        self.shapes = shapes = {}
        # base2lora stores the direct name(s) returned by the mapper for a base_key_str
        # e.g., base_key_str -> "lora_name" OR base_key_str -> ["alias1", "alias2"]
        self.base2lora = base2lora_mapping = {}
        # lora2base maps any known LoRA name variant to its base layer info
        # e.g., "lora_name" -> "base_key_str" OR "lora_part_name" -> ("base_key_str", idx, count)
        self.lora2base = lora2base_mapping = {}

        for base_key_str in base_keys_from_file:
            if base_key_str.startswith("first_stage_model."):
                continue  # ignore VAE
            shapes[base_key_str] = shape = fd.get_slice(base_key_str).get_shape()
            if not base_key_str.endswith("weight") or len(shape) < 2:
                continue

            # mapped_info will be (is_split, names_from_mapper) or None
            mapped_info = key_mapper(base_key_str)

            if mapped_info is None:
                continue

            is_split_by_mapper, names_from_mapper = mapped_info

            base2lora_mapping[base_key_str] = names_from_mapper

            if is_split_by_mapper:
                # names_from_mapper must be a list of part names (e.g., Q, K, V)
                if not isinstance(names_from_mapper, list):
                    logging.error(
                        f"Internal inconsistency: key_mapper indicated a split for {base_key_str} "
                        f"but did not return a list of names: {names_from_mapper}. Skipping this base key."
                    )
                    continue

                num_parts = len(names_from_mapper)
                if num_parts == 0:
                    logging.warning(
                        f"key_mapper indicated split for {base_key_str} but returned empty list of names."
                    )
                    continue

                for i, part_lora_key in enumerate(names_from_mapper):
                    lora2base_mapping[part_lora_key] = (base_key_str, i, num_parts)
            else:
                # names_from_mapper is a single string (one LoRA name)
                # or a list of alias strings for an unsplit layer.
                if isinstance(names_from_mapper, list):  # List of aliases
                    for alias_lora_key in names_from_mapper:
                        lora2base_mapping[alias_lora_key] = base_key_str
                elif isinstance(names_from_mapper, str):  # Single name
                    lora2base_mapping[names_from_mapper] = base_key_str
                else:
                    logging.error(
                        f"Internal inconsistency: key_mapper indicated no split for {base_key_str} "
                        f"but returned an unexpected type: {names_from_mapper}. Skipping."
                    )
                    continue

        # Assign to self attributes after loop
        self.lora2base = lora2base_mapping
        self.base2lora = base2lora_mapping

    def get_weights(self, lora_key):
        # Single entry cache to reuse loaded weights
        if self._cached_weights_name == lora_key:
            return self._cached_weights

        base_key = self.lora2base[lora_key]
        logging.debug("Loading base weight %s for %s", base_key, lora_key)
        if isinstance(base_key, tuple):
            # Attention projection layer requires splitting into one of K, Q, V
            base_key, chunk_idx, n_chunks = base_key
            shape = self.shapes[base_key]
            entries = shape[0]
            chunk_len = entries // n_chunks
            if chunk_len * n_chunks != entries:
                raise ValueError(
                    f"{base_key} shape={tuple(shape)} is not divisible in {n_chunks} chunks"
                )
            slice = self.fd.get_slice(base_key)
            W = slice[chunk_idx * chunk_len : (chunk_idx + 1) * chunk_len]
        else:
            W = self.fd.get_tensor(base_key)

        self._cached_weights_name = lora_key
        self._cached_weights = W
        return W

    @cached("frobenius_norms")
    def frobenius_norm(self, layer, dtype=torch.float32, **kwargs):
        weights = self.get_weights(layer).to(dtype=dtype, **kwargs)
        return (
            torch.linalg.matrix_norm(weights.flatten(start_dim=1), ord="fro")
            .cpu()
            .item()
        )

    @cached("spectral_norms")
    def spectral_norm(self, layer, niter=64, dtype=torch.float32, **kwargs):
        weights = self.get_weights(layer).to(dtype=dtype, **kwargs)
        return (
            torch.svd_lowrank(weights.flatten(start_dim=1), q=1, niter=niter)[1][0]
            .cpu()
            .item()
        )


class LoRADict:
    def __init__(self, lora_path, **to_args):
        self.path = lora_path = Path(lora_path)
        self.lora_fd = lora_fd = safetensors.safe_open(
            lora_path, framework="pt", device=str(to_args.get("device", "cpu"))
        )
        self.to_args = to_args
        self.keys = set(lora_fd.keys())

    def __getitem__(self, name):
        lora_fd = self.lora_fd
        to_kwargs = self.to_args
        alpha = lora_fd.get_tensor(f"{name}.alpha").item()
        down = lora_fd.get_tensor(f"{name}.lora_down.weight").to(**to_kwargs)
        up = lora_fd.get_tensor(f"{name}.lora_up.weight").to(**to_kwargs)
        dora_scale = None
        if f"{name}.dora_scale" in self.keys:
            # FIXME: .to(**to_kwargs) once we take care of the normalization
            dora_scale = lora_fd.get_tensor(f"{name}.dora_scale")
        return alpha, up, down, dora_scale

    @property
    def name(self):
        return self.path.stem

    def metadata(self):
        return self.lora_fd.metadata()


class ConcatLoRAsDict:
    """Load multiple LoRAs, and for each layer, concatenate the weights along the `dim` dimension"""

    def __init__(self, lora_paths_and_weights, **to_args):
        self.loras_and_weights = [
            (LoRADict(lora_path, **to_args), w)
            for lora_path, w in lora_paths_and_weights
        ]
        self.keys = set.union(*(lora.keys for lora, _ in self.loras_and_weights))

    def __getitem__(self, name):
        alpha_name = f"{name}.alpha"
        weights, alphas, ups, downs = zip(
            *[
                (w, *lora[name])
                for lora, w in self.loras_and_weights
                if alpha_name in lora.keys
            ]
        )
        weights = torch.tensor(weights)
        alphas = torch.tensor(alphas)
        dims = torch.tensor([down.size(0) for down in downs])
        sum_dims = torch.sum(dims)
        output_alpha = sum_dims * torch.prod(
            (weights * alphas / dims) ** (dims / sum_dims)
        )
        # geometric average of rescale_factors weighted by dims is 1
        rescale_factors = (weights * alphas * sum_dims) / (output_alpha * dims)
        assert torch.allclose(
            torch.tensor(1.0),
            (rescale_factors.log() * dims).sum().exp(),
            rtol=1e-3,
            atol=1e-3,
        )
        assert torch.allclose(
            rescale_factors * output_alpha / dims.sum(), weights * alphas / dims
        )
        # Halves the factors for rescaling both up and down
        rescale_factors = rescale_factors.sqrt().to(
            dtype=downs[0].dtype, device=downs[0].device
        )

        ups = [w * up for w, up in zip(rescale_factors, ups)]
        up = torch.cat(ups, dim=1)
        downs = [w * down for w, down in zip(rescale_factors, downs)]
        down = torch.cat(downs, dim=0)

        return output_alpha, up, down

    @property
    def name(self):
        names = defaultdict(list)
        for lora, w in self.loras_and_weights:
            name = RE_NAME_SPLIT(lora.name, 0)[0]
            names[name].append(w)
        res = []
        for name, weights in names.items():
            if any(abs(w - 1.0) > 1e-6 for w in weights):
                weights = "+".join(f"{w:.2f}" for w in weights)
                name = f"{name}({weights})"
            res.append(name)
        return "+".join(res)

    @property
    def path(self):
        return self.loras_and_weights[0][0].path

    def metadata(self):
        # FIXME: what can we do here?
        # metadata is so messy I don't want to touch it
        return self.loras_and_weights[0][0].metadata()


class PairedLoraModel:
    """A class for pairing LoRA layers with their base layers"""

    def __init__(self, lora_dict: LoRADict, checkpoint: BaseCheckpoint):
        self.lora_dict = lora_dict
        self.checkpoint = checkpoint

        # Get all unique LoRA layer names from the LoRA file (e.g., "lora_unet_down_blocks_0_attentions_0_proj_in")
        lora_file_keys_alpha_stripped = set()
        for (
            k
        ) in lora_dict.keys:  # k includes suffixes like ".alpha", ".lora_down.weight"
            if k.endswith(".alpha"):
                lora_file_keys_alpha_stripped.add(k.removesuffix(".alpha"))

        # This will map: LoRA key *as found in the file* -> base_info from checkpoint
        # base_info is 'base_model_key_str' or ('base_model_key_str', chunk_idx, num_chunks)
        self.lora2base = {}

        # Iterate through LoRA keys found in the LoRA file
        for lora_key_in_file in lora_file_keys_alpha_stripped:
            if lora_key_in_file in checkpoint.lora2base:
                # This LoRA key from the file is a known name/alias/part_name to the checkpoint's mapper
                base_info = checkpoint.lora2base[lora_key_in_file]
                self.lora2base[lora_key_in_file] = base_info
            else:
                # This LoRA key from the file is not recognized by the checkpoint's comprehensive mapper.
                logging.warning(
                    f"LoRA key %r found in file but is not mapped to any base layer "
                    f"by the checkpoint's key_mapper.",
                    lora_key_in_file,
                )

        # Logging for unmapped base layers (checkpoint layers that weren't covered by any LoRA in the file)
        # Collect all unique base_infos that the checkpoint expects LoRAs for.
        expected_base_infos_in_checkpoint = set(checkpoint.lora2base.values())

        # Collect all base_infos that were successfully mapped using keys from the LoRA file.
        mapped_base_infos_from_file = set(self.lora2base.values())

        unmapped_base_infos = (
            expected_base_infos_in_checkpoint - mapped_base_infos_from_file
        )

        if unmapped_base_infos:
            # For logging, create a reverse map from base_info to one of its expected LoRA names.
            # This is just to provide a helpful "expected LoRA key" in the log.
            base_info_to_one_lora_key_for_log = {}
            for ckpt_lora_key, ckpt_base_info in checkpoint.lora2base.items():
                if (
                    ckpt_base_info not in base_info_to_one_lora_key_for_log
                ):  # Keep first one found
                    base_info_to_one_lora_key_for_log[ckpt_base_info] = ckpt_lora_key

            for base_info in sorted(
                list(unmapped_base_infos), key=str
            ):  # Sort for consistent logging
                base_key_str_for_log = (
                    base_info[0] if isinstance(base_info, tuple) else base_info
                )
                shape = checkpoint.shapes[base_key_str_for_log]
                # Get one representative expected LoRA key for this base_info
                representative_lora_key = base_info_to_one_lora_key_for_log.get(
                    base_info, "unknown_expected_key"
                )

                logging.info(
                    "No LoRA layer found in file for base layer %r %s (e.g., an expected LoRA key was: %r)",
                    base_key_str_for_log,
                    tuple(shape),
                    representative_lora_key,
                )

        num_alpha_keys_in_file = len(lora_file_keys_alpha_stripped)
        logging.info(
            f"Mapped {len(self.lora2base)} LoRA layers from file to base layers "
            f"(out of {num_alpha_keys_in_file} LoRA alpha keys in file)."
        )

    def keys(self):
        return self.lora2base.keys()

    def decompose_layer(
        self, lora_key, **kwargs
    ):  # lora_key here is a key from the LoRA file
        return DecomposedLoRA(self.lora_dict, lora_key, **kwargs)


class DecomposedLoRA:
    "LoRA layer decomposed using SVD"

    def __init__(self, lora_dict: LoRADict, name, **kwargs):
        self.name = name
        alpha, up, down, dora_scale = lora_dict[name]
        self.input_shape = down.shape[1:]
        self.U, S, self.Vh = fast_decompose(up, down)
        self.alpha_factor = alpha_factor = alpha / down.shape[0]
        self.S = S * alpha_factor
        self.dora_scale = dora_scale

    @property
    def dim(self):
        return self.S.shape[0]

    @property
    def alpha(self):
        return self.alpha_factor * self.dim

    def dim_size(self, element_size=2):
        return element_size * (self.U.shape[0] + self.Vh.shape[1])

    def statedict(self, mask=None, rescale=1.0, **kwargs):
        S = self.S
        Vh = self.Vh
        U = self.U
        if mask is not None:
            S = S[mask]
            Vh = Vh[mask]
            U = U[:, mask]
        dim = S.shape[0]
        if dim == 0:
            return {}

        name = self.name
        alpha_factor = self.alpha_factor
        input_shape = self.input_shape

        S_sqrt = torch.sqrt(S * (rescale / alpha_factor))
        down = (Vh * S_sqrt.unsqueeze(1)).view(dim, *input_shape)
        up = (U * S_sqrt).view(*U.shape, *[1] * (len(input_shape) - 1))
        alpha = torch.scalar_tensor(
            alpha_factor * dim, dtype=down.dtype, device=down.device
        )

        d = {
            f"{name}.alpha": alpha,
            f"{name}.lora_down.weight": down,
            f"{name}.lora_up.weight": up,
        }
        if self.dora_scale is not None:
            d[f"{name}.dora_scale"] = self.dora_scale
        if kwargs:
            d = {k: v.to(**kwargs) for k, v in d.items()}
        return d

    def compute_subspace_scales(self, W_base):
        if self.U.shape[0] != W_base.shape[0] or self.input_shape != W_base.shape[1:]:
            raise ValueError(
                f"Shape mismatch: lora is {tuple(self.shape)} while base is {tuple(W_base.shape)}"
            )
        W_base = W_base.flatten(start_dim=1).to(
            device=self.S.device, dtype=self.S.dtype
        )
        return torch.linalg.vecdot(self.Vh @ W_base.T, self.U.T)

    def to(self, **kwargs):
        self.Vh = self.Vh.to(**kwargs)
        self.S = self.S.to(**kwargs)
        self.U = self.U.to(**kwargs)
        return self
