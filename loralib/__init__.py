from collections import defaultdict
import re
from pathlib import Path
import logging

import torch
import safetensors.torch

from .utils import cached, JsonCache
from .num_utils import fast_decompose
from .sdxl_mapper import get_sdxl_lora_keys


__all__ = ["BaseCheckpoint", "PairedLoraModel", "JsonCache"]

RE_NAME_SPLIT = re.compile(r"[\-_ ]").split


class BaseCheckpoint:
    """A class for indexing checkpoints weights by LoRA layer names"""

    def __init__(self, path, key_mapper=get_sdxl_lora_keys, cache=None):
        self.path = path = Path(path)
        self.fd = fd = safetensors.safe_open(path, framework="pt")
        self._cache = {} if cache is None else cache[str(path.resolve())]
        self._cached_weights_name = None

        base_keys = fd.keys()
        self.shapes = shapes = {}
        self.base2lora = base2lora = {}
        self.lora2base = lora2base = {}
        for base_key in base_keys:
            if base_key.startswith("first_stage_model."):
                continue  # ignore VAE
            shapes[base_key] = shape = fd.get_slice(base_key).get_shape()
            if not base_key.endswith("weight") or len(shape) < 2:
                continue
            lora_layer_keys = key_mapper(base_key)
            if lora_layer_keys is None:
                continue

            base2lora[base_key] = lora_layer_keys
            if isinstance(lora_layer_keys, list):
                for i, lora_layer_key in enumerate(lora_layer_keys):
                    lora2base[lora_layer_key] = (base_key, i, len(lora_layer_keys))
            else:
                lora2base[lora_layer_keys] = base_key

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

        lora_keys = lora_dict.keys
        self.lora2base = lora2base = {}
        for lora_key, base_key in checkpoint.lora2base.items():
            if f"{lora_key}.alpha" in lora_keys:
                lora2base[lora_key] = base_key
            else:
                if isinstance(base_key, tuple):
                    base_key = base_key[0]
                shape = checkpoint.shapes[base_key]
                logging.info(
                    "No LoRA layer for %r %s, expected LoRA key: %r",
                    base_key,
                    tuple(shape),
                    lora_key,
                )

        # Checks that all LoRA layers have been mapped
        for lora_layer_keys in lora_keys:
            lora_layer_keys = (
                lora_layer_keys.removesuffix(".alpha")
                .removesuffix(".lora_down.weight")
                .removesuffix(".lora_up.weight")
                .removesuffix(".dora_scale")
            )
            if lora_layer_keys not in lora2base:
                raise ValueError(f"Target layer not found for LoRA {lora_layer_keys}")

    def keys(self):
        return self.lora2base.keys()

    def decompose_layer(self, lora_key, **kwargs):
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
