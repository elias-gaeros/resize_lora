from pathlib import Path
import logging

import torch
import safetensors.torch

from .utils import cached, JsonCache
from .num_utils import load_lora_layer, fast_decompose, special_ortho_group
from .sdxl_mapper import get_sdxl_lora_keys


__all__ = ["BaseCheckpoint", "PairedLoraModel", "JsonCache"]


class BaseCheckpoint:
    """A class for indexing checkpoints weights by LoRA layer names"""

    def __init__(self, path, key_mapper=get_sdxl_lora_keys, cache=None):
        self.path = Path(path)
        self.fd = fd = safetensors.safe_open(path, framework="pt")
        self._cache = {} if cache is None else cache
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


class PairedLoraModel:
    def __init__(self, lora_path: str | Path, checkpoint: BaseCheckpoint):
        self.lora_path = Path(lora_path)
        self.lora_fd = lora_fd = safetensors.safe_open(lora_path, framework="pt")
        self.checkpoint = checkpoint

        lora_keys = lora_fd.keys()
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
            )
            if lora_layer_keys not in lora2base:
                raise ValueError(f"Target layer not found for LoRA {lora_layer_keys}")

    def keys(self):
        return self.lora2base.keys()

    def decompose_layer(self, lora_key, **kwargs):
        return DecomposedLoRA(self.lora_fd, lora_key, **kwargs)


class DecomposedLoRA:
    "LoRA layer decomposed using SVD"

    def __init__(self, lora_fd, name, **kwargs):
        self.name = name
        alpha, up, down = load_lora_layer(lora_fd, name, **kwargs)
        self.input_shape = down.shape[1:]
        self.U, S, self.Vh = fast_decompose(up, down)
        self.alpha_factor = alpha_factor = alpha / down.shape[0]
        self.S = S * alpha_factor

    @property
    def dim(self):
        return self.S.shape[0]

    @property
    def alpha(self):
        return self.alpha_factor * self.dim

    def dim_size(self, element_size=2):
        return element_size * (self.U.shape[0] + self.Vh.shape[1])

    def statedict(self, mask=None, rescale=1.0, rot=False, **kwargs):
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

        
        if rot is not False and dim > 1:
            if rot is True:
                rot = special_ortho_group(dim)
            std_before = torch.std(torch.cat((down.T,up)), axis=0).std()
            down = rot @ down
            up = up @ rot.T
            std_after = torch.std(torch.cat((down.T,up)), axis=0).std()
            # print(std_before, std_after)

        d = {
            f"{name}.alpha": alpha,
            f"{name}.lora_down.weight": down,
            f"{name}.lora_up.weight": up,
        }
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
