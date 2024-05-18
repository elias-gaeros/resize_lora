from pathlib import Path
import logging

import torch as pt
import safetensors.torch
import json

UNET_PREFIX = "model.diffusion_model."
TE_PREFIXES = [
    "conditioner.embedders.0.transformer.text_model.encoder.layers.",
    "conditioner.embedders.1.model.transformer.resblocks.",
]
LORA_UNET_PREFIX = "lora_unet_"
LORA_TE_PREFIXES = [
    "lora_te1_text_model_encoder_layers_",
    "lora_te2_text_model_encoder_layers_",
]


def get_sdxl_lora_keys(base_key):
    layer_name = base_key.removesuffix(".weight")

    if layer_name.startswith(UNET_PREFIX):
        return LORA_UNET_PREFIX + layer_name.removeprefix(UNET_PREFIX).replace(".", "_")
    lora_keys = None
    for te_prefix, lora_te_prefix in zip(TE_PREFIXES, LORA_TE_PREFIXES):
        if layer_name.startswith(te_prefix):
            assert lora_keys is None
            layer_name = lora_te_prefix + layer_name.removeprefix(te_prefix).replace(
                ".", "_"
            )

            if te_prefix is TE_PREFIXES[0]:
                lora_keys = layer_name  # CLIP L is easy
            else:  # CLIP G
                if "attn_in_proj_weight" in layer_name:
                    lora_keys = layer_name.replace("_attn_in_proj_weight", "_self_attn")
                    lora_keys = [
                        f"{lora_keys}_{chunk_name}_proj" for chunk_name in "kqv"
                    ]
                elif layer_name.endswith("_attn_out_proj"):
                    lora_keys = layer_name.replace(
                        "_attn_out_proj", "_self_attn_out_proj"
                    )
                elif "_ln_" in layer_name:
                    lora_keys = layer_name.replace("_ln_", "_layer_norm")
                elif "_mlp_" in layer_name:
                    lora_keys = layer_name.replace("_c_fc", "_fc1").replace(
                        "_c_proj", "_fc2"
                    )
    return lora_keys


class BaseCheckpoint:
    def __init__(self, path, key_mapper=get_sdxl_lora_keys, cache=None):
        self.path = Path(path)
        self.fd = fd = safetensors.safe_open(path, framework="pt")
        self.spectral_norms_cache = {}
        if cache is not None:
            self.spectral_norms_cache = cache.get(path).setdefault(
                "spectral_norms", self.spectral_norms_cache
            )

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
        base_key = self.lora2base[lora_key]
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
            return slice[chunk_idx * chunk_len : (chunk_idx + 1) * chunk_len]
        else:
            return self.fd.get_tensor(base_key)

    def spectral_norm(self, layer, weights=None, niter=64, dtype=pt.float32, **kwargs):
        cache = self.spectral_norms_cache
        sn = cache.get(layer)
        if sn is not None:
            return sn
        if weights is None:
            weights = self.get_weights(layer).to(dtype=dtype, **kwargs)
        sn = pt.svd_lowrank(weights.flatten(start_dim=1), q=1, niter=niter)[1][0].cpu().item()
        cache[layer] = sn
        return sn


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


def fast_decompose(up, down):
    Ud, Sd, Vhd = pt.linalg.svd(down.flatten(start_dim=1), full_matrices=False)
    Uu, Su, Vhu = pt.linalg.svd(up.flatten(start_dim=1), full_matrices=False)
    Uc, Sc, Vhc = pt.linalg.svd((Vhu * Su.unsqueeze(1)) @ (Ud * Sd))
    U = Uu @ Uc
    Vh = Vhc @ Vhd
    return U, Sc, Vh


def load_lora_layer(lora_file, name, **to_kwargs):
    alpha = lora_file.get_tensor(f"{name}.alpha").item()
    down = lora_file.get_tensor(f"{name}.lora_down.weight").to(**to_kwargs)
    up = lora_file.get_tensor(f"{name}.lora_up.weight").to(**to_kwargs)
    return alpha, up, down


def outer_cosine_sim(U1, U2):
    U1n = U1 / pt.linalg.norm(U1, dim=0)
    U2n = U2 / pt.linalg.norm(U2, dim=0)
    return U1n.T @ U2n


class DecomposedLoRA:
    "Decomposed LoRA layer"

    def __init__(self, lora_fd, name, **kwargs):
        self.name = name
        alpha, up, down = load_lora_layer(lora_fd, name, **kwargs)
        self.input_shape = down.shape[1:]
        self.U, S, self.Vh = fast_decompose(up, down)
        self.scale = scale = alpha / down.shape[0]
        self.S = S * scale

    @property
    def dim(self):
        return self.S.shape[0]

    @property
    def alpha(self):
        return self.scale * self.dim

    def dim_size(self, element_size=2):
        return element_size * (self.U.shape[0] + self.Vh.shape[1])

    def statedict(self, mask=None, **kwargs):
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
        scale = self.scale
        input_shape = self.input_shape

        S_sqrt = pt.sqrt(S * (1.0 / scale))
        down = (Vh * S_sqrt.unsqueeze(1)).view(dim, *input_shape)
        up = (U * S_sqrt).view(*U.shape, *[1] * (len(input_shape) - 1))
        alpha = pt.scalar_tensor(scale * dim, dtype=down.dtype, device=down.device)

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
        return pt.linalg.vecdot(self.Vh @ W_base.T, self.U.T)

    def to(self, **kwargs):
        self.Vh = self.Vh.to(**kwargs)
        self.S = self.S.to(**kwargs)
        self.U = self.U.to(**kwargs)
        return self


class JsonCache:
    def __init__(self, fp):
        self.fp = Path(fp)
        self.cache = {}
        self.load()

    def load(self):
        if self.fp.exists():
            logging.info("Loading %s", self.fp)
            with open(self.fp, "rt") as fd:
                self.cache = json.load(fd)

    def save(self, discard=False):
        if not self.cache:
            return
        with open(self.fp, "wt") as fd:
            json.dump(self.cache, fd)
        if discard:  # avoid double save
            self.cache = False

    def get(self, model_path):
        model_path = Path(model_path).resolve()
        return self.cache.setdefault(str(model_path), {})

    def __del__(self):
        self.save(discard=True)
