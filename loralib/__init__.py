from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import math
import re
from pathlib import Path

import torch
import safetensors.torch

from .utils import cached, JsonCache
from .num_utils import fast_decompose
from .keymap import Target, build_keymap, read_shapes
from .merge import core_space, ties, tucker2, tucker_spectrum


__all__ = ["BaseCheckpoint", "PairedLoraModel", "JsonCache"]

RE_NAME_SPLIT = re.compile(r"[\-_ ]").split

# (up, down) suffixes of the low-rank formats ComfyUI loads, in its lookup order.
LORA_SUFFIXES = [
    (".lora_up.weight", ".lora_down.weight"),  # kohya, LyCORIS
    ("_lora.up.weight", "_lora.down.weight"),  # diffusers attention processors
    (".lora_B.weight", ".lora_A.weight"),  # PEFT
    (".lora.up.weight", ".lora.down.weight"),  # diffusers
    (".lora_B", ".lora_A"),  # Mochi
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),  # transformers
    (".lora_B.default.weight", ".lora_A.default.weight"),  # PEFT, Qwen-Image
]
# A low-rank layer with one of these is not a plain product of two matrices.
NOT_RESIZABLE = (".lora_mid.weight", ".reshape_weight")
MAX_LOGGED_NAMES = 8


@dataclass(frozen=True)
class LayerFormat:
    """Tensor names of one low-rank layer: `{name}{up}`, `{name}{down}`, `{name}.alpha`."""

    up: str
    down: str
    alpha: bool


def find_lora_layers(keys):
    """Module name -> LayerFormat, for the plain low-rank layers among `keys`."""
    layers = {}
    for key in keys:
        for up, down in LORA_SUFFIXES:
            if key.endswith(up):
                name = key[: -len(up)]
                if name + down in keys and not any(
                    name + s in keys for s in NOT_RESIZABLE
                ):
                    layers[name] = LayerFormat(up, down, f"{name}.alpha" in keys)
                break
    return layers


def _names_for_log(names):
    names = sorted(names)
    more = len(names) - MAX_LOGGED_NAMES
    return ", ".join(names[:MAX_LOGGED_NAMES]) + (
        f" and {more} more" if more > 0 else ""
    )


class BaseCheckpoint:
    """Base-model matrices, indexed by the adapter module names ComfyUI accepts.

    `path` names one or more comma-separated files that make up the base model, such
    as a Flux diffusion model and its text encoders. Each is a safetensors file, or a
    JSON copy of its header for name mapping only.
    """

    def __init__(self, path, cache=None):
        self.paths = [Path(p) for p in str(path).split(",")]
        self.path = self.paths[0]
        self.shapes = [read_shapes(p) for p in self.paths]
        self.keymap = build_keymap(*self.shapes)
        self.fds = [
            None if p.suffix == ".json" else safetensors.safe_open(p, framework="pt")
            for p in self.paths
        ]
        cache_key = ",".join(str(p.resolve()) for p in self.paths)
        self._cache = {} if cache is None else cache[cache_key]
        self._cached_target = None

    def target(self, lora_name):
        """Target of an adapter module name, or None."""
        if lora_name.startswith("lora_unet__"):  # Wan Fun LoRAs
            lora_name = "lora_unet_" + lora_name[len("lora_unet__") :]
        return self.keymap.get(lora_name)

    def get_weights(self, target: Target):
        # Single entry cache to reuse loaded weights
        if self._cached_target == target:
            return self._cached_weights

        logging.debug("Loading base weight %s", target)
        fd = self.fds[target.file]
        if target.start is None:
            W = fd.get_tensor(target.key)
        else:
            W = fd.get_slice(target.key)[target.start : target.stop]
        # ComfyUI fp8 files store weight / scale: "scaled fp8" names the scale
        # `scale_weight`, the newer quantized format `weight_scale`.
        shapes = self.shapes[target.file]
        for suffix in (".scale_weight", ".weight_scale"):
            scale_key = target.key.removesuffix(".weight") + suffix
            if scale_key in shapes:
                if math.prod(shapes[scale_key]) != 1:
                    raise ValueError(
                        f"{target.key} is quantized with block-wise scales, which are "
                        "not supported; use an unquantized or fp8 base model"
                    )
                scale = fd.get_tensor(scale_key).to(torch.float32)
                W = W.to(torch.float32) * scale
                break
        if target.transpose:
            W = W.T

        self._cached_target = target
        self._cached_weights = W
        return W

    @cached("frobenius_norms")
    def frobenius_norm(self, target, dtype=torch.float32, **kwargs):
        weights = self.get_weights(target).to(dtype=dtype, **kwargs)
        return (
            torch.linalg.matrix_norm(weights.flatten(start_dim=1), ord="fro")
            .cpu()
            .item()
        )

    @cached("spectral_norms")
    def spectral_norm(self, target, niter=64, dtype=torch.float32, **kwargs):
        weights = self.get_weights(target).to(dtype=dtype, **kwargs)
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
        self.layers = find_lora_layers(self.keys)

    def __getitem__(self, name):
        lora_fd = self.lora_fd
        to_kwargs = self.to_args
        fmt = self.layers[name]
        alpha = lora_fd.get_tensor(f"{name}.alpha").item() if fmt.alpha else None
        down = lora_fd.get_tensor(f"{name}{fmt.down}").to(**to_kwargs)
        up = lora_fd.get_tensor(f"{name}{fmt.up}").to(**to_kwargs)
        dora_scale = None
        if f"{name}.dora_scale" in self.keys:
            # FIXME: .to(**to_kwargs) once we take care of the normalization
            dora_scale = lora_fd.get_tensor(f"{name}.dora_scale")
        return alpha, up, down, dora_scale

    def shape(self, key):
        return tuple(self.lora_fd.get_slice(key).get_shape())

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
        # The merged layer stores an alpha even where the members do not.
        self.layers = {}
        for lora, _ in self.loras_and_weights:
            for name, fmt in lora.layers.items():
                self.layers.setdefault(name, LayerFormat(fmt.up, fmt.down, True))
        if any(k.endswith(".dora_scale") for k in self.keys):
            logging.warning("Merging drops the DoRA magnitudes (dora_scale)")

    def __getitem__(self, name):
        weights, alphas, ups, downs, _ = zip(
            *[
                (w, *lora[name])
                for lora, w in self.loras_and_weights
                if name in lora.layers
            ]
        )
        weights = torch.tensor(weights)
        dims = torch.tensor([down.size(0) for down in downs])
        # A missing alpha means a scale of 1, i.e. alpha = rank.
        alphas = torch.tensor(
            [float(d) if a is None else a for a, d in zip(alphas, dims.tolist())]
        )
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

        return output_alpha.item(), up, down, None

    def shape(self, key):
        for lora, _ in self.loras_and_weights:
            if key in lora.keys:
                return lora.shape(key)
        raise KeyError(key)

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

    def merge_params(self):
        return {
            "method": "sum",
            "loras": [[lora.name, w] for lora, w in self.loras_and_weights],
        }

    def metadata(self):
        # FIXME: what can we do here?
        # metadata is so messy I don't want to touch it
        metadata = self.loras_and_weights[0][0].metadata() or {}
        metadata["merge_params"] = json.dumps(self.merge_params())
        return metadata


class CtMLoRAsDict(ConcatLoRAsDict):
    """Merge LoRAs by Compress then Merge (CtM; He et al., ICML 2026).

    For each layer, the weighted LoRA products are written in shared orthonormal bases
    (lossless) and decomposed into `CtMLoRA` components, whose norms the recipe scores
    to choose the layer's rank k. A Tucker-2 decomposition then selects the
    k-dimensional output and input subspaces that keep the most energy of all the
    LoRAs, and the LoRAs' coordinates in these subspaces are summed, or TIES-merged
    with density `ties`. When selecting the subspaces, `beta` moves each LoRA's norm
    towards the mean norm: 1 keeps the norms, 0 equalizes them. A layer that only one
    LoRA has, or that the others leave at zero, is resized like that LoRA alone.
    """

    def __init__(self, lora_paths_and_weights, beta=0.25, ties=None, **to_args):
        super().__init__(lora_paths_and_weights, **to_args)
        self.beta, self.ties = beta, ties

    def decompose(self, name, target=None):
        members, ups, downs = [], [], []
        for i, (lora, w) in enumerate(self.loras_and_weights):
            if name in lora.layers:
                alpha, up, down, _ = lora[name]
                scale = w if alpha is None else w * alpha / down.shape[0]
                members.append(i)
                ups.append(scale * up.flatten(start_dim=1))
                downs.append(down.flatten(start_dim=1))
        input_shape = down.shape[1:]

        Q_up, Q_down, cores = core_space(ups, downs)
        # The cores are small: their many SVDs run faster on the CPU than a GPU, and
        # float64 lets Tucker-2 converge tightly.
        cores = cores.to(device="cpu", dtype=torch.float64)
        norms = torch.linalg.matrix_norm(cores)
        nonzero = norms > 1e-6
        if not nonzero.any():
            return DecomposedLoRA(self, name, target)
        # Untrained layers (zero LoRA product) take no part in the merge.
        cores, norms = cores[nonzero], norms[nonzero]
        members = [i for i, keep in zip(members, nonzero.tolist()) if keep]

        P_up, P_down, S, up_cols, down_cols = tucker_spectrum(cores)
        target_norms = self.beta * norms + (1 - self.beta) * norms.mean()
        like_Q = dict(device=Q_up.device, dtype=Q_up.dtype)
        return CtMLoRA(
            name,
            target,
            self.layers[name],
            input_shape,
            basis_up=Q_up @ P_up.to(**like_Q),
            basis_down=Q_down @ P_down.to(**like_Q),
            S=S.to(dtype=Q_up.dtype),
            paired=(up_cols, down_cols),
            cores=P_up.T @ cores @ P_down,
            basis_scales=target_norms / norms,
            ties=self.ties,
            members=[self.loras_and_weights[i][0].name for i in members],
            rank_sum=sum(d.shape[0] for d in downs),
        )

    def log_subspace_error(self, layers):
        """Log, for each LoRA, the norm its layers lose to the CtM subspaces."""
        sums = {lora.name: [0.0, 0.0] for lora, _ in self.loras_and_weights}
        for layer in layers:
            for name, total, lost in getattr(layer, "errors_sq", []):
                sums[name][0] += total
                sums[name][1] += lost
        logging.info(
            "CtM subspace error over all layers: %s",
            ", ".join(
                f"{name} {100 * math.sqrt(lost / total):.2f}%"
                for name, (total, lost) in sums.items()
                if total > 0
            ),
        )

    @property
    def name(self):
        tag = ["ctm", f"b{self.beta:g}"]
        if self.ties is not None:
            tag.append(f"ties{self.ties:g}")
        return f"{super().name}_{'-'.join(tag)}"

    def merge_params(self):
        return super().merge_params() | {
            "method": "ctm",
            "beta": self.beta,
            "ties": self.ties,
        }


class PairedLoraModel:
    """Pairs the low-rank layers of a LoRA with the base matrices they modify."""

    def __init__(self, lora_dict: LoRADict, checkpoint: BaseCheckpoint):
        self.lora_dict = lora_dict
        self.checkpoint = checkpoint

        # LoRA module name -> Target in the checkpoint
        self.targets = {}
        self.unmapped, self.mismatched = [], []
        for name, fmt in lora_dict.layers.items():
            target = checkpoint.target(name)
            if target is None:
                self.unmapped.append(name)
                continue
            out_dim = lora_dict.shape(name + fmt.up)[0]
            in_shape = lora_dict.shape(name + fmt.down)[1:]
            base_shape = target.shape
            if base_shape[0] != out_dim or math.prod(base_shape[1:]) != math.prod(
                in_shape
            ):
                self.mismatched.append(
                    f"{name} {(out_dim, *in_shape)} vs {target} {base_shape}"
                )
                continue
            self.targets[name] = target

        if self.mismatched:
            logging.warning(
                "%d LoRA layers do not have the shape of their base layer: %s",
                len(self.mismatched),
                _names_for_log(self.mismatched),
            )
        # ComfyUI keeps one patch per target, e.g. when merged LoRAs name a layer in
        # kohya and PEFT style. Slices of a fused weight are distinct targets.
        names_by_target = defaultdict(list)
        for name, target in self.targets.items():
            names_by_target[target].append(name)
        shared = [
            " and ".join(sorted(names))
            for names in names_by_target.values()
            if len(names) > 1
        ]
        if shared:
            logging.warning(
                "%d base layers are modified by several LoRA layers, of which ComfyUI "
                "applies only one: %s",
                len(shared),
                _names_for_log(shared),
            )

        # Tensors outside the resizable layers (LyCORIS .diff, LoHa/LoKr factors,
        # unmapped or mismatched layers) are copied to the output unchanged.
        resized_keys = set()
        for name in self.targets:
            fmt = lora_dict.layers[name]
            resized_keys.update(
                {name + fmt.up, name + fmt.down, f"{name}.alpha", f"{name}.dora_scale"}
            )
        unresized_keys = sorted(lora_dict.keys - resized_keys)
        # A merge has no defined way to combine arbitrary tensors, so it drops them.
        self.passthrough_keys = (
            unresized_keys if isinstance(lora_dict, LoRADict) else []
        )
        if unresized_keys:
            logging.warning(
                "%d LoRA tensors cannot be resized and are %s: %s",
                len(unresized_keys),
                "copied unchanged" if self.passthrough_keys else "dropped",
                _names_for_log(unresized_keys),
            )

        uncovered = sorted(
            {(t.file, t.key) for t in checkpoint.keymap.values()}
            - {(t.file, t.key) for t in self.targets.values()}
        )
        logging.info(
            "Mapped %d of %d LoRA layers to base layers; %d base layers have no LoRA layer.",
            len(self.targets),
            len(lora_dict.layers),
            len(uncovered),
        )
        logging.debug(
            "Base layers without a LoRA layer: %s", ", ".join(k for _, k in uncovered)
        )

    def keys(self):
        return self.targets.keys()

    def passthrough_tensors(self):
        return {k: self.lora_dict.lora_fd.get_tensor(k) for k in self.passthrough_keys}

    def decompose_layer(self, lora_key, **kwargs):
        if isinstance(self.lora_dict, CtMLoRAsDict):
            return self.lora_dict.decompose(lora_key, self.targets[lora_key])
        return DecomposedLoRA(
            self.lora_dict, lora_key, self.targets[lora_key], **kwargs
        )


class DecomposedLoRA:
    "LoRA layer decomposed using SVD"

    def __init__(self, lora_dict: LoRADict, name, target=None, **kwargs):
        self.name = name
        self.target = target
        self.format = lora_dict.layers[name]
        alpha, up, down, dora_scale = lora_dict[name]
        self.input_shape = down.shape[1:]
        self.U, S, self.Vh = fast_decompose(up, down)
        # Without an alpha the scale is 1, as in ComfyUI.
        self.alpha_factor = alpha_factor = (
            1.0 if alpha is None else alpha / down.shape[0]
        )
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
        return self._factors_statedict(U, S, Vh, rescale, **kwargs)

    def _factors_statedict(self, U, S, Vh, rescale, **kwargs):
        """The tensors of the LoRA layer U @ diag(S) @ Vh, multiplied by `rescale`."""
        dim = S.shape[0]
        if dim == 0:
            return {}

        name = self.name
        fmt = self.format
        alpha_factor = self.alpha_factor
        input_shape = self.input_shape

        S_sqrt = torch.sqrt(S * (rescale / alpha_factor))
        down = (Vh * S_sqrt.unsqueeze(1)).view(dim, *input_shape)
        up = (U * S_sqrt).view(*U.shape, *[1] * (len(input_shape) - 1))

        d = {
            f"{name}{fmt.down}": down,
            f"{name}{fmt.up}": up,
        }
        if fmt.alpha:
            d[f"{name}.alpha"] = torch.scalar_tensor(
                alpha_factor * dim, dtype=down.dtype, device=down.device
            )
        if self.dora_scale is not None:
            d[f"{name}.dora_scale"] = self.dora_scale
        if kwargs:
            d = {k: v.to(**kwargs) for k, v in d.items()}
        return d

    def compute_subspace_scales(self, W_base):
        W_base = W_base.flatten(start_dim=1).to(
            device=self.S.device, dtype=self.S.dtype
        )
        if W_base.shape != (self.U.shape[0], self.Vh.shape[1]):
            raise ValueError(
                f"Shape mismatch: lora is {(self.U.shape[0], self.Vh.shape[1])} "
                f"while base is {tuple(W_base.shape)}"
            )
        return torch.linalg.vecdot(self.Vh @ W_base.T, self.U.T)

    def to(self, **kwargs):
        self.Vh = self.Vh.to(**kwargs)
        self.S = self.S.to(**kwargs)
        self.U = self.U.to(**kwargs)
        return self


class CtMLoRA(DecomposedLoRA):
    """A layer of several LoRAs, decomposed into the components CtM merges.

    basis_up (m, a) and basis_down (n_in, b) hold the higher-order SVD directions, and
    `cores` (T, a, b) each LoRA in these bases. Component i pairs U[:, i] and Vh[i],
    columns `paired` of the bases, and S[i] is its norm from `tucker_spectrum`: recipes
    score S as the singular values of a single LoRA. `statedict` merges the LoRAs into
    as many dimensions as `mask` keeps, but not necessarily the masked ones.
    """

    def __init__(
        self,
        name,
        target,
        fmt,
        input_shape,
        basis_up,
        basis_down,
        S,
        paired,
        cores,
        basis_scales,
        ties,
        members,
        rank_sum,
    ):
        self.name, self.target, self.format = name, target, fmt
        self.input_shape = input_shape
        self.basis_up, self.basis_down, self.S = basis_up, basis_down, S
        self.up_cols, self.down_cols = paired
        self.cores, self.basis_scales = cores, basis_scales
        self.ties, self.members, self.rank_sum = ties, members, rank_sum
        self.alpha_factor = 1.0
        self.dora_scale = None
        # (LoRA name, squared norm, squared norm lost) for each LoRA, by the last statedict
        self.errors_sq = []

    @property
    def U(self):
        return self.basis_up[:, self.up_cols]

    @property
    def Vh(self):
        return self.basis_down[:, self.down_cols].T

    def dim_size(self, element_size=2):
        return element_size * (self.basis_up.shape[0] + self.basis_down.shape[0])

    def to(self, **kwargs):
        self.basis_up = self.basis_up.to(**kwargs)
        self.basis_down = self.basis_down.to(**kwargs)
        self.S = self.S.to(**kwargs)
        return self

    def statedict(self, mask=None, rescale=1.0, **kwargs):
        k = self.dim if mask is None else int(mask.sum())
        total_sq = torch.linalg.matrix_norm(self.cores).square()
        if k == 0:
            self.errors_sq = list(
                zip(self.members, total_sq.tolist(), total_sq.tolist())
            )
            return {}
        # When the cores are not square, the larger side keeps its extra directions,
        # as in tucker_spectrum, so that k = n loses nothing. The rank is still k.
        a, b = self.cores.shape[1:]
        n = min(a, b)
        P_up, P_down = tucker2(
            self.cores * self.basis_scales[:, None, None], (k + a - n, k + b - n)
        )
        small = P_up.T @ self.cores @ P_down
        if self.ties is None or len(small) == 1:
            merged = small.sum(0)
        else:
            merged = ties(small, self.ties)

        lost_sq = (total_sq - torch.linalg.matrix_norm(small).square()).clamp(min=0)
        self.errors_sq = list(zip(self.members, total_sq.tolist(), lost_sq.tolist()))
        logging.debug(
            "ctm dim:%3d->%-3d subspace rle: %s %s",
            self.rank_sum,
            k,
            " ".join(
                f"{name}:{100 * rle:.2f}%"
                for name, rle in zip(self.members, (lost_sq / total_sq).sqrt().tolist())
            ),
            self.name,
        )

        Uc, Sc, Vhc = torch.linalg.svd(merged, full_matrices=False)
        like_basis = dict(device=self.basis_up.device, dtype=self.basis_up.dtype)
        U = self.basis_up @ (P_up @ Uc).to(**like_basis)
        Vh = (Vhc @ P_down.T).to(**like_basis) @ self.basis_down.T
        return self._factors_statedict(U, Sc.to(**like_basis), Vh, rescale, **kwargs)
