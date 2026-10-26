"""Map adapter layer names to the base-model matrices they modify.

The rules follow ComfyUI's LoRA loader (`comfy/lora.py` and the `*_to_diffusers` tables
in `comfy/utils.py`). They are reimplemented here rather than copied, because ComfyUI is
GPL-3.0 and this project is MIT. ComfyUI loads a LoRA in two steps, and so does this
module, from checkpoint headers alone:

1. `comfy_names()` gives each matrix of a checkpoint file its ComfyUI name. For example
   rows 1280:2560 of `conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight`
   become `clip_g.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight`.
2. `adapter_names()` lists every adapter module name that ComfyUI accepts for each of
   those matrices: ComfyUI's own (`diffusion_model.…`, `text_encoders.…`), kohya
   (`lora_unet_…`, `lora_te1_…`), diffusers/PEFT (`transformer.…`, `unet.…`,
   `text_encoder.…`) and LyCORIS (`lycoris_…`). Diffusers names of fused weights, such
   as Flux `img_attn.qkv`, map to row slices.

`build_keymap()` composes both steps into `adapter module name -> Target`. A base model
can span several files, such as a Flux diffusion model and its text encoders.
"""

import json
import re
import struct
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Target:
    """Rows `start:stop` of tensor `key` of base file number `file`, transposed if
    `transpose`. `stored_shape` is the shape of the whole tensor in the file."""

    key: str
    stored_shape: tuple
    start: int | None = None
    stop: int | None = None
    transpose: bool = False
    file: int = 0

    def __str__(self):
        s = self.key if self.file == 0 else f"{self.file}:{self.key}"
        if self.start is not None:
            s += f"[{self.start}:{self.stop}]"
        if self.transpose:
            s += ".T"
        return s

    @property
    def shape(self):
        shape = list(self.stored_shape)
        if self.start is not None:
            shape[0] = self.stop - self.start
        if self.transpose:
            shape.reverse()
        return tuple(shape)

    def rows(self, start, stop):
        """Rows `start:stop` of this target."""
        offset = self.start or 0
        return replace(self, start=offset + start, stop=offset + stop)


def read_shapes(path):
    """Tensor shapes of a safetensors file, or of a JSON copy of its header."""
    path = Path(path)
    if path.suffix == ".json":
        header = json.loads(path.read_text(encoding="utf-8"))
    else:
        with open(path, "rb") as fd:
            (size,) = struct.unpack("<Q", fd.read(8))
            header = json.loads(fd.read(size))
    header.pop("__metadata__", None)
    return {k: tuple(v["shape"]) for k, v in header.items()}


def build_keymap(*files):
    """Adapter module name -> Target, for a base model made of files with these
    tensor shapes."""
    comfy = {}
    for i, shapes in enumerate(files):
        for name, target in comfy_names(shapes, file=i).items():
            if name in comfy:
                raise ValueError(
                    f"Base files {comfy[name].file} and {i} both hold {name}"
                )
            comfy[name] = target
    return adapter_names(comfy)


# Step 1: checkpoint file names -> ComfyUI names

OPEN_CLIP_LAYER = {
    "attn.out_proj.weight": "self_attn.out_proj.weight",
    "mlp.c_fc.weight": "mlp.fc1.weight",
    "mlp.c_proj.weight": "mlp.fc2.weight",
}


def _hf_clip(name):
    def convert(key, rest, shape):
        if rest.startswith(("encoder.", "embeddings.", "final_layer_norm.")):
            rest = "text_model." + rest  # layout of transformers < 4.0
        if rest.endswith(".weight"):
            return {f"{name}.transformer.{rest}": Target(key, shape)}
        return {}

    return convert


def _open_clip(name):
    def convert(key, rest, shape):
        if rest == "text_projection":  # applied as x @ P, i.e. stored as (in, out)
            return {
                f"{name}.transformer.text_projection.weight": Target(
                    key, shape, transpose=True
                )
            }
        m = re.fullmatch(r"transformer\.resblocks\.(\d+)\.(.+)", rest)
        if m is None:
            return {}
        layer = f"{name}.transformer.text_model.encoder.layers.{m[1]}."
        if m[2] == "attn.in_proj_weight":
            d = shape[0] // 3
            return {
                f"{layer}self_attn.{p}_proj.weight": Target(
                    key, shape, i * d, (i + 1) * d
                )
                for i, p in enumerate("qkv")
            }
        if m[2] in OPEN_CLIP_LAYER:
            return {layer + OPEN_CLIP_LAYER[m[2]]: Target(key, shape)}
        return {}

    return convert


# Checkpoint prefix -> ComfyUI prefix, or a converter for layouts that differ.
CHECKPOINT_PREFIXES = [
    ("model.diffusion_model.", "diffusion_model."),
    ("diffusion_model.", "diffusion_model."),
    ("cond_stage_model.transformer.", _hf_clip("clip_l")),  # SD1.x
    ("cond_stage_model.model.", _open_clip("clip_h")),  # SD2.x
    ("conditioner.embedders.0.transformer.", _hf_clip("clip_l")),  # SDXL
    ("conditioner.embedders.1.model.", _open_clip("clip_g")),  # SDXL
    ("conditioner.embedders.0.model.", _open_clip("clip_g")),  # SDXL refiner
    ("text_encoders.", ""),  # SD3 and Flux all-in-one files
]
# A file with none of these prefixes holds one bare model.
NON_BARE_PREFIXES = tuple(p for p, _ in CHECKPOINT_PREFIXES) + (
    "first_stage_model.",
    "vae.",
    "cond_stage_model.",
    "conditioner.",
)
CLIP_BY_WIDTH = {768: "clip_l", 1280: "clip_g", 1024: "clip_h"}
# ComfyUI names LLM text encoders by model; these widths are the Qwen3 ones it uses
# for Flux.2 Klein and Z-Image. Other LLMs only get names without the wrapper prefix.
LLM_BY_WIDTH = {2560: "qwen3_4b", 4096: "qwen3_8b"}


def _bare_prefix(shapes):
    """ComfyUI prefix of a file holding one bare model."""
    q = shapes.get("text_model.encoder.layers.0.self_attn.q_proj.weight")
    if q is not None:
        return f"{CLIP_BY_WIDTH.get(q[0], 'clip_l')}.transformer."
    if "encoder.block.0.layer.0.SelfAttention.q.weight" in shapes:
        # UMT5 has relative attention biases in every block, T5 only in the first.
        umt5 = "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight"
        return ("umt5xxl" if umt5 in shapes else "t5xxl") + ".transformer."
    embed = shapes.get("model.embed_tokens.weight")
    if embed is not None and "model.layers.0.self_attn.q_proj.weight" in shapes:
        return f"{LLM_BY_WIDTH.get(embed[1], 'llm')}.transformer."
    return "diffusion_model."


def comfy_names(shapes, file=0):
    """ComfyUI name -> Target, for the matrices of a checkpoint file."""
    names = {}
    bare = None
    if not any(k.startswith(NON_BARE_PREFIXES) for k in shapes):
        bare = _bare_prefix(shapes)
    for key, shape in shapes.items():
        if len(shape) < 2:
            continue
        if bare is not None:
            if key.endswith(".weight"):
                names[bare + key] = Target(key, shape)
            continue
        for prefix, to in CHECKPOINT_PREFIXES:
            if key.startswith(prefix):
                rest = key[len(prefix) :]
                if callable(to):
                    names.update(to(key, rest, shape))
                elif key.endswith(".weight"):
                    names[to + rest] = Target(key, shape)
                break
    return {name: replace(t, file=file) for name, t in names.items()}


# Step 2: ComfyUI names -> adapter module names


def adapter_names(comfy):
    """Adapter module name -> Target, for every name ComfyUI accepts.

    An alias claimed by two different targets is dropped.
    """
    out, ambiguous = {}, set()

    def add(alias, target):
        if out.setdefault(alias, target) != target:
            ambiguous.add(alias)

    dit = {
        ckey[len("diffusion_model.") : -len(".weight")]: target
        for ckey, target in comfy.items()
        if ckey.startswith("diffusion_model.")
    }
    for x, target in dit.items():
        u = x.replace(".", "_")
        for alias in (
            f"lora_unet_{u}",
            f"lycoris_{u}",
            f"diffusion_model.{x}",
            x,
            f"transformer.{x}",
            f"base_model.model.{x}",
        ):
            add(alias, target)
        if x.startswith("single_blocks.") and x.endswith(".linear1"):
            hidden = target.shape[1]
            add(f"diffusion_model.{x}_qkv", target.rows(0, 3 * hidden))

    nrb = _unet_res_blocks(dit)
    for x, target in dit.items() if nrb else ():
        d = unet_to_diffusers(x, nrb)
        if d is None:
            continue
        # Attention-processor LoRAs name `attn1.to_q` as `attn1.processor.to_q`.
        processor = d.replace(".to_", ".processor.to_")
        if processor.endswith(".to_out.0"):
            processor = processor[: -len(".0")]
        for alias in (
            f"lora_unet_{d.replace('.', '_')}",
            f"lycoris_{d.replace('.', '_')}",
            d,
            f"unet.{d}",
            processor,
            f"unet.{processor}",
        ):
            add(alias, target)

    for d, x, part in [*_flux_to_diffusers(dit), *_mmdit_to_diffusers(dit)]:
        target = dit[x]
        if part is not None:
            target = target.rows(*part(target.shape))
        u = d.replace(".", "_")
        for alias in (
            d,
            f"transformer.{d}",
            f"base_model.model.{d}",
            f"lora_transformer_{u}",
            f"lycoris_{u}",
        ):
            add(alias, target)

    _text_encoder_names(comfy, add)

    for alias in ambiguous:
        del out[alias]
    return out


def _text_encoder_names(comfy, add):
    heads = {ckey.split(".", 1)[0] for ckey in comfy} - {"diffusion_model"}
    clips = heads & {"clip_l", "clip_g", "clip_h"}
    if clips >= {"clip_l", "clip_g"}:
        kohya = {"clip_l": ["lora_te_", "lora_te1_"], "clip_g": ["lora_te2_"]}
        diffusers = {"clip_l": "text_encoder.", "clip_g": "text_encoder_2."}
    else:
        kohya = {c: ["lora_te_", "lora_te1_", "lora_prior_te_"] for c in clips}
        diffusers = {c: "text_encoder." for c in clips}
    # T5 comes after the CLIP encoders; OneTrainer Flux LoRAs use te2 for clip_l + T5.
    t5_index = 1 + len(clips & {"clip_l", "clip_g"})
    kohya["t5xxl"] = (
        ["lora_te2_", "lora_te3_"] if clips == {"clip_l"} else [f"lora_te{t5_index}_"]
    )

    for ckey, target in comfy.items():
        head, _, rest = ckey.partition(".")
        if head not in heads:
            continue
        stem = ckey[: -len(".weight")]
        rest = rest.removeprefix("transformer.").removesuffix(".weight")
        add(f"text_encoders.{stem}", target)  # ComfyUI's own format
        if head not in clips:  # and without the wrapper of composite encoders
            add(f"text_encoders.transformer.{rest}", target)
        for prefix in kohya.get(head, ()):
            add(prefix + rest.replace(".", "_"), target)
        if head in diffusers:
            add(diffusers[head] + rest, target)
        # kohya names the layers of a lone LLM text encoder after its `model.` path
        if heads == {head} and rest.startswith("model."):
            add("lora_te_" + rest[len("model.") :].replace(".", "_"), target)


# Diffusers names. Each table entry gives the diffusers module, the ComfyUI module and
# an optional function of the ComfyUI matrix shape returning a row range.

RESNET_TO_DIFFUSERS = {
    "in_layers.0": "norm1",
    "in_layers.2": "conv1",
    "emb_layers.1": "time_emb_proj",
    "out_layers.0": "norm2",
    "out_layers.3": "conv2",
    "skip_connection": "conv_shortcut",
}
UNET_TO_DIFFUSERS = {
    "time_embed.0": "time_embedding.linear_1",
    "time_embed.2": "time_embedding.linear_2",
    "label_emb.0.0": "add_embedding.linear_1",
    "label_emb.0.2": "add_embedding.linear_2",
    "input_blocks.0.0": "conv_in",
    "out.0": "conv_norm_out",
    "out.2": "conv_out",
}


def _unet_res_blocks(dit):
    """Res blocks per level of an SD/SDXL UNet, from the first downsampler's index."""
    ops = [
        int(m[1]) for x in dit if (m := re.fullmatch(r"input_blocks\.(\d+)\.0\.op", x))
    ]
    return min(ops) - 1 if ops else None


def unet_to_diffusers(x, nrb):
    """Diffusers name of the SD/SDXL UNet module `x` (ldm name), or None."""
    if x in UNET_TO_DIFFUSERS:
        return UNET_TO_DIFFUSERS[x]
    m = re.fullmatch(
        r"(input_blocks|output_blocks|middle_block)\.(\d+)\.(?:(\d+)\.)?(.+)", x
    )
    if m is None:
        return None
    kind, rest = m[1], m[4]
    if kind == "middle_block":
        if m[3] is not None:
            return None
        block = int(m[2])
        if block == 1:
            return f"mid_block.attentions.0.{rest}"
        sub = RESNET_TO_DIFFUSERS.get(rest)
        return sub and f"mid_block.resnets.{block // 2}.{sub}"
    if m[3] is None:
        return None
    n, layer = int(m[2]), int(m[3])
    if kind == "input_blocks":
        level, j = divmod(n - 1, nrb + 1)
        if j == nrb:
            return f"down_blocks.{level}.downsamplers.0.conv" if rest == "op" else None
        prefix = "down_blocks"
    else:
        level, j = divmod(n, nrb + 1)
        if layer > 0 and rest == "conv":
            return f"up_blocks.{level}.upsamplers.0.conv"
        prefix = "up_blocks"
    if layer == 0:
        sub = RESNET_TO_DIFFUSERS.get(rest)
        return sub and f"{prefix}.{level}.resnets.{j}.{sub}"
    if layer == 1:
        return f"{prefix}.{level}.attentions.{j}.{rest}"
    return None


def _thirds(i):
    return lambda shape: (i * shape[0] // 3, (i + 1) * shape[0] // 3)


def _qkv(names):
    return [(name, _thirds(i)) for i, name in enumerate(names)]


FLUX_DOUBLE = {
    "img_attn.qkv": _qkv(["attn.to_q", "attn.to_k", "attn.to_v"]),
    "txt_attn.qkv": _qkv(["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"]),
    "img_attn.proj": [("attn.to_out.0", None)],
    "txt_attn.proj": [("attn.to_add_out", None)],
    "img_mlp.0": [("ff.net.0.proj", None)],
    "img_mlp.2": [("ff.net.2", None)],
    "txt_mlp.0": [("ff_context.net.0.proj", None)],
    "txt_mlp.2": [("ff_context.net.2", None)],
    "img_mod.lin": [("norm1.linear", None)],
    "txt_mod.lin": [("norm1_context.linear", None)],
}
FLUX_SINGLE = {
    # linear1 rows are [q, k, v, mlp]; its input width is the hidden size.
    "linear1": [
        ("attn.to_q", lambda s: (0, s[1])),
        ("attn.to_k", lambda s: (s[1], 2 * s[1])),
        ("attn.to_v", lambda s: (2 * s[1], 3 * s[1])),
        ("proj_mlp", lambda s: (3 * s[1], s[0])),
    ],
    "linear2": [("proj_out", None)],
    "modulation.lin": [("norm.linear", None)],
}
FLUX_GLOBAL = {
    "img_in": "x_embedder",
    "txt_in": "context_embedder",
    "time_in.in_layer": "time_text_embed.timestep_embedder.linear_1",
    "time_in.out_layer": "time_text_embed.timestep_embedder.linear_2",
    "vector_in.in_layer": "time_text_embed.text_embedder.linear_1",
    "vector_in.out_layer": "time_text_embed.text_embedder.linear_2",
    "guidance_in.in_layer": "time_text_embed.guidance_embedder.linear_1",
    "guidance_in.out_layer": "time_text_embed.guidance_embedder.linear_2",
    "final_layer.adaLN_modulation.1": "norm_out.linear",  # scale/shift halves swapped
    "final_layer.linear": "proj_out",
}


def _block_table(dit, pattern, blocks, table, globals_):
    for x in dit:
        if x in globals_:
            yield globals_[x], x, None
        m = re.fullmatch(pattern, x)
        if m and m[3] in table:
            for d, part in table[m[3]]:
                yield f"{blocks[m[1]]}.{m[2]}.{d}", x, part


def _flux_to_diffusers(dit):
    """Diffusers names of Flux.1 and Chroma. Flux.2 uses different fused layers."""
    if (
        "double_blocks.0.img_attn.qkv" not in dit
        or "double_stream_modulation_img.lin" in dit
    ):
        return
    yield from _block_table(
        dit,
        r"(double_blocks|single_blocks)\.(\d+)\.(.+)",
        {
            "double_blocks": "transformer_blocks",
            "single_blocks": "single_transformer_blocks",
        },
        {**FLUX_DOUBLE, **FLUX_SINGLE},
        FLUX_GLOBAL,
    )


MMDIT_BLOCK = {
    "x_block.attn.qkv": _qkv(["attn.to_q", "attn.to_k", "attn.to_v"]),
    "context_block.attn.qkv": _qkv(
        ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"]
    ),
    "x_block.attn2.qkv": _qkv(["attn2.to_q", "attn2.to_k", "attn2.to_v"]),
    "x_block.attn.proj": [("attn.to_out.0", None)],
    "context_block.attn.proj": [("attn.to_add_out", None)],
    "x_block.attn2.proj": [("attn2.to_out.0", None)],
    "x_block.mlp.fc1": [("ff.net.0.proj", None)],
    "x_block.mlp.fc2": [("ff.net.2", None)],
    "context_block.mlp.fc1": [("ff_context.net.0.proj", None)],
    "context_block.mlp.fc2": [("ff_context.net.2", None)],
    "x_block.adaLN_modulation.1": [("norm1.linear", None)],
    "context_block.adaLN_modulation.1": [("norm1_context.linear", None)],
}
MMDIT_GLOBAL = {
    "x_embedder.proj": "pos_embed.proj",
    "context_embedder": "context_embedder",
    "t_embedder.mlp.0": "time_text_embed.timestep_embedder.linear_1",
    "t_embedder.mlp.2": "time_text_embed.timestep_embedder.linear_2",
    "y_embedder.mlp.0": "time_text_embed.text_embedder.linear_1",
    "y_embedder.mlp.2": "time_text_embed.text_embedder.linear_2",
    "final_layer.adaLN_modulation.1": "norm_out.linear",  # scale/shift halves swapped
    "final_layer.linear": "proj_out",
}


def _mmdit_to_diffusers(dit):
    """Diffusers names of SD3 and SD3.5."""
    if "joint_blocks.0.x_block.attn.qkv" not in dit:
        return
    yield from _block_table(
        dit,
        r"(joint_blocks)\.(\d+)\.(.+)",
        {"joint_blocks": "transformer_blocks"},
        MMDIT_BLOCK,
        MMDIT_GLOBAL,
    )
