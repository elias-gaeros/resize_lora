# Key Map Design

## Purpose

`loralib/keymap.py` maps each adapter module name to the base-model matrix that the
module modifies. `resize_lora.py` uses it to find the reference matrix for the
`spn_ckpt`, `fro_ckpt` and `subspace` scores, and `check_mapping.py` uses it to report
coverage. The goal is to accept every LoRA that ComfyUI loads.

## Reference

The rules follow ComfyUI's loader: `model_lora_keys_unet()` and
`model_lora_keys_clip()` in `comfy/lora.py`, and the `unet_to_diffusers()`,
`flux_to_diffusers()` and `mmdit_to_diffusers()` tables in `comfy/utils.py`. ComfyUI is
GPL-3.0 and this project is MIT, so the rules are reimplemented, not copied. The SD
UNet conversion follows the diffusers conversion scripts, as tabulated in sd-mecha's
`convert_sd*_to_original.py` (MIT).

## Two Steps

ComfyUI renames a checkpoint's tensors into its own namespace when it loads the
checkpoint, then lists the adapter names each renamed weight accepts. The key map does
both from safetensors headers alone, without loading weights.

### Step 1: checkpoint names to ComfyUI names

`comfy_names()` keeps the matrices (tensors with at least two dimensions) and renames
them by prefix:

| Checkpoint prefix | ComfyUI name |
| --- | --- |
| `model.diffusion_model.`, `diffusion_model.` | `diffusion_model.` |
| `cond_stage_model.transformer.` (SD1) | `clip_l.transformer.` |
| `conditioner.embedders.0.transformer.` (SDXL) | `clip_l.transformer.` |
| `cond_stage_model.model.` (SD2), `conditioner.embedders.1.model.` (SDXL) | `clip_h` / `clip_g`, converted from open_clip |
| `text_encoders.` (SD3, Flux all-in-one files) | prefix removed |

The open_clip conversion splits `attn.in_proj_weight` into the rows of `q_proj`,
`k_proj` and `v_proj` (in that order), renames `mlp.c_fc`/`mlp.c_proj` to
`mlp.fc1`/`mlp.fc2`, and maps `text_projection` transposed, because open_clip applies
it as `x @ text_projection`.

A file with none of these prefixes holds one bare model, which `_bare_prefix()`
identifies:

| Bare file contents | ComfyUI name |
| --- | --- |
| `text_model.encoder.…` | `clip_l`, `clip_g` or `clip_h`, by width 768, 1280 or 1024 |
| `encoder.block.…` | `t5xxl`, or `umt5xxl` when every block has a relative attention bias |
| `model.layers.…` and `model.embed_tokens.weight` | `qwen3_4b` or `qwen3_8b` by width, otherwise `llm` |
| anything else | `diffusion_model.` |

A base model can span several files, such as a Flux diffusion model and its
`clip_l`/`t5xxl` text encoders. `build_keymap()` takes the shapes of each file and
rejects two files that produce the same ComfyUI name.

### Step 2: ComfyUI names to adapter names

`adapter_names()` lists the names ComfyUI accepts for each ComfyUI name `diffusion_model.X`:

- `diffusion_model.X` (ComfyUI's own format, written by its LoRA save node), `X`,
  `transformer.X`, `base_model.model.X`, `lora_unet_X` and `lycoris_X` with dots
  replaced by underscores. These apply to every diffusion model, including Wan,
  Chroma and Flux.2.
- `diffusion_model.single_blocks.N.linear1_qkv`, the q/k/v rows of Flux `linear1`.
- Diffusers names of SD1/SD2/SDXL UNets (`unet_to_diffusers()`), with the `unet.`,
  `lora_unet_`, `lycoris_` and attention-processor (`attn1.processor.to_q`) variants.
- Diffusers names of Flux.1 and Chroma (`FLUX_DOUBLE`, `FLUX_SINGLE`, `FLUX_GLOBAL`)
  and of SD3/SD3.5 (`MMDIT_BLOCK`, `MMDIT_GLOBAL`), with the `transformer.`,
  `base_model.model.`, `lora_transformer_` and `lycoris_` variants. Separate diffusers
  q/k/v projections map to row slices of the fused `qkv` and `linear1` matrices. Flux.2
  is excluded because its fused layers differ.

For text encoders, `_text_encoder_names()` adds `text_encoders.<ComfyUI name>` and, for
non-CLIP encoders, `text_encoders.transformer.…`; the kohya prefixes `lora_te_`,
`lora_te1_` and `lora_te2_` assigned as ComfyUI assigns them (SDXL: CLIP-L is te1, CLIP-G
is te2); `text_encoder.`/`text_encoder_2.` diffusers names; the T5 index (`lora_te3_`
after two CLIP encoders, `lora_te2_` and `lora_te3_` after CLIP-L alone); and kohya
`lora_te_` names for a lone LLM text encoder.

The result is the union of ComfyUI's per-architecture rules, so a name is accepted
whenever any architecture accepts it. An alias that two different matrices claim is
dropped rather than resolved by insertion order.

## Targets

A `Target` names rows `start:stop` of tensor `key` in base file number `file`,
optionally transposed, and records the stored shape. `str(Target)` identifies it in
logs and in `norms_cache.json`. `BaseCheckpoint.get_weights()` reads only those rows
and multiplies ComfyUI fp8 weights by their per-tensor `scale_weight` or
`weight_scale`. It rejects block-wise scales (4-bit formats).

## Pairing

`find_lora_layers()` in `loralib/__init__.py` finds the plain low-rank layers by the
up/down suffix pairs ComfyUI accepts (`lora_up`/`lora_down`, `lora_B`/`lora_A`,
`lora.up`/`lora.down`, `_lora.up`/`_lora.down`, `lora_linear_layer.up`/`.down`,
`lora_B.default`/`lora_A.default`). A layer with `lora_mid` (Tucker LoCon) or
`reshape_weight` is not resizable. `PairedLoraModel` looks each layer up in the key
map, and accepts the pairing only when the layer's output size equals the target's
rows and its input size equals the target's input size. A lookup of `lora_unet__X`
(Wan Fun LoRAs) falls back to `lora_unet_X`, as ComfyUI's converter does.

Unpaired layers, shape mismatches and all other tensors (`.diff`, LoHa, LoKr, OFT) are
copied to the output unchanged. The output keeps the input's module names and
suffixes, and writes `.alpha` only when the input has it; a missing alpha means a scale
of 1, as in ComfyUI.

## Limitations

- Diffusers names are mapped for SD/SDXL UNets, SD3/SD3.5 and Flux.1/Chroma only.
  ComfyUI also converts diffusers names of AuraFlow, PixArt, HunyuanVideo, Qwen-Image,
  Z-Image and Krea2; those LoRAs are reported as unmapped.
- ComfyUI's LoRA pre-conversions for BFL control LoRAs and USO LoRAs are not applied.
- Bare LLM text encoders other than Qwen3 4B/8B get only the names that do not depend
  on ComfyUI's model name.
- Base files in diffusers layout (`diffusion_pytorch_model.safetensors`), GGUF and
  4-bit quantized files are not read.
- The shape check cannot tell apart square matrices of the same size, such as the q
  and k thirds of a fused projection. The order of fused rows follows the reference
  implementations.

## Extension Checklist

1. Add the rule to the narrowest table or function above, emitting only targets that
   exist in the base.
2. Add a synthetic test in `tests/test_keymap.py` for each new name form.
3. Run `check_mapping.py` on a corpus with a matching base, as described in
   `data_driven_dev.md`, and confirm that no shape mismatch appears.
