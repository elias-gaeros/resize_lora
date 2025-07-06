# Adapter Formats

### Overview

The logic for handling different adapter variants is primarily located within two functions in `comfy.lora.py`:

1.  `lora_state_dict_mapping`: This function parses the keys from the LoRA file.
2.  `calculate_weight`: This function applies the actual mathematical operations based on the tensors found for a given module.

By examining these, we can create a definitive list.

### Code Analysis

**`nodes.py` (LoraLoader) -> `comfy.sd.py` (load_lora_for_models) -> `comfy.lora.py` (load_lora, lora_state_dict_mapping, calculate_weight) -> `comfy.model_patcher.py` (add_patches)**

The key translation logic in `lora_state_dict_mapping` and the calculation logic in `calculate_weight` confirm support for a wide range of adapter types beyond standard LoRA.

Here's what the code tells us:

- **Standard LoRA/LoCon**: The code explicitly looks for `lora_down.weight`, `lora_up.weight`, and `lora_mid.weight`. It also handles the `.alpha` value. This confirms support for standard LoRA on Linear layers and LoCon (LoRA for Convolutional layers).
- **DoRA**: The presence of a `dora_scale` tensor for a module triggers the DoRA calculation path. This is explicitly checked.
- **LyCORIS (LoHa, LoKr, etc.)**: `calculate_weight` contains logic to handle multiple formats from the LyCORIS project.
  - **LoHa**: It checks for the presence of `hada_w1_a`, `hada_w1_b`, `hada_w2_a`, `hada_w2_b` tensors, which are the components of a LoHa block.
  - **LoKr**: It checks for `lokr_w1` and `lokr_w2`, which are the factors for a LoKr (Kronecker Product) block.
  - **IA³**: It checks for `ia3_input_mask`, which indicates the IA³ (Infused Adapter by Inhibiting and Amplifying) method. This is a very simple adapter that only scales activations.
  - **Full Rank Adapters**: The code can also handle what are effectively full-rank adapters by looking for a single `diff.weight` tensor, a common format for full fine-tuning or Dreambooth extraction.

### List of Supported LoRA Formats and Naming Conventions

Here is a breakdown of the formats and naming schemes understood by ComfyUI.

#### Part 1: Model Targeting (Key Prefixes)

The beginning of a key in the LoRA file's state dictionary determines which part of the overall model architecture it will patch.

| Prefix       | Target Model            | Description                                                                                 |
| :----------- | :---------------------- | :------------------------------------------------------------------------------------------ |
| `lora_unet_` | UNet (Diffusion Model)  | The most common prefix. Applies patches to the main denoising U-Net.                        |
| `lora_te_`   | Text Encoder            | Used for SD 1.x models. In SDXL, this is ambiguous and the numbered versions are preferred. |
| `lora_te1_`  | Text Encoder 1 (CLIP-L) | Specifically targets the first text encoder in SDXL models (usually OpenCLIP-L/14).         |
| `lora_te2_`  | Text Encoder 2 (CLIP-G) | Specifically targets the second text encoder in SDXL models (usually OpenCLIP-G/14).        |

#### Part 2: Module Path Translation

After the prefix is removed, ComfyUI translates the remainder of the key into a module path that PyTorch can understand.

**Separator Conversion**: All underscores (`_`) are replaced with dots (`.`). This is the primary mechanism for converting the flat string into a hierarchical module path.

**Example**:

- **LoRA Key**: `lora_unet_input_blocks_1_1_transformer_blocks_0_attn2_to_k`
- **After Prefix Removal**: `input_blocks_1_1_transformer_blocks_0_attn2_to_k`
- **After Separator Conversion**: `input.blocks.1.1.transformer_blocks.0.attn2.to.k`
- This final string is used by `ModelPatcher` to locate the target `torch.nn.Module` within the UNet.

#### Part 3: Adapter Type and Tensor Suffixes

For each module being patched, the type of adapter is determined by the suffixes of the tensor names.

| Adapter Type      | Required Tensor Suffixes                                                                           | Optional Suffixes                               | Description                                                                                                 |
| :---------------- | :------------------------------------------------------------------------------------------------- | :---------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Standard LoRA** | `*.lora_down.weight` <br> `*.lora_up.weight`                                                       | `*.alpha`                                       | Standard Low-Rank Adaptation for linear layers.                                                             |
| **LoCon**         | `*.lora_down.weight` <br> `*.lora_up.weight` <br> _and/or_ <br> `*.lora_mid.weight`                | `*.alpha`                                       | LoRA for Convolutional layers. Uses `lora_mid` for the 3x3 kernel part.                                     |
| **DoRA**          | `*.lora_down.weight` <br> `*.lora_up.weight`                                                       | `*.dora_scale` <br> `*.alpha`                   | Weight-Decomposed Rank-Adaptation. If `dora_scale` is present, DoRA math is used.                           |
| **LoHa**          | `*.hada_w1_a.weight` <br> `*.hada_w1_b.weight` <br> `*.hada_w2_a.weight` <br> `*.hada_w2_b.weight` | `*.alpha`                                       | LyCORIS LoRA with Hadamard Product. Decomposes matrices differently from standard LoRA.                     |
| **LoKr**          | `*.lokr_w1.weight` <br> _and/or_ <br> `*.lokr_w2.weight`                                           | `*.alpha` <br> `*.t1.weight` <br> `*.t2.weight` | LyCORIS LoRA with Kronecker Product. Uses smaller matrices combined with a Kronecker product.               |
| **IA³**           | `*.ia3_input_mask.weight`                                                                          | (none)                                          | Infused Adapter by Inhibiting and Amplifying. A very lightweight adapter that only learns a scaling vector. |
| **Full / Diff**   | **(See table below)**                                                                              | `*.alpha`                                       | Represents a full-weight difference, typically from Dreambooth or full fine-tuning.                         |

#### Part 4: The Full/Diff Naming Conventions (Explicit vs. Shorthand)

This category covers direct modification of parameters, like weights and biases. ComfyUI supports both explicit and shorthand formats for maximum compatibility with community training scripts.

| Target Parameter | Explicit Format (General)      | Shorthand Format (Community Standard) | Description                                                 |
| :--------------- | :----------------------------- | :------------------------------------ | :---------------------------------------------------------- |
| **Weight**       | `...<module_path>.weight.diff` | `...<module_path>.diff_w`             | Patches the **weight** matrix. Both formats are recognized. |
| **Bias**         | `...<module_path>.bias.diff`   | `...<module_path>.diff_b`             | Patches the **bias** vector. Both formats are recognized.   |

The shorthand formats (`diff_w`, `diff_b`) are widely used by popular training scripts like `kohya_ss`'s sd-scripts and are essential for compatibility.

#### Summary Table with Examples

| Adapter Type           | Target Parameter | Key Example in LoRA File                                                  | Target Module in Model                          |
| :--------------------- | :--------------- | :------------------------------------------------------------------------ | :---------------------------------------------- |
| **Standard LoRA**      | `weight`         | `lora_unet_down_blocks_0_attention_0_to_q.lora_down.weight`               | `down_blocks.0.attention.0.to_q`                |
| **LoCon**              | `weight`         | `lora_unet_up_blocks_0_resnets_1_conv1.lora_mid.weight`                   | `up_blocks.0.resnets.1.conv1`                   |
| **DoRA**               | `weight`         | `lora_unet_down_blocks_0_attention_0_to_q.dora_scale`                     | `down_blocks.0.attention.0.to_q`                |
| **SDXL CLIP-L LoRA**   | `weight`         | `lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha`                      | `text_model.encoder.layers.0.mlp.fc1`           |
| **SDXL CLIP-G LoHa**   | `weight`         | `lora_te2_text_model_encoder_layers_20_self_attn_q_proj.hada_w1_a.weight` | `text_model.encoder.layers.20.self_attn.q_proj` |
| **IA³**                | `weight`         | `lora_unet_down_blocks_1_resnets_0_conv1.ia3_input_mask.weight`           | `down_blocks.1.resnets.0.conv1`                 |
| **Full/Diff (Weight)** | `weight`         | `lora_unet_down_blocks_0_attention_0_to_q.weight.diff` (Explicit)         | `down_blocks.0.attention.0.to_q`                |
| **Full/Diff (Weight)** | `weight`         | `lora_unet_down_blocks_0_attention_0_to_q.diff_w` (Shorthand)             | `down_blocks.0.attention.0.to_q`                |
| **Full/Diff (Bias)**   | `bias`           | `lora_unet_down_blocks_0_attention_0_to_q.bias.diff` (Explicit)           | `down_blocks.0.attention.0.to_q`                |
| **Full/Diff (Bias)**   | `bias`           | `lora_unet_down_blocks_0_attention_0_to_q.diff_b` (Shorthand)             | `down_blocks.0.attention.0.to_q`                |

This list covers the known adapter formats supported by ComfyUI's internal LoRA loader. The system is designed flexibly, so if a new adapter type emerges, it would primarily require updates to the `calculate_weight` function in `comfy.lora.py` to be supported.
