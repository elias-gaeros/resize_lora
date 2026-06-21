# ComfyUI's Adapter Loading System: A Definitive Guide

ComfyUI possesses a highly flexible and robust system for applying adapters like LoRA, LoCon, and LyCORIS variants. This system is designed for maximum compatibility, automatically detecting and applying patches from LoRA files trained with various conventions, often without any need for conversion. This document provides an exhaustive breakdown of this loading logic, with direct references to the source code.

### The Core Logic Flow

The process of loading an adapter and applying it to a model follows a clear path through the ComfyUI codebase, beginning at the user-facing node and drilling down into specialized functions.

1.  **Entry Point (`nodes.py`)**: The process starts with the `LoraLoader` node. This node takes the `model` and `clip` patchers, the `lora_name`, and strength values. Its primary action is to call `comfy.sd.load_lora_for_models`.

2.  **Key Mapping (`comfy.sd.py`)**: The `load_lora_for_models` function is the main dispatcher.
    - It first builds a `key_map` by calling `comfy.lora.model_lora_keys_unet` and `comfy.lora.model_lora_keys_clip`. This map contains all potential LoRA key names (e.g., `lora_unet_...`, `lora_te1_...`, and various community formats) and links them to their corresponding weight names in the ComfyUI model state dictionaries.
    - ```python
      # in comfy/sd.py
      def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
          # 1. Create a map of all possible LoRA key names to the model's weight names.
          key_map = {}
          if model is not None:
              # Populates the map with UNet keys
              key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
          if clip is not None:
              # Populates the map with Text Encoder keys
              key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
      
          # ...
      ```

3.  **Patch Dictionary Creation (`comfy.lora.py`)**:
    - The `key_map` and the LoRA file's `state_dict` are passed to `comfy.lora.load_lora`.
    - This function iterates through the generated `key_map`. For each potential LoRA key, it checks if the corresponding tensors (e.g., `lora_down`, `lora_up`, `alpha`) exist in the LoRA file.
    - ```python
      # in comfy/sd.py
      def load_lora_for_models(...):
          # ... (key_map creation) ...
          
          # 2. Convert the LoRA to a compatible format if needed (e.g., for `safetensors` metadata).
          lora = comfy.lora_convert.convert_lora(lora)
          # 3. Load the LoRA weights into a patch dictionary.
          loaded = comfy.lora.load_lora(lora, key_map)
          # ...
      ```
    - If tensors are found, `load_lora` creates a "patch" object (e.g., a `LoRAAdapter` instance) and adds it to a `patch_dict`. The dictionary is keyed by the target model weight name (e.g., `diffusion_model.input_blocks.0.0.weight`).

4.  **Patching the Model (`comfy.model_patcher.py`)**:
    - The `patch_dict` (named `loaded` in the code) is returned to `load_lora_for_models`.
    - The `model` and `clip` `ModelPatcher` objects are cloned, and the `add_patches` method is called on each clone, passing the `patch_dict`. This method stores the patches internally within the `ModelPatcher` instance.
    - ```python
      # in comfy/sd.py
      def load_lora_for_models(...):
          # ... (key_map, loading) ...
          
          # 4. Clone the model and clip patchers and apply the patches.
          if model is not None:
              new_modelpatcher = model.clone()
              k = new_modelpatcher.add_patches(loaded, strength_model)
          # ... (clip is handled similarly) ...
      ```
    - During the sampling/inference process, methods like `patch_weight_to_device` are called. This function retrieves the stored patches and uses `comfy.lora.calculate_weight` to compute the final modified weight by applying the LoRA adjustments.

### Key Parsing: A Model-First Mapping Strategy

ComfyUI's compatibility stems from its "model-first" approach to identifying LoRA keys, primarily within `comfy.lora.model_lora_keys_unet` and `comfy.lora.model_lora_keys_clip`. Instead of parsing keys from the LoRA file directly and trying to guess their target, it does the reverse: it inspects the loaded UNet and CLIP models and generates a map of all *possible* LoRA key names that could correspond to each weight.

This map supports multiple naming conventions simultaneously.

#### 1. ComfyUI Native Prefix Convention

The system generates keys with a special prefix indicating the target model component. This is the most explicit convention.

**Structure**: `<prefix><module_path_with_underscores>`

**Generation Logic (from `model_lora_keys_unet`)**:
The function iterates through the UNet's state dictionary. For each weight, it creates a corresponding LoRA key by replacing dots with underscores and prepending a prefix.

```python
# in comfy/lora.py
def model_lora_keys_unet(model, key_map={}):
    sd = model.state_dict()
    sdk = sd.keys()

    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            # Convert model key 'diffusion_model.input_blocks.1.1.norm.weight'
            # to lora key 'lora_unet_input_blocks_1_1_norm'
            key_lora = k[len("diffusion_model."):-len(".weight")].replace(".", "_")
            key_map["lora_unet_{}".format(key_lora)] = k
    # ...
```
- **Example**: The model weight `diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight` is mapped from the LoRA key `lora_unet_input_blocks_1_1_transformer_blocks_0_attn2_to_k`.

**Prefixes:**

| Prefix       | Target Model             | Description                                                              |
| :----------- | :----------------------- | :----------------------------------------------------------------------- |
| `lora_unet_` | UNet (Diffusion Model)   | The most common prefix, for patching the main denoising model.           |
| `lora_te_`   | Text Encoder (Ambiguous) | For SD 1.x. In SDXL, `te1`/`te2` are preferred.                          |
| `lora_te1_`  | Text Encoder 1 (CLIP-L)  | Specifically targets the first SDXL text encoder (e.g., OpenCLIP-L/14).  |
| `lora_te2_`  | Text Encoder 2 (CLIP-G)  | Specifically targets the second SDXL text encoder (e.g., OpenCLIP-G/14). |


#### 2. Community and Diffusers Conventions

To support LoRAs from other libraries (like A1111's WebUI or HuggingFace Diffusers), the key-mapping functions *also* add entries to the `key_map` that match those conventions directly. This is not a fallback, but a parallel strategy.

**Generation Logic (from `model_lora_keys_unet`)**:
For each weight, a key is added to the map that is simply the module path itself, without any prefix. This allows `load_lora` to find LoRA tensors like `...proj_in.lora_down.weight`.

```python
# in comfy/lora.py
def model_lora_keys_unet(model, key_map={}):
    # ...
    for k in sdk:
        if k.startswith("diffusion_model.") and k.endswith(".weight"):
            # ... (handles comfy prefix)
            # For a model key like 'diffusion_model.foo.bar.weight',
            # add a direct mapping for 'diffusion_model.foo.bar'
            key_map["{}".format(k[:-len(".weight")])] = k #generic lora format
    # ...
    # Additionally, many explicit mappings for Diffusers formats are added
    diffusers_keys = comfy.utils.unet_to_diffusers(model.model_config.unet_config)
    for k in diffusers_keys:
        if k.endswith(".weight"):
            #...
            unet_key = "diffusion_model.{}".format(diffusers_keys[k])
            # Maps a diffusers-style key like 'down_blocks.0.attentions.0.proj_in'
            # to the corresponding ComfyUI unet_key.
            key_map[diffusers_lora_key] = unet_key
```

This mapping is why most community LoRAs work without modification. The system doesn't guess; it knows all the possible valid names for a given model weight and checks for all of them.

### Supported Adapter Formats & Tensor Suffixes

> **Scope:** This section describes ComfyUI's loader. The experimental
> `loralib.key_mapper.KeyMapper` recognizes only the suffixes listed in its `SUFFIXES`
> set, and `resize_lora.py` remains limited to SDXL LoRA/LoCon. Loader support in
> ComfyUI does not imply mapping or resizing support in this repository.

The specific adapter format is determined by the suffixes of the tensor keys, which are checked within `comfy.lora.load_lora` by iterating through the `weight_adapter` classes.

| Adapter Type      | Required Tensor Suffixes                                                                           | Optional Suffixes                               | Description                                                                       |
| :---------------- | :------------------------------------------------------------------------------------------------- | :---------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Standard LoRA** | `*.lora_down.weight` <br> `*.lora_up.weight`                                                       | `*.alpha`                                       | Standard Low-Rank Adaptation.                                                     |
| **LoCon**         | `*.lora_down.weight` <br> `*.lora_up.weight` <br> and/or `*.lora_mid.weight`                       | `*.alpha`                                       | LoRA for Convolutional layers, using `lora_mid` for the 3x3 kernel part.          |
| **DoRA**          | `*.lora_down.weight` <br> `*.lora_up.weight`                                                       | `*.dora_scale` <br> `*.alpha`                   | Weight-Decomposed Rank-Adaptation. Presence of `dora_scale` triggers DoRA math.   |
| **LoHa**          | `*.hada_w1_a.weight` <br> `*.hada_w1_b.weight` <br> `*.hada_w2_a.weight` <br> `*.hada_w2_b.weight` | `*.alpha`                                       | LyCORIS LoRA with Hadamard Product.                                               |
| **LoKr**          | `*.lokr_w1.weight` <br> and/or `*.lokr_w2.weight`                                                  | `*.alpha` <br> `*.t1.weight` <br> `*.t2.weight` | LyCORIS LoRA with Kronecker Product. Also supports factored `w1_a`, `w1_b`, etc.  |
| **GLoRA**         | `*.a1.weight` <br> `*.a2.weight` <br> `*.b1.weight` <br> `*.b2.weight`                             | `*.alpha`                                       | Generalized LoRA.                                                                 |
| **OFT**           | `*.oft_blocks` (3D Tensor)                                                                         | `*.rescale` <br> `*.alpha`                      | Orthogonal Finetuning.                                                            |
| **BOFT**          | `*.oft_blocks` (4D Tensor)                                                                         | `*.rescale` <br> `*.alpha`                      | Butterfly Orthogonal Finetuning.                                                  |
| **Full / Diff**   | **See Table Below**                                                                                | (N/A)                                           | A full-weight difference, typically from Dreambooth or full fine-tuning.          |

#### Full/Diff Naming Conventions

For direct parameter modification, ComfyUI supports several formats to cover community training scripts. These are not adapters in the low-rank sense but are handled by the same loading mechanism. The key check happens inside `comfy.lora.load_lora`.

| Target Parameter | Key Suffix 1         | Key Suffix 2          | Code Check in `load_lora`                        |
| :--------------- | :------------------- | :-------------------- | :----------------------------------------------- |
| **Weight**       | `.diff`              | `.w_norm`             | `diff_weight = lora.get(...)`, `w_norm = lora.get(...)` |
| **Bias**         | `.diff_b`            | `.b_norm`             | `diff_bias = lora.get(...)`, `b_norm = lora.get(...)`   |

### Master Summary Table

The following table provides concrete examples demonstrating how the model-first mapping works in practice.

**ComfyUI Native Prefix Format**
| Adapter Type | Target Parameter | Key Example in LoRA File | Target Module in Model |
| :--- | :--- | :--- | :--- |
| LoRA | `weight` | `lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight` | `diffusion_model.down_blocks_0_attentions_0_proj_in.weight` |
| Full/Diff | `bias` | `lora_unet_down_blocks_0_attentions_0_proj_in.diff_b` | `diffusion_model.down_blocks_0_attentions_0_proj_in.bias` |
| LoHa | `weight` | `lora_te2_text_model_encoder_layers_20_self_attn_q_proj.hada_w1_a.weight` | `clip_g.transformer.text_model.encoder.layers.20.self_attn.q_proj.weight` |

**Community / Direct Mapping Format**
| Adapter Type | Target Parameter | Key Example in LoRA File | Target Module in Model |
| :--- | :--- | :--- | :--- |
| LoRA | `weight` | `text_encoder_2.text_model.encoder.layers.10.mlp.fc1.lora_up.weight` | `clip_g.transformer.text_model.encoder.layers.10.mlp.fc1.weight` |
| LoRA | `weight` | `mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_down.weight` | `diffusion_model.mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight` |
| Full/Diff | `bias` | `up_blocks.3.resnets.2.conv2.diff_b` | `diffusion_model.up_blocks.3.resnets.2.conv2.bias` |

This system ensures that nearly any LoRA-like adapter can be loaded and applied correctly, making ComfyUI a highly interoperable and powerful tool for model customization.
