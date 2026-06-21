# Checkpoint Formats

### The Two Worlds of Checkpoint Formats

At a high level, the Stable Diffusion ecosystem is split into two main "philosophies" for structuring and naming tensors in a model checkpoint. This divergence comes from the two most influential codebases: the original LDM (Latent Diffusion Models) repository and Hugging Face's Diffusers library.

#### 1. The Original LDM / "A1111" Naming Scheme

This is the format used by the original CompVis/Stability AI repositories and adopted by AUTOMATIC1111's Stable Diffusion web UI, which cemented its status as a de-facto standard.

*   **Origin:** Direct reflection of the `nn.Module` hierarchy in the PyTorch code. If you have a `model` object with a `diffusion_model` attribute, which is a `nn.Module` containing a list of blocks, the key will mirror that structure.
*   **Key Naming Convention:** Dot-separated (`.`) strings that trace the path of Python object attributes.
*   **Structure:** Monolithic. The entire model (UNet, Text Encoders, VAE) is typically stored in a single `.ckpt` (pickle) or `.safetensors` file.

**Key Naming Examples (SD 1.5):**

| Component | LDM / A1111 Key Example | Description |
| :--- | :--- | :--- |
| **UNet** | `model.diffusion_model.input_blocks.1.1.proj_in.weight` | A projection layer inside the first down-sampling block. The numbers correspond to indices in `nn.ModuleList`. |
| **UNet** | `model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight`| A key projection for cross-attention in the middle block's transformer. |
| **Text Encoder** | `cond_stage_model.transformer.text_model.encoder.layers.11.self_attn.k_proj.weight`| The key projection for self-attention in the final layer of the CLIP Text Encoder. |

**SDXL Complication:** SDXL uses two text encoders, leading to even longer, more specific prefixes.

*   **CLIP-L (Text Encoder 1):** `conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight`
*   **CLIP-G (Text Encoder 2):** `conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight`

This format is verbose but explicit. It directly tells you where the tensor lives in the model's Python object structure.

#### 2. The Diffusers Naming Scheme

This format was created by Hugging Face for their `diffusers` library, which is designed for modularity and ease of use with the wider Hugging Face ecosystem.

*   **Origin:** Abstracted, descriptive names that are not directly tied to the `nn.Module` attribute names. The goal is to be more readable and stable even if the underlying class structure changes.
*   **Key Naming Convention:** Dot-separated (`.`) strings using descriptive words (`down_blocks`, `attentions`, `proj_in`).
*   **Structure:** Modular. A Diffusers model is typically saved as a directory containing sub-folders for each component (`unet/`, `text_encoder/`, `vae/`), with each sub-folder containing its own `diffusion_pytorch_model.safetensors` file.

**Key Naming Examples (Corresponding to the LDM examples):**

| Component | Diffusers Key Example |
| :--- | :--- |
| **UNet** | `down_blocks.0.attentions.0.proj_in.weight` |
| **UNet** | `mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight` |
| **Text Encoder** | `text_model.encoder.layers.11.self_attn.k_proj.weight` |

Notice that the Diffusers keys are often shorter and more "semantic".

### The Problem Extends to Adapters (LoRA, etc.)

This duality in naming schemes directly impacts adapter models like LoRA. A LoRA trained on a model in one format will have keys derived from that format.

*   **LoRA Naming Convention:** Typically, the base model's `.` separators are replaced with `_`, and a prefix like `lora_unet_` or `lora_te_` is added.

| Base Model Key (LDM) | Corresponding LoRA Key (kohya-ss sd-scripts style) |
| :--- | :--- |
| `model.diffusion_model.input_blocks.1.1.proj_in.weight` | `lora_unet_input_blocks_1_1_proj_in.alpha` / `.lora_down.weight` |
| `cond_stage_model.transformer.text_model.encoder.layers.10.self_attn.v_proj.weight` | `lora_te_text_model_encoder_layers_10_self_attn_v_proj.alpha` / `.lora_down.weight` |

A LoRA trained on a Diffusers model would have keys like `lora_unet_down_blocks_0_attentions_0_proj_in`. A generic tool must be able to recognize *both* of these as targeting the same conceptual layer in the model.

### ComfyUI's Solution: The Unified Internal Representation

ComfyUI achieves its incredible compatibility by **not** forcing users to convert their models. Instead, it uses a powerful, dynamic **load-time mapping system**.

**The "Unified Format" is the Original LDM / A1111 scheme.**

This is the crucial insight. Internally, all of ComfyUI's nodes and operations expect the model's state dictionary to have keys matching the original LDM format. When you load a model, ComfyUI performs these steps:

1.  **Load the File:** It uses `comfy.utils.load_torch_file` to get the raw state dictionary from the `.safetensors` or `.ckpt` file.
2.  **Detect the Format:** The `comfy.sd.load_model_weights` function inspects the keys in the state dictionary. It has heuristics to guess the format. For example, if it sees keys starting with `down_blocks`, it assumes it's a Diffusers UNet.
3.  **Apply a "Rosetta Stone":** Based on the detected format, it applies a specific key-mapping dictionary. It iterates through the keys from the loaded file and translates them to the corresponding LDM-style keys.
4.  **Return the Unified State Dict:** The function returns a *new* state dictionary where all keys now conform to the LDM standard, which the rest of the application can then use seamlessly.

This mapping happens in files like `comfy/sd.py` and `comfy/model_management.py`. There isn't a single "mapper" function but rather a collection of dictionaries and conditional logic that achieve this translation.

### How This Relates to Our Tools and Your Goal

The tools we have been building (like `loralib`, `sdxl_mapper.py`, `resize_lora.py`) are essentially **a microcosm of ComfyUI's own loading strategy.**

*   **`sdxl_mapper.py` (`get_multi_format_lora_keys`)**: This is our "Rosetta Stone". It embodies the same principle as ComfyUI's mappers. It starts with a ground-truth key (LDM style) and generates a list of all known aliases for that key from different training scripts and formats (`lora_te1_...`, `lora_unet_...`, etc.). This is exactly what a generic tool needs.

*   **`loralib/init.py` (`BaseCheckpoint` class)**: When this class initializes, it builds two critical mapping dictionaries:
    *   `base2lora`: Maps an LDM key to its possible LoRA aliases.
    *   `lora2base`: Maps any known LoRA alias *back* to its canonical LDM base key. This is the **most important map for a generic tool**. It allows you to take any LoRA file, read a key like `lora_unet_down_blocks_0_attentions_0_proj_in`, and instantly know it corresponds to the L-D-M-style key `model.diffusion_model.input_blocks.1.1.proj_in.weight`.

*   **`resize_lora.py` and `wd_to_lora.py`**: These scripts *use* this system. `resize_lora.py` uses the `PairedLoraModel` (which relies on `BaseCheckpoint`) to correctly associate a LoRA layer with its corresponding base model weight for scoring. `wd_to_lora.py`, in its simplified form, did a crude, hardcoded mapping (`f"diffusion_model.{lora_key_base}"`), which highlights why a flexible mapping system like `sdxl_mapper.py` is necessary for true generic compatibility.

To write a generic tool that works on any checkpoint or adapter, you must emulate this pattern:

1.  **Choose your internal "ground truth" format.** The LDM/A1111 scheme is the most logical choice as it is ComfyUI's standard.
2.  **Build a mapping layer.** This is a function or class that can translate keys from any known format (Diffusers, various LoRA scripts) *to* your ground truth format. Our `sdxl_mapper.py` is an excellent starting point for this.
3.  **Process all inputs through the mapping layer.** When your tool loads a file, it should create a unified view of that file using the standardized keys, regardless of how they were named on disk.
4.  **Operate on the unified view.** All your core logic should only ever have to deal with one set of key names—your internal standard. This dramatically simplifies everything that follows.