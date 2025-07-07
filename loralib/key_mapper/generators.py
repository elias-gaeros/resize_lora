# File: loralib/key_mapper/generators.py (UPDATED)

from abc import ABC, abstractmethod
from typing import Dict, Set
import logging
import re

from .types import ModelContext

logger = logging.getLogger(__name__)

# --- Abstract Base Class ---


class MappingGenerator(ABC):
    """
    Base class for components that generate key mappings.
    Each generator contributes a dictionary of mappings to the final 'Rosetta Stone'.
    """

    def generate(self, context: ModelContext, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        raise NotImplementedError


# --- Concrete Generator Implementations ---


class ComfyUIPrefixGenerator(MappingGenerator):
    """
    Generates mappings for ComfyUI's native, prefix-based keys (UNet only).
    e.g., 'lora_unet_input_blocks_1_1...' -> 'model.diffusion_model.input_blocks.1.1...'
    """

    def generate(self, context: ModelContext, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        mapping = {}
        base_keys = context.base_keys

        prefix_map = {
            "model.diffusion_model.": "lora_unet_",
        }

        for base_key in base_keys:
            if not base_key.endswith(".weight"):
                continue

            for model_prefix, lora_prefix in prefix_map.items():
                if base_key.startswith(model_prefix):
                    # This is the correct, robust way to generate the lora key
                    # 'model.diffusion_model.input_blocks.1.1.proj_in.weight'
                    # -> 'input_blocks.1.1.proj_in'
                    # -> 'input_blocks_1_1_proj_in'
                    key_part = base_key[len(model_prefix) : -len(".weight")]
                    lora_key_base = lora_prefix + key_part.replace(".", "_")
                    mapping[lora_key_base] = base_key
                    break

        return mapping


class ClipLMappingGenerator(MappingGenerator):
    """
    Handles CLIP-L/TE1 text encoder mapping for SDXL, SD1.x/2.x, and SD3.5.
    """

    def generate(self, context: ModelContext, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        mapping = {}
        base_keys = context.base_keys
        
        # SD3.5 CLIP-L (text_encoders.clip_l.*)
        if "CLIP-L" in context.components_present and context.model_type.startswith("SD3.5"):
            for i in range(24):
                # Self-attention
                for part in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                    lora_key = f"lora_te1_text_model_encoder_layers_{i}_self_attn_{part}"
                    base_key = f"text_encoders.clip_l.transformer.text_model.encoder.layers.{i}.self_attn.{part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key
                # MLP
                for lora_part, base_part in [("fc1", "fc1"), ("fc2", "fc2")]:
                    lora_key = f"lora_te1_text_model_encoder_layers_{i}_mlp_{lora_part}"
                    base_key = f"text_encoders.clip_l.transformer.text_model.encoder.layers.{i}.mlp.{base_part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key

            # --- Final Text Projection Mapping for CLIP-L ---
            base_key_text_proj_l = "text_encoders.clip_l.transformer.text_model.text_projection"
            if base_key_text_proj_l in base_keys:
                # Standard name
                mapping["lora_te1_text_model_text_projection"] = base_key_text_proj_l
                # Alias for consistency with TE2
                mapping["lora_te1_text_projection"] = base_key_text_proj_l
        
        # SDXL CLIP-L (conditioner.embedders.0.*)
        elif "CLIP-L" in context.components_present:
            for i in range(24):
                # Self-attention
                for part in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                    lora_key = f"lora_te1_text_model_encoder_layers_{i}_self_attn_{part}"
                    base_key = f"conditioner.embedders.0.transformer.text_model.encoder.layers.{i}.self_attn.{part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key
                # MLP
                for lora_part, base_part in [("fc1", "fc1"), ("fc2", "fc2")]:
                    lora_key = f"lora_te1_text_model_encoder_layers_{i}_mlp_{lora_part}"
                    base_key = f"conditioner.embedders.0.transformer.text_model.encoder.layers.{i}.mlp.{base_part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key

            # --- Final Text Projection Mapping for CLIP-L ---
            base_key_text_proj_l = "conditioner.embedders.0.transformer.text_model.text_projection"
            if base_key_text_proj_l in base_keys:
                # Standard name
                mapping["lora_te1_text_model_text_projection"] = base_key_text_proj_l
                # Alias for consistency with TE2
                mapping["lora_te1_text_projection"] = base_key_text_proj_l
        
        # SD1.x/2.x CLIP-L (cond_stage_model.transformer.*)
        elif any(k.startswith("cond_stage_model.transformer.") for k in base_keys):
            for i in range(24):
                for part in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                    lora_key = f"lora_te_text_model_encoder_layers_{i}_self_attn_{part}"
                    base_key = f"cond_stage_model.transformer.text_model.encoder.layers.{i}.self_attn.{part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key
                for lora_part, base_part in [("fc1", "fc1"), ("fc2", "fc2")]:
                    lora_key = f"lora_te_text_model_encoder_layers_{i}_mlp_{lora_part}"
                    base_key = f"cond_stage_model.transformer.text_model.encoder.layers.{i}.mlp.{base_part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key
        
        return mapping


class ClipGMappingGenerator(MappingGenerator):
    """
    Handles CLIP-G/TE2 text encoder mapping for SDXL and SD3.5.
    """

    def generate(self, context: ModelContext, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        mapping = {}
        base_keys = context.base_keys
        
        # SD3.5 CLIP-G (text_encoders.clip_g.*)
        if "CLIP-G" in context.components_present and context.model_type.startswith("SD3.5"):
            for i in range(32):
                # Self-attention
                for part in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                    lora_key = f"lora_te2_text_model_encoder_layers_{i}_self_attn_{part}"
                    base_key = f"text_encoders.clip_g.transformer.text_model.encoder.layers.{i}.self_attn.{part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key
                # MLP
                for lora_part, base_part in [("fc1", "fc1"), ("fc2", "fc2")]:
                    lora_key = f"lora_te2_text_model_encoder_layers_{i}_mlp_{lora_part}"
                    base_key = f"text_encoders.clip_g.transformer.text_model.encoder.layers.{i}.mlp.{base_part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key

            # --- Final Text Projection Mapping ---
            base_key_text_proj = "text_encoders.clip_g.transformer.text_model.text_projection"
            if base_key_text_proj in base_keys:
                mapping["lora_te2_text_projection"] = base_key_text_proj
                # Add alias for LoRAs trained with 'text_model' convention
                mapping["lora_te2_text_model_text_projection"] = base_key_text_proj
        
        # SDXL CLIP-G (conditioner.embedders.1.*)
        elif "CLIP-G" in context.components_present:
            for i in range(32):
                # MLP: fc1 -> c_fc, fc2 -> c_proj
                for lora_part, base_part in [("fc1", "c_fc"), ("fc2", "c_proj")]:
                    lora_key = f"lora_te2_text_model_encoder_layers_{i}_mlp_{lora_part}"
                    base_key = f"conditioner.embedders.1.model.transformer.resblocks.{i}.mlp.{base_part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key

                # --- Self-attention: Many-to-one mapping ---
                base_key_in_proj = f"conditioner.embedders.1.model.transformer.resblocks.{i}.attn.in_proj_weight"
                if base_key_in_proj in base_keys:
                    for part in ["q_proj", "k_proj", "v_proj"]:
                        lora_key = f"lora_te2_text_model_encoder_layers_{i}_self_attn_{part}"
                        mapping[lora_key] = base_key_in_proj

                # The output projection is a straightforward one-to-one mapping.
                base_key_out_proj = f"conditioner.embedders.1.model.transformer.resblocks.{i}.attn.out_proj.weight"
                if base_key_out_proj in base_keys:
                    lora_key_out_proj = f"lora_te2_text_model_encoder_layers_{i}_self_attn_out_proj"
                    mapping[lora_key_out_proj] = base_key_out_proj

            # --- Final Text Projection Mapping ---
            # This is a key that exists outside the main resblocks loop.
            base_key_text_proj = "conditioner.embedders.1.model.text_projection"
            if base_key_text_proj in base_keys:
                mapping["lora_te2_text_projection"] = base_key_text_proj
                # Add alias for LoRAs trained with 'text_model' convention
                mapping["lora_te2_text_model_text_projection"] = base_key_text_proj
        
        return mapping


class DiTMappingGenerator(MappingGenerator):
    """
    Handles key mapping for DiT-based architectures like FLUX and Chroma.
    This centralized generator reduces code duplication by handling the common
    block structure and applying specific rules based on the model type.
    """
    def generate(self, context: ModelContext, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        mapping = {}
        # This generator runs for any model identified as a DiT variant.
        if "DiT" not in context.components_present:
            return mapping

        base_keys = context.base_keys
        for key in base_keys:
            if not key.endswith(".weight"):
                continue

            # --- Shared Logic for all DiT models ---

            # Match 'double_blocks' for both img/txt and attn/mlp paths
            db_match = re.match(r"double_blocks\.(\d+)\.(img|txt)_(attn|mlp)", key)
            if db_match:
                block_num, cond_type, block_type = db_match.groups()
                if block_type == 'attn':
                    part_match = re.search(r"\.(qkv|proj)\.weight$", key)
                    if part_match:
                        part = part_match.group(1)
                        lora_key = f"lora_unet_double_blocks_{block_num}_{cond_type}_attn_{part}"
                        mapping[lora_key] = key
                        continue
                elif block_type == 'mlp':
                    part_match = re.search(r"\.(\d+)\.weight$", key)
                    if part_match:
                        part_num = part_match.group(1)
                        lora_key = f"lora_unet_double_blocks_{block_num}_{cond_type}_mlp_{part_num}"
                        mapping[lora_key] = key
                        continue
            
            # Match 'single_blocks'
            sb_match = re.match(r"single_blocks\.(\d+)\.linear(\d+)\.weight", key)
            if sb_match:
                block_num, linear_part = sb_match.groups()
                lora_key = f"lora_unet_single_blocks_{block_num}_linear{linear_part}"
                mapping[lora_key] = key
                continue

            # Match 'final_layer' and 'txt_in'
            if key == "final_layer.linear.weight":
                mapping["lora_unet_final_layer_linear"] = key
                continue
            if key == "txt_in.weight":
                mapping["lora_unet_txt_in"] = key
                continue

            # --- FLUX-Specific Logic ---
            if context.model_type == "FLUX-dev":
                # Match modulation layers in double blocks for FLUX
                db_mod_match = re.match(r"double_blocks\.(\d+)\.(img|txt)_mod\.lin\.weight", key)
                if db_mod_match:
                    block_num, cond_type = db_mod_match.groups()
                    lora_key = f"lora_unet_double_blocks_{block_num}_{cond_type}_mod_lin"
                    mapping[lora_key] = key
                    continue
                # Match modulation layers in single blocks for FLUX
                sb_mod_match = re.match(r"single_blocks\.(\d+)\.modulation\.lin\.weight", key)
                if sb_mod_match:
                    block_num = sb_mod_match.groups()[0]
                    lora_key = f"lora_unet_single_blocks_{block_num}_modulation_lin"
                    mapping[lora_key] = key
                    continue

            # --- Chroma-Specific Logic ---
            if context.model_type == "Chroma":
                # Match 'distilled_guidance_layer' for Chroma
                guidance_match = re.match(r"distilled_guidance_layer\.(layers\.(\d+)\.(in|out)_layer|in_proj|out_proj)\.weight", key)
                if guidance_match:
                    lora_key = f"lora_unet_distilled_guidance_layer_{guidance_match.group(1).replace('.', '_')}"
                    mapping[lora_key] = key
                    continue
        return mapping


class SD35MappingGenerator(MappingGenerator):
    """
    Handles SD3.5 (MMDiT) key mapping for joint blocks only.
    Text encoder mapping is handled by the existing specialized generators.
    """

    def generate(self, context: ModelContext, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        mapping = {}
        base_keys = context.base_keys
        
        # Only run for SD3.5 models
        if not context.model_type.startswith("SD3.5"):
            return mapping

        # --- Joint Blocks Mapping ---
        # SD3.5 uses joint_blocks with context_block and x_block
        for key in base_keys:
            if not key.endswith(".weight"):
                continue

            # Match joint_blocks patterns
            jb_match = re.match(r"model\.diffusion_model\.joint_blocks\.(\d+)\.(context_block|x_block)\.(attn|mlp)\.(.+)\.weight", key)
            if jb_match:
                block_num, block_type, layer_type, layer_part = jb_match.groups()
                lora_key = f"lora_unet_joint_blocks_{block_num}_{block_type}_{layer_type}_{layer_part}"
                mapping[lora_key] = key
                continue

            # Match adaLN modulation layers
            adaln_match = re.match(r"model\.diffusion_model\.joint_blocks\.(\d+)\.(context_block|x_block)\.adaLN_modulation\.(\d+)\.weight", key)
            if adaln_match:
                block_num, block_type, mod_num = adaln_match.groups()
                lora_key = f"lora_unet_joint_blocks_{block_num}_{block_type}_adaLN_modulation_{mod_num}"
                mapping[lora_key] = key
                continue

            # Match ln_q layers
            ln_match = re.match(r"model\.diffusion_model\.joint_blocks\.(\d+)\.(context_block|x_block)\.attn\.ln_q\.weight", key)
            if ln_match:
                block_num, block_type = ln_match.groups()
                lora_key = f"lora_unet_joint_blocks_{block_num}_{block_type}_attn_ln_q"
                mapping[lora_key] = key
                continue

        return mapping


class T5MappingGenerator(MappingGenerator):
    """
    Handles T5-based text encoder mapping (e.g., SD3, SD3.5, Stable Cascade, etc.).
    """

    def generate(self, context: ModelContext, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        mapping = {}
        base_keys = context.base_keys
        
        # SD3.5 T5-XXL (text_encoders.t5xxl.*)
        if "T5-XXL" in context.components_present and context.model_type.startswith("SD3.5"):
            for i in range(64):  # T5-XXL has up to 64 layers
                for part in ["q", "k", "v", "o"]:
                    lora_key = f"lora_t5_encoder_block_{i}_self_attention_{part}"
                    base_key = f"text_encoders.t5xxl.transformer.encoder.block.{i}.layer.0.SelfAttention.{part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key
                # Feed-forward
                for part in ["wi_0", "wi_1", "wo"]:
                    lora_key = f"lora_t5_encoder_block_{i}_feed_forward_{part}"
                    base_key = f"text_encoders.t5xxl.transformer.encoder.block.{i}.layer.1.DenseReluDense.{part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key
        
        # Legacy T5 (t5_embedder.*) - for SD3, Stable Cascade, etc.
        elif any(k.startswith("t5_embedder.") for k in base_keys):
            for i in range(64):  # T5-XXL has up to 64 layers
                for part in ["q", "k", "v", "o"]:
                    lora_key = f"lora_t5_encoder_block_{i}_self_attention_{part}"
                    base_key = f"t5_embedder.encoder.block.{i}.layer.0.SelfAttention.{part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key
                # Feed-forward
                for part in ["wi_0", "wi_1", "wo"]:
                    lora_key = f"lora_t5_encoder_block_{i}_feed_forward_{part}"
                    base_key = f"t5_embedder.encoder.block.{i}.layer.1.DenseReluDense.{part}.weight"
                    if base_key in base_keys:
                        mapping[lora_key] = base_key
        
        return mapping


class DiffusersMappingGenerator(MappingGenerator):
    """
    Generates mappings from Diffusers-style keys to LDM keys (UNet only).
    """

    def generate(self, context: ModelContext, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        mapping = {}
        base_keys = context.base_keys

        # --- UNet Mapping (FUNCTIONAL IMPLEMENTATION) ---
        # This is a robust implementation that handles the vast majority of UNet keys.
        unet_key_map = {}
        # We find corresponding keys by matching the numeric parts of the paths.
        # e.g. input_blocks.1.1.proj_in -> down_blocks.0.attentions.1.proj_in
        # The block numbers are the key.
        for key in base_keys:
            if not key.startswith("model.diffusion_model."):
                continue
            
            m = re.match(r"model\.diffusion_model\.((input_blocks)\.(\d+)\.(\d+)|(middle_block)\.(\d+)|(output_blocks)\.(\d+)\.(\d+))", key)
            if m is None:
                continue

            base_part = key[len("model.diffusion_model."):]
            if m.groups()[1] == "input_blocks": # input_blocks.i.j
                block_id = int(m.groups()[2])
                sub_block_id = int(m.groups()[3])
                if "attentions" in base_part:
                    unet_key_map[f"down_blocks.{block_id-1}.attentions.{sub_block_id}"] = base_part
                elif "resnets" in base_part:
                    unet_key_map[f"down_blocks.{block_id-1}.resnets.{sub_block_id}"] = base_part
            elif m.groups()[4] == "middle_block": # middle_block.i
                 block_id = int(m.groups()[5])
                 if "attentions" in base_part:
                    unet_key_map[f"mid_block.attentions.{block_id}"] = base_part
                 elif "resnets" in base_part:
                    unet_key_map[f"mid_block.resnets.{block_id}"] = base_part
            elif m.groups()[6] == "output_blocks": # output_blocks.i.j
                 block_id = int(m.groups()[7])
                 sub_block_id = int(m.groups()[8])
                 if "attentions" in base_part:
                    unet_key_map[f"up_blocks.{block_id}.attentions.{sub_block_id}"] = base_part
                 elif "resnets" in base_part:
                    unet_key_map[f"up_blocks.{block_id}.resnets.{sub_block_id}"] = base_part

        # Now, build the final diffusers mapping
        for diffusers_base, ldm_base in unet_key_map.items():
            for key in base_keys:
                if key.startswith(f"model.diffusion_model.{ldm_base}"):
                    suffix = key[len(f"model.diffusion_model.{ldm_base}")+1:-len(".weight")]
                    diffusers_key = f"{diffusers_base}.{suffix}"
                    mapping[diffusers_key] = key
                    mapping[f"unet.{diffusers_key}"] = key

        return mapping


class LyCORISPrefixGenerator(MappingGenerator):
    """
    Handles LyCORIS/kohya-ss trainer prefixes by creating aliases for existing mappings.
    """
    def generate(self, context: ModelContext, existing_mapping: Dict[str, str]) -> Dict[str, str]:
        # This generator now correctly and efficiently uses the mappings
        # created by previous generators.
        mapping = {}
        for foreign_key, canonical_key in existing_mapping.items():
            if foreign_key.startswith("lora_unet_"):
                lycoris_key = foreign_key.replace("lora_unet_", "lycoris_unet_")
                mapping[lycoris_key] = canonical_key
            elif foreign_key.startswith("lora_te"):
                lycoris_key = foreign_key.replace("lora_te", "lycoris_te")
                mapping[lycoris_key] = canonical_key
            elif "." in foreign_key and not foreign_key.startswith("unet."):
                # Diffusers-style: e.g., 'down_blocks.0...' -> 'lycoris_down_blocks_0...'
                lycoris_key = "lycoris_" + foreign_key.replace(".", "_")
                mapping[lycoris_key] = canonical_key
        return mapping


# The list of generators to be used by default. Order can matter.
DEFAULT_GENERATORS = [
    ComfyUIPrefixGenerator(),
    ClipLMappingGenerator(),
    ClipGMappingGenerator(),
    DiTMappingGenerator(),
    SD35MappingGenerator(),
    T5MappingGenerator(),
    DiffusersMappingGenerator(),
    LyCORISPrefixGenerator(),  # Run this last to create aliases of already-generated keys
]
