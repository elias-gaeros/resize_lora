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

            base_stem = base_key[: -len(".weight")]
            mapping[base_stem] = base_key
            if base_stem.startswith("model.diffusion_model."):
                unwrapped = base_stem[len("model.") :]
                mapping[unwrapped] = base_key
            elif "DiT" in context.components_present:
                mapping[f"diffusion_model.{base_stem}"] = base_key

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

        prefix = "model.diffusion_model."
        ldm_keys = {
            key[len(prefix) :]: key
            for key in base_keys
            if key.startswith(prefix) and key.endswith(".weight")
        }

        input_blocks = sorted(
            {int(match.group(1)) for key in ldm_keys if (match := re.match(r"input_blocks\.(\d+)\.", key))}
        )
        input_locations = {}
        stage = slot = 0
        for block in input_blocks:
            if block == 0:
                continue
            input_locations[block] = (stage, slot)
            if any(key.startswith(f"input_blocks.{block}.0.op.") for key in ldm_keys):
                stage, slot = stage + 1, 0
            else:
                slot += 1

        output_blocks = sorted(
            {int(match.group(1)) for key in ldm_keys if (match := re.match(r"output_blocks\.(\d+)\.", key))}
        )
        output_locations = {}
        stage = slot = 0
        for block in output_blocks:
            output_locations[block] = (stage, slot)
            if any(re.match(rf"output_blocks\.{block}\.\d+\.conv\.", key) for key in ldm_keys):
                stage, slot = stage + 1, 0
            else:
                slot += 1

        for ldm_key, canonical_key in ldm_keys.items():
            diffusers_key = self._to_diffusers_key(
                ldm_key, input_locations, output_locations
            )
            if diffusers_key is None:
                continue
            diffusers_stem = diffusers_key[: -len(".weight")]
            aliases = {diffusers_stem, f"unet.{diffusers_stem}"}
            if diffusers_stem.endswith(".to_out.0"):
                aliases.add(diffusers_stem[:-2])
                aliases.add(f"unet.{diffusers_stem[:-2]}")
            underscored = diffusers_stem.replace(".", "_")
            aliases.update({f"lora_unet_{underscored}", f"lycoris_{underscored}"})
            for alias in aliases:
                mapping[alias] = canonical_key

        return mapping

    @staticmethod
    def _to_diffusers_key(key, input_locations, output_locations):
        basic = {
            "input_blocks.0.0.weight": "conv_in.weight",
            "out.0.weight": "conv_norm_out.weight",
            "out.2.weight": "conv_out.weight",
            "time_embed.0.weight": "time_embedding.linear_1.weight",
            "time_embed.2.weight": "time_embedding.linear_2.weight",
            "label_emb.0.0.weight": "add_embedding.linear_1.weight",
            "label_emb.0.2.weight": "add_embedding.linear_2.weight",
        }
        if key in basic:
            return basic[key]

        resnet_parts = {
            "in_layers.0": "norm1",
            "in_layers.2": "conv1",
            "emb_layers.1": "time_emb_proj",
            "out_layers.0": "norm2",
            "out_layers.3": "conv2",
            "skip_connection": "conv_shortcut",
        }

        match = re.match(r"input_blocks\.(\d+)\.(\d+)\.(.+)", key)
        if match:
            block, subblock, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
            if block not in input_locations:
                return None
            stage, slot = input_locations[block]
            if subblock == 0 and suffix.startswith("op."):
                return f"down_blocks.{stage}.downsamplers.0.conv.{suffix[3:]}"
            if subblock == 0:
                suffix = DiffusersMappingGenerator._translate_resnet(suffix, resnet_parts)
                return f"down_blocks.{stage}.resnets.{slot}.{suffix}"
            return f"down_blocks.{stage}.attentions.{slot}.{suffix}"

        match = re.match(r"middle_block\.(\d+)\.(.+)", key)
        if match:
            block, suffix = int(match.group(1)), match.group(2)
            if block == 1:
                return f"mid_block.attentions.0.{suffix}"
            if block in (0, 2):
                suffix = DiffusersMappingGenerator._translate_resnet(suffix, resnet_parts)
                return f"mid_block.resnets.{block // 2}.{suffix}"

        match = re.match(r"output_blocks\.(\d+)\.(\d+)\.(.+)", key)
        if match:
            block, subblock, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
            if block not in output_locations:
                return None
            stage, slot = output_locations[block]
            if suffix.startswith("conv."):
                return f"up_blocks.{stage}.upsamplers.0.{suffix}"
            if subblock == 0:
                suffix = DiffusersMappingGenerator._translate_resnet(suffix, resnet_parts)
                return f"up_blocks.{stage}.resnets.{slot}.{suffix}"
            return f"up_blocks.{stage}.attentions.{slot}.{suffix}"
        return None

    @staticmethod
    def _translate_resnet(suffix, translations):
        for ldm_part, diffusers_part in translations.items():
            if suffix == f"{ldm_part}.weight":
                return f"{diffusers_part}.weight"
        return suffix


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
