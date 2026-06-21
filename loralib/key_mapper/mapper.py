# File: loralib/key_mapper/mapper.py (Refactored)

import logging
from pathlib import Path
from typing import List, Optional, Set, Dict
from collections import defaultdict

from .types import (
    ModelContext,
    MappingResult,
)  # Import from types.py
from .generators import DEFAULT_GENERATORS, MappingGenerator
import safetensors.torch

logger = logging.getLogger(__name__)


class KeyMapper:
    SUFFIXES: Set[str] = {
        ".alpha", ".dora_scale", ".lora_down.weight", ".lora_up.weight",
        ".lora_mid.weight", ".lora_A.weight", ".lora_B.weight",
        ".lora_proj_down", ".lora_proj_up",
        ".hada_w1_a.weight", ".hada_w1_b.weight", ".hada_w2_a.weight",
        ".hada_w2_b.weight", ".lokr_w1.weight", ".lokr_w2.weight",
        ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b",
        ".t1.weight", ".t2.weight", ".ia3_input_mask.weight", ".diff", ".diff_w", ".diff_b",
        ".weight", ".bias"
    }

    def __init__(
        self, base_model_path: Path, generators: List[MappingGenerator] = None
    ):
        if generators is None:
            generators = DEFAULT_GENERATORS

        base_model_path = Path(base_model_path)
        logger.info(f"Initializing KeyMapper with base model: {base_model_path.name}")
        self.base_model_path = base_model_path
        self.generators = generators  # Store the generators for tracing
        self.stats = defaultdict(int)

        self.context = self._initialize_context(base_model_path)
        self.final_mapping: Dict[str, str] = {}

        logger.info("Building comprehensive key map from generators...")
        for generator in generators:
            new_mappings = generator.generate(self.context, self.final_mapping)
            self._merge_mappings(new_mappings, generator)
            logger.info(
                f"  - Ran {generator.__class__.__name__}, added/updated {len(new_mappings)} mappings."
            )

        logger.info(
            f"Total unique mappings in 'Rosetta Stone': {len(self.final_mapping)}"
        )
        
        # Longest-first matching is required because several adapter suffixes also
        # end in the generic ".weight" suffix.
        self.engine_suffixes = sorted(self.SUFFIXES, key=len, reverse=True)
        
        # Populate stats for reporting
        self.stats['base_keys_found'] = len(self.context.base_keys)
        self.stats['detected_model_type'] = self.context.model_type
        self.stats['detected_components'] = ", ".join(sorted(list(self.context.components_present)))
        self.stats['total_mappings'] = len(self.final_mapping)

    def _merge_mappings(
        self, new_mappings: Dict[str, str], generator: MappingGenerator
    ) -> None:
        for foreign_key, canonical_key in new_mappings.items():
            existing = self.final_mapping.get(foreign_key)
            if existing is not None and existing != canonical_key:
                raise ValueError(
                    f"{generator.__class__.__name__} generated conflicting mapping "
                    f"for {foreign_key!r}: {existing!r} != {canonical_key!r}"
                )
        self.final_mapping.update(new_mappings)

    def _initialize_context(self, model_path: Path) -> ModelContext:
        try:
            with safetensors.torch.safe_open(
                model_path, framework="pt", device="cpu"
            ) as f:
                base_keys = set(f.keys())
        except Exception as e:
            logger.error(f"Failed to open or read base model {model_path}: {e}")
            raise

        model_type, components = self._detect_model_type_and_components(base_keys)

        # The context now directly holds the keys, no more nested 'maps'.
        context = ModelContext(
            model_type=model_type,
            components_present=components,
            base_keys=base_keys,
            maps={}, # Keep for future complex pre-calculated maps
        )
        return context
    
    def _detect_model_type_and_components(self, keys: Set[str]) -> tuple[str, Set[str]]:
        """
        Infers model type and components from the tensor keys. This is the
        lightweight, self-contained model detection logic.
        """
        components = set()
        model_type = "Unknown"

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("--- Analyzing model keys for type detection ---")
            sample_keys = list(keys)[:10]
            logger.debug(f"Analyzing {len(keys)} keys. Sample:")
            for k in sample_keys:
                logger.debug(f"  - {k}")

        # --- Detection based on key prefixes ---

        # SD3.5 (MMDiT) - Most specific first
        if any(k.startswith("model.diffusion_model.joint_blocks.") for k in keys):
            components.add("MMDiT")
            # Check for text encoders
            if any(k.startswith("text_encoders.clip_l.") for k in keys):
                components.add("CLIP-L")
            if any(k.startswith("text_encoders.clip_g.") for k in keys):
                components.add("CLIP-G")
            if any(k.startswith("text_encoders.t5xxl.") for k in keys):
                components.add("T5-XXL")
            
            # Determine variant based on architectural characteristics
            # We need to load the model to get the y_embedder shape
            # For now, we'll detect as generic SD3.5 and let the generator handle specifics
            model_type = "SD3.5"
            return model_type, components

        # Legacy SD3 (MMDiT) - older format
        if any(k.startswith("transformer.joint_blocks.") for k in keys):
            components.add("MMDiT")
            if any(k.startswith("t5_embedder.") for k in keys):
                components.add("T5-XXL")
            if any(k.startswith("clip_l_embedder.") for k in keys):
                components.add("CLIP-L")
            model_type = "SD3"
            return model_type, components

        # Stable Cascade
        if any(k.startswith("diffusion_model.clip_txt_mapper.") for k in keys):
            return "StableCascade_C", {"CLIP-G", "UNet-C"}
        if any(k.startswith("diffusion_model.clip_mapper.") for k in keys):
            return "StableCascade_B", {"CLIP-G", "UNet-B"}

        # SDXL
        is_sdxl_te1 = any(
            k.startswith("conditioner.embedders.0.transformer.") for k in keys
        )
        is_sdxl_te2 = any(k.startswith("conditioner.embedders.1.model.") for k in keys)

        if is_sdxl_te1 and is_sdxl_te2:
            model_type = "SDXL-Base"
            components.add("CLIP-L")
            components.add("CLIP-G")
        elif is_sdxl_te2:  # Only TE2 present
            model_type = "SDXL-Refiner"
            components.add("CLIP-G")

        # SD1.x / SD2.x
        if any(k.startswith("cond_stage_model.transformer.") for k in keys):
            if model_type == "Unknown":
                model_type = "SD1.x/2.x"
            components.add("CLIP-L")

        if any(k.startswith("model.diffusion_model.") for k in keys):
            components.add("UNet")

        # FLUX and Chroma checkpoints occur both bare and under common wrappers.
        dit_keys = set()
        for key in keys:
            for prefix in ("model.diffusion_model.", "diffusion_model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            dit_keys.add(key)

        has_double_blocks = any(k.startswith("double_blocks.") for k in dit_keys)
        has_single_blocks = any(k.startswith("single_blocks.") for k in dit_keys)

        if has_double_blocks or has_single_blocks:
            components.add("DiT")
            # Check for the unique layer to differentiate them
            if any(k.startswith("distilled_guidance_layer.") for k in dit_keys):
                model_type = "Chroma"
            elif any(k.startswith("double_stream_modulation_") for k in dit_keys):
                model_type = "FLUX.2-Klein"
            # FLUX's unique feature is modulation layers
            elif any(
                "modulation.lin.weight" in k and k.startswith("single_blocks")
                for k in dit_keys
            ):
                model_type = "FLUX-dev"

            # FLUX/Chroma also include text encoders, let's detect them
            if model_type != "Unknown":
                if any(k.startswith("clip_embedder.") for k in keys):
                    components.add("CLIP-L")
                if any(k.startswith("t5_embedder.") for k in keys):
                    components.add("T5-XXL")

        return model_type, components

    def _strip_suffix(self, raw_key: str) -> Optional[tuple[str, str]]:
        for suffix in self.engine_suffixes:
            if raw_key.endswith(suffix):
                return raw_key[:-len(suffix)], suffix
        return None

    def map_from_lora(self, raw_key: str) -> Optional[MappingResult]:
        stripped = self._strip_suffix(raw_key)
        if not stripped:
            return None

        lora_key_base, suffix = stripped
        canonical_key = self.final_mapping.get(lora_key_base)

        if canonical_key:
            return MappingResult(
                canonical_key=canonical_key,
                lora_key_base=lora_key_base,
                suffix=suffix,
                matched_rule="DictionaryLookup",  # The 'rule' is now just one thing
            )
        return None

    def get_all_mappings(self) -> Dict[str, str]:
        return self.final_mapping


# Old key mapper
# class KeyMapper:

#     def __init__(self, base_model_path: Path, rules: List[Rule] = None):
#         if rules is None:
#             # Load our default set of rules
#             rules = DEFAULT_RULES

#         logger.info(f"Initializing KeyMapper with base model: {base_model_path.name}")
#         self.base_model_path = base_model_path
#         self.stats = defaultdict(int)

#         self.context = self._initialize_context(base_model_path)
#         self.stats['base_keys_found'] = len(self.context.maps.get('base_keys', set()))
#         self.stats['detected_model_type'] = self.context.model_type
#         self.stats['detected_components'] = ", ".join(sorted(list(self.context.components_present)))
#         self.stats['diffusers_map_entries'] = len(self.context.maps.get('diffusers_to_ldm', {}))

#         self.engine = RuleEngine(rules, self.SUFFIXES)

#         logger.info("KeyMapper initialization complete.")
#         for stat, count in self.stats.items():
#             logger.info(f"  - {stat:<25}: {count}")
