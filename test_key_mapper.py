import argparse
import sys
import logging
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm

# Make sure the local loralib is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from compressed_sft_index import CompressedIndex
from loralib.sources import AdapterFileSource
from loralib.key_mapper import KeyMapper
from loralib.key_mapper.generators import ClipGMappingGenerator, ClipLMappingGenerator

# --- Configuration ---
# Use a more detailed format for logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
COMPATIBILITY_THRESHOLD = 0.50

# --- Test Harness Classes and Functions ---

class CheckpointAssessor:
    """
    A more sophisticated classifier to determine if a schema represents a
    base model checkpoint or an adapter.
    """
    # Core prefixes that MUST be present for a file to be a base model.
    SD15_UNET = "model.diffusion_model."
    SD15_TE = "cond_stage_model.transformer."
    SDXL_UNET = "model.diffusion_model."
    SDXL_TE1 = "conditioner.embedders.0.transformer."
    SDXL_TE2 = "conditioner.embedders.1.model."

    # Suffixes that strongly indicate a file is an adapter.
    ADAPTER_SUFFIXES = {
        ".alpha", ".lora_down.weight", ".lora_up.weight", ".lora_A.weight",
        ".lora_B.weight", ".hada_w1_a.weight", ".hada_w1_b.weight",
        ".hada_w2_a.weight", ".hada_w2_b.weight", ".lokr_w1.weight",
        ".lokr_w2.weight", ".diff",
    }

    def __init__(self, keys: [Dict, List, Set]):
        if isinstance(keys, dict):
            self.key_set = set(keys.keys())
        else:
            self.key_set = set(keys)

    def _has_adapter_keys(self) -> bool:
        """Check if any keys have common adapter suffixes."""
        return any(
            any(key.endswith(suffix) for suffix in self.ADAPTER_SUFFIXES)
            for key in self.key_set
        )

    def _is_sdxl_base(self) -> bool:
        """Check for the presence of all core SDXL components."""
        has_unet = any(k.startswith(self.SDXL_UNET) for k in self.key_set)
        has_te1 = any(k.startswith(self.SDXL_TE1) for k in self.key_set)
        has_te2 = any(k.startswith(self.SDXL_TE2) for k in self.key_set)
        return has_unet and has_te1 and has_te2

    def _is_sd15_base(self) -> bool:
        """Check for the presence of all core SD1.5 components."""
        has_unet = any(k.startswith(self.SD15_UNET) for k in self.key_set)
        has_te = any(k.startswith(self.SD15_TE) for k in self.key_set)
        return has_unet and has_te

    def classify(self) -> Tuple[str, str]:
        """
        Classify the schema as CHECKPOINT, ADAPTER, or JUNK based on key structure.
        Returns: (classification, reason)
        """
        is_adapter = self._has_adapter_keys()
        is_sdxl = self._is_sdxl_base()
        is_sd15 = self._is_sd15_base()

        if is_sdxl:
            return "CHECKPOINT", "Contains all required SDXL base model key prefixes."
        if is_sd15:
            return "CHECKPOINT", "Contains all required SD1.5 base model key prefixes."
        
        if is_adapter:
            return "ADAPTER", "Contains common adapter key suffixes."

        return "JUNK", "Ambiguous structure: No adapter keys and not a known base model format."


class Reporter:
    """Reporter with detailed failure analysis and pattern detection."""

    def __init__(self):
        # Basic stats
        self.stats = defaultdict(int)
        self.per_base_results = defaultdict(dict)

        # failure tracking
        self.global_failures = defaultdict(int)
        self.failure_patterns = defaultdict(int)  # Group failures by common patterns
        self.prefix_failures = defaultdict(int)  # Group failures by key prefixes

        # Compatibility analysis
        self.no_match_reasons = defaultdict(int)
        self.compatibility_scores = defaultdict(list)  # Track score distribution
        self.compatibility_details = defaultdict(dict)  # Detailed breakdown per adapter

        # Classification tracking
        self.junk_reasons = defaultdict(int)
        self.ambiguous_rejections = [] # For detailed JUNK logging
        self.low_score_rejections = [] # For detailed low-score logging

        # Generator performance tracking
        self.generator_stats = defaultdict(lambda: defaultdict(int))

    def add_compatibility_score(
        self, adapter_path: str, base_name: str, score: float, details: dict
    ):
        """Track detailed compatibility information."""
        self.compatibility_scores[base_name].append(score)
        self.compatibility_details[adapter_path] = {
            "base_name": base_name,
            "score": score,
            "details": details,
        }

    def add_failure(self, key_base: str, adapter_path: str = None):
        """Track a mapping failure with pattern analysis."""
        self.global_failures[key_base] += 1

        # Pattern analysis: group by common patterns (optimized)
        pattern = self._extract_pattern_fast(key_base)
        self.failure_patterns[pattern] += 1

        # Prefix analysis: group by key prefixes (optimized)
        prefix = self._extract_prefix_fast(key_base)
        self.prefix_failures[prefix] += 1

    def _extract_pattern_fast(self, key: str) -> str:
        """Extract a pattern from a key for grouping similar failures (optimized)."""
        # Fast path: most common patterns
        if (
            "lora_te" in key
            and "text_model_encoder_layers" in key
            and "self_attn" in key
        ):
            return "lora_teN_text_model_encoder_layers_N_self_attn_PROJ"
        if "lora_unet" in key and "input_blocks" in key:
            return "lora_unet_input_blocks_N_PROJ"
        if "lora_unet" in key and "output_blocks" in key:
            return "lora_unet_output_blocks_N_PROJ"

        # Fallback: use regex but only for complex cases
        pattern = re.sub(r"\d+", "N", key)
        pattern = re.sub(r"(fc1|fc2|q_proj|k_proj|v_proj|out_proj)", "PROJ", pattern)
        pattern = re.sub(r"(attn1|attn2)", "ATTN", pattern)
        return pattern

    def _extract_prefix_fast(self, key: str) -> str:
        """Extract the prefix from a key (optimized)."""
        # Fast path: most common prefixes
        if key.startswith("lora_te"):
            return "lora_te"
        if key.startswith("lora_unet"):
            return "lora_unet"
        if key.startswith("lycoris_"):
            return "lycoris_"

        # Fallback: use string operations
        if "_" in key:
            return key.split("_")[0] + "_"
        elif "." in key:
            return key.split(".")[0] + "."
        return key[:10] + "..."  # Truncate long keys

    def add_ambiguous_rejection(self, path: str, reason: str, schema_info: dict):
        """Log a file rejected for being ambiguous."""
        if len(self.ambiguous_rejections) < 10: # Log first 10 examples
             self.ambiguous_rejections.append((path, reason, schema_info))

    def add_low_score_rejection(self, path: str, score: float, base_name: str):
        """Log a file rejected for low compatibility score."""
        if len(self.low_score_rejections) < 10: # Log first 10 examples
            self.low_score_rejections.append((path, score, base_name))

    def print_summary(self, key_mappers):
        print("\n" + "=" * 50 + " Test Harness Report " + "=" * 50)

        # Base Model Analysis
        print("\n--- Base Model Analysis ---")
        if not key_mappers:
            print("  - No base models were successfully loaded.")
        for path, km in key_mappers.items():
            print(f"  - Base: {path.name}")
            for stat, count in km.stats.items():
                print(f"    - {stat:<25}: {count}")

        # File Classification Summary
        print("\n--- File Classification & Testing Summary ---")
        print(f"  - Total Files in Index:      {self.stats['total_files']}")
        print(f"  - Checkpoints (ignored):     {self.stats['classified_checkpoint']}")
        print(f"  - Adapters Found:            {self.stats['classified_adapter']}")
        print(f"  - Junk/Corrupt (ignored):    {self.stats['classified_junk']}")
        print(f"  - Adapters Tested:           {self.stats['adapters_tested']}")
        print(f"  - Adapters without match:    {self.stats['no_compatible_base']}")

        # Compatibility Analysis
        if self.compatibility_scores:
            print("\n--- Compatibility Score Analysis ---")
            for base_name, scores in self.compatibility_scores.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(
                        f"  ▶ {base_name:<25} | Avg Score: {avg_score:.3f} | {len(scores)} adapters"
                    )
                    # Score distribution
                    score_ranges = [
                        (0.0, 0.1),
                        (0.1, 0.3),
                        (0.3, 0.5),
                        (0.5, 0.7),
                        (0.7, 1.0),
                    ]
                    for low, high in score_ranges:
                        count = sum(1 for s in scores if low <= s < high)
                        if count > 0:
                            print(f"    - Score {low:.1f}-{high:.1f}: {count} adapters")



        # Detailed Failure Reasons
        if self.ambiguous_rejections:
            print("\n--- Ambiguous Structure Rejection Samples ---")
            for path, reason, s_info in self.ambiguous_rejections:
                print(f"  - Path: {path}")
                print(f"    Reason: {reason} (Keys: {s_info.get('key_count', 'N/A')})")

        if self.low_score_rejections:
            print("\n--- Low Compatibility Score Rejection Samples (Score > 0) ---")
            for path, score, base_name in self.low_score_rejections:
                print(f"  - Path: {path}")
                print(f"    Score: {score:.3f} against {base_name}")

        if self.junk_reasons:
            print("\n--- Reasons for Junk/Corrupt Classification ---")
            for reason, count in sorted(
                self.junk_reasons.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                print(f"    - {count:>5} times: {reason}")

        if self.no_match_reasons:
            print("\n--- Reasons for No Compatible Base Found ---")
            for reason, count in sorted(
                self.no_match_reasons.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                print(f"    - {count:>5} times: {reason}")

        if not self.stats['adapters_tested']:
            print('--- No compatible adapters were tested ---')
            return

        # Failure Pattern Analysis
        if self.failure_patterns:
            print("\n--- Top 10 Failure Patterns ---")
            sorted_patterns = sorted(
                self.failure_patterns.items(), key=lambda x: x[1], reverse=True
            )
            for pattern, count in sorted_patterns[:10]:
                print(f"  - {count:>4} failures: {pattern}")

        # Prefix-based Failure Analysis
        if self.prefix_failures:
            print("\n--- Failure Analysis by Key Prefix ---")
            sorted_prefixes = sorted(
                self.prefix_failures.items(), key=lambda x: x[1], reverse=True
            )
            for prefix, count in sorted_prefixes[:10]:
                print(f"  - {count:>4} failures: {prefix}")

        # Per-Base Model Results
        print("\n--- Per-Base Model Mapping Results ---")
        if not self.per_base_results:
            print(
                "  - No adapters were successfully matched to a base model for testing."
            )
        for base_name, data in sorted(self.per_base_results.items()):
            total_mapped = sum(cat_data['mapped'] for cat_data in data.values() if isinstance(cat_data, dict))
            total_keys = sum(cat_data['total'] for cat_data in data.values() if isinstance(cat_data, dict))
            success_rate = (total_mapped / total_keys) * 100 if total_keys > 0 else 0
            
            print(f"  ▶ {base_name:<25} | Tested {data['files_tested']} adapters | Overall Success: {success_rate:.2f}%")
            
            # Print categorical breakdown
            for category in sorted(data.keys()):
                if category == 'files_tested': continue
                cat_data = data[category]
                cat_success = (cat_data['mapped'] / cat_data['total']) * 100 if cat_data['total'] > 0 else 0
                print(f"    - {category:<20}: {cat_success:>6.2f}% ({cat_data['mapped']}/{cat_data['total']} keys)")

        # Top Individual Failures
        print("\n--- Top 10 Individual Key Mapping Failures ---")
        if not self.global_failures:
            print("  No mapping failures found across all tested adapters!")
        else:
            sorted_failures = sorted(
                self.global_failures.items(), key=lambda item: item[1], reverse=True
            )
            for key_base, count in sorted_failures[:10]:
                print(f"  - FAILED {count:>4} times: {key_base}")

        print("=" * 131)


def classify_file(loader: CompressedIndex, path: str) -> tuple[str, dict]:
    """Classifies a file as CHECKPOINT, ADAPTER, or JUNK."""
    try:
        instance = loader.instances.get(path)
        if not instance:
            return "JUNK", {"reason": "Path not found in index instances."}

        schema_id = instance["schema_id"]
        schema_info = loader.get_schema_info(schema_id)
        key_map = loader.get_key_map(schema_id)

        if not key_map:
            return "JUNK", {"reason": "Schema has no keys"}

        if schema_info["type"] == "base" and schema_info["key_count"] > 500:
            return "CHECKPOINT", schema_info

        if any(k.endswith((".alpha", ".lora_down.weight")) for k in key_map):
            return "ADAPTER", schema_info

        if schema_info["type"] == "base":
            return "CHECKPOINT", schema_info

        return "JUNK", {"reason": "Ambiguous structure (not base, no adapter keys)"}
    except Exception as e:
        return "JUNK", {"reason": f"Processing error: {e}"}


def calculate_compatibility_detailed(
    key_mapper: KeyMapper, source_keys: set, _all_mappable_keys_cache=None
) -> Tuple[float, dict]:
    """
    Compatibility check with detailed breakdown and architecture awareness.

    Returns:
        Tuple of (score, details_dict)
    """
    # Strip suffixes to get module keys
    source_module_keys = set()
    unstripable_keys = set()

    for key in source_keys:
        stripped = key_mapper._strip_suffix(key)
        if stripped:
            source_module_keys.add(stripped[0])
        else:
            unstripable_keys.add(key)

    if not source_module_keys:
        return 0.0, {
            "total_keys": len(source_keys),
            "stripable_keys": 0,
            "unstripable_keys": len(unstripable_keys),
            "matched_keys": 0,
            "unmatched_keys": 0,
            "unstripable_examples": list(unstripable_keys)[:3],
        }

    # Get all mappable keys from the KeyMapper (use cache if provided)
    if _all_mappable_keys_cache is None:
        all_mappable_keys = set(key_mapper.get_all_mappings().keys())
    else:
        all_mappable_keys = _all_mappable_keys_cache

    # Find intersection
    matched_keys = source_module_keys.intersection(all_mappable_keys)
    unmatched_keys = source_module_keys - all_mappable_keys

    # Architecture-aware compatibility check
    base_context = key_mapper.context
    architecture_penalty = 0.0
    
    # Check for architectural mismatches
    has_sdxl_unet = any("output_blocks" in key for key in source_module_keys)
    has_sd3_unet = any("joint_blocks" in key for key in source_module_keys)
    has_flux_unet = any("double_blocks" in key or "single_blocks" in key for key in source_module_keys)
    
    # Check for FLUX vs Chroma specific mismatches
    has_flux_modulation = any("modulation_lin" in key or "_mod_lin" in key for key in source_module_keys)
    has_chroma_guidance = any("distilled_guidance_layer" in key for key in source_module_keys)
    
    # Apply architecture penalties
    if has_sdxl_unet and base_context.model_type == "SD3.5":
        # SDXL UNet keys are completely incompatible with SD3.5
        architecture_penalty = 0.5
    elif has_sd3_unet and (
        base_context.model_type.startswith("SDXL")
        or base_context.model_type == "SD1.x/2.x"
    ):
        # SD3.5 UNet keys are completely incompatible with SDXL/SD1.5
        architecture_penalty = 0.5
    elif has_flux_unet and base_context.model_type not in ["FLUX-dev", "Chroma"]:
        # FLUX/Chroma UNet keys are incompatible with other architectures
        architecture_penalty = 0.5
    elif has_flux_modulation and base_context.model_type == "Chroma":
        # FLUX modulation keys are incompatible with Chroma
        architecture_penalty = 0.5
    elif has_chroma_guidance and base_context.model_type == "FLUX-dev":
        # Chroma guidance keys are incompatible with FLUX
        architecture_penalty = 0.5
    
    # Calculate base score
    base_score = len(matched_keys) / len(source_module_keys) if source_module_keys else 0.0
    
    # Apply architecture penalty
    final_score = max(0.0, base_score - architecture_penalty)

    # Prepare detailed breakdown
    details = {
        "total_keys": len(source_keys),
        "stripable_keys": len(source_module_keys),
        "unstripable_keys": len(unstripable_keys),
        "matched_keys": len(matched_keys),
        "unmatched_keys": len(unmatched_keys),
        "base_score": base_score,
        "architecture_penalty": architecture_penalty,
        "score": final_score,
        "unstripable_examples": list(unstripable_keys)[:3],
        "unmatched_examples": list(unmatched_keys)[:5],
    }

    return final_score, details


def debug_key_mapping(key_mapper: KeyMapper, key: str) -> None:
    """Debug a specific key mapping attempt."""
    print(f"\n--- Debugging Key: {key} ---")

    # Try to strip suffix
    stripped = key_mapper._strip_suffix(key)
    if not stripped:
        print(f"  ❌ Could not strip suffix from key")
        print(
            f"  Available suffixes: {sorted(key_mapper.SUFFIXES, key=len, reverse=True)}"
        )
        return

    lora_key_base, suffix = stripped
    print(f"  ✅ Stripped suffix: '{suffix}'")
    print(f"  📝 LoRA key base: '{lora_key_base}'")

    # Check if it's in our mapping
    all_mappings = key_mapper.get_all_mappings()
    if lora_key_base in all_mappings:
        canonical_key = all_mappings[lora_key_base]
        print(f"  ✅ Found mapping: '{lora_key_base}' -> '{canonical_key}'")
    else:
        print(f"  ❌ No mapping found for '{lora_key_base}'")

        # Show similar keys for debugging
        similar_keys = [
            k for k in all_mappings.keys() if k.startswith(lora_key_base[:20])
        ]
        if similar_keys:
            print(f"  🔍 Similar keys found:")
            for similar in similar_keys[:5]:
                print(f"    - {similar}")
        else:
            print(f"  🔍 No similar keys found")


def trace_key_generation(key_mapper: KeyMapper, lora_key_base: str) -> None:
    """
    Traces how a lora_key_base is processed by each generator in the KeyMapper.
    """
    print(f"\n--- Tracing Generation for LoRA Key Base: '{lora_key_base}' ---")

    context = key_mapper.context
    for generator in key_mapper.generators:
        generator_name = generator.__class__.__name__
        print(f"\n▶ Generator: {generator_name}")

        try:
            if isinstance(generator, (ClipGMappingGenerator, ClipLMappingGenerator)):
                print(f"  - Checking if '{lora_key_base}' matches patterns...")
                # Simulate running the generator to see if it produces a match
                single_gen_map = generator.generate(context, {})
                if lora_key_base in single_gen_map:
                    print(f"  - ✅ Result: Generator produced mapping -> '{single_gen_map[lora_key_base]}'")
                else:
                    print(f"  - ❌ Result: Generator did not produce a mapping for this key.")
                    # --- ENHANCED DIAGNOSTICS ---
                    if "text_projection" in lora_key_base:
                        print("    - DIAGNOSTIC: Searching for 'text_projection' keys in base model context...")
                        found_proj_keys = [k for k in context.base_keys if "text_projection" in k]
                        if found_proj_keys:
                            print("    - Found potential matches in base model:")
                            for p_key in found_proj_keys:
                                print(f"      - {p_key}")
                        else:
                            print("    - No 'text_projection' keys found in base model context.")
            else:
                print(f"  - Tracing not implemented for this generator type.")

        except Exception as e:
            print(f"  - 💥 Error during trace: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Loralib v2 Data-Driven Test Harness"
    )
    parser.add_argument(
        "compressed_index", type=Path, help="Path to compressed_sft_index.json"
    )
    parser.add_argument(
        "base_model_paths",
        type=Path,
        nargs="+",
        help="Path(s) to base model checkpoints or directories containing them.",
    )
    parser.add_argument("--debug-key", type=str, help="Debug a specific key mapping")
    parser.add_argument(
        "--trace-key", type=str, help="Trace the generation process for a LoRA key base."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    reporter = Reporter()

    # --- Step 1 & 2: Base Model and KeyMapper Initialization ---
    base_model_files = []
    for path in args.base_model_paths:
        if path.is_dir():
            base_model_files.extend(path.glob("*.safetensors"))
            logging.info(f"Discovered *.safetensors in directory: {path}")
        elif path.is_file() and path.suffix == ".safetensors":
            base_model_files.append(path)
            logging.info(f"Found specified base model file: {path}")
        else:
            logging.warning(
                f"Path '{path}' is not a valid .safetensors file or directory. Skipping."
            )

    key_mappers = {}
    if not base_model_files:
        logging.error("No base model files were found. Cannot proceed.")
        reporter.print_summary(key_mappers)
        sys.exit(1)

    logging.info(
        f"\n--- Initializing KeyMappers for {len(base_model_files)} base models ---"
    )
    for model_path in base_model_files:
        logging.info(f"Attempting to load base model: {model_path.name}")
        try:
            print("-" * 20 + f" Loading {model_path.name} " + "-" * 20)
            key_mappers[model_path] = KeyMapper(model_path)
            print("-" * (42 + len(model_path.name)))
        except Exception as e:
            logging.error(
                f"FATAL: Failed to initialize KeyMapper for {model_path.name}: {e}",
                exc_info=True,
            )

    if not key_mappers:
        logging.error(
            "Could not initialize any KeyMappers from the provided base models."
        )
        reporter.print_summary(key_mappers)
        sys.exit(1)
    logging.info("--- KeyMapper initialization complete ---")

    # Handle debug mode
    if args.debug_key:
        print(f"\n🔍 Debug Mode: Analyzing key '{args.debug_key}'")
        for path, km in key_mappers.items():
            print(f"\n--- KeyMapper for {path.name} ---")
            debug_key_mapping(km, args.debug_key)
        return

    # Handle trace mode
    if args.trace_key:
        print(f"\n🔍 Trace Mode: Analyzing LoRA key base '{args.trace_key}'")
        for path, km in key_mappers.items():
            print(f"\n--- KeyMapper for {path.name} ---")
            trace_key_generation(km, args.trace_key)
        return

    # --- Step 3: Load the compressed index ---
    logging.info(f"\n--- Loading and Processing Compressed Index ---")
    try:
        loader_index = CompressedIndex(args.compressed_index)
        logging.info(
            f"Index for {len(loader_index.instances)} files loaded successfully."
        )
    except Exception as e:
        logging.error(
            f"FATAL: Could not load or parse the compressed index file '{args.compressed_index}': {e}",
            exc_info=True,
        )
        sys.exit(1)

    # --- Step 4: Pre-compute schema caches for performance ---
    logging.info(f"\n--- Pre-computing schema caches ---")
    schema_cache = {}  # schema_id -> (key_map, schema_info)

    # Group files by schema for efficient processing first
    schema_to_paths = defaultdict(list)
    for path_str, instance in loader_index.instances.items():
        schema_to_paths[instance["schema_id"]].append(path_str)

    # Get unique schema IDs from the grouped paths
    unique_schema_ids = set(schema_to_paths.keys())
    logging.info(f"Found {len(unique_schema_ids)} unique schemas to cache")

    # Pre-compute all schema data
    for schema_id in unique_schema_ids:
        try:
            key_map = loader_index.get_key_map(schema_id)
            schema_info = loader_index.get_schema_info(schema_id)
            schema_cache[schema_id] = (key_map, schema_info)
        except Exception as e:
            logging.warning(f"Failed to cache schema {schema_id}: {e}")
            schema_cache[schema_id] = ([], {"type": "error", "reason": str(e)})

    logging.info(f"Successfully cached {len(schema_cache)} schemas")

    # Pre-compute all_mappable_keys cache for each KeyMapper
    key_mapper_caches = {}
    for base_path, km in key_mappers.items():
        key_mapper_caches[base_path] = set(km.get_all_mappings().keys())

    reporter.stats["total_files"] = len(loader_index.instances)

    # --- Step 5: Process files in a single pass, factored by schema ---
    logging.info(f"\n--- Classifying and processing {len(schema_to_paths)} schemas ---")

    with tqdm(total=len(schema_to_paths), desc="Processing Schemas", unit="schemas") as pbar:
        for schema_id, paths in schema_to_paths.items():
            num_files_in_schema = len(paths)

            if schema_id not in schema_cache:
                for _ in paths:
                    reporter.stats["classified_junk"] += 1
                    reporter.junk_reasons[f"Schema {schema_id} not in cache"] += 1
                pbar.update(1)
                continue

            key_map, schema_info = schema_cache[schema_id]

            if not key_map:
                for _ in paths:
                    reporter.stats["classified_junk"] += 1
                    reporter.junk_reasons["Schema has no keys"] += 1
                pbar.update(1)
                continue

            assessor = CheckpointAssessor(key_map)
            classification, reason = assessor.classify()

            if classification == "CHECKPOINT":
                for _ in paths:
                    reporter.stats["classified_checkpoint"] += 1
                pbar.update(1)
                continue
            
            if classification == "JUNK":
                for path_str in paths:
                    reporter.stats["classified_junk"] += 1
                    reporter.junk_reasons[reason] += 1
                    if args.verbose:
                        reporter.add_ambiguous_rejection(path_str, reason, schema_info)
                pbar.update(1)
                continue

            # --- Adapter Schema Processing ---
            for _ in paths:
                reporter.stats["classified_adapter"] += 1

            if isinstance(key_map, dict):
                source_keys = set(key_map.keys())
            else:
                source_keys = set(key_map)

            best_match, max_score, best_details = None, -1.0, {}
            for base_path, km in key_mappers.items():
                score, details = calculate_compatibility_detailed(
                    km, source_keys, key_mapper_caches[base_path]
                )
                if score > max_score:
                    max_score, best_match, best_details = score, base_path, details

            if best_match:
                for path_str in paths:
                    reporter.add_compatibility_score(
                        path_str, best_match.name, max_score, best_details
                    )

            if max_score < COMPATIBILITY_THRESHOLD:
                for path_str in paths:
                    reporter.stats["no_compatible_base"] += 1
                    reason_str = f"Max compatibility score ({max_score:.2f}) below threshold"
                    reporter.no_match_reasons[reason_str] += 1
                    if max_score > 0:
                        reporter.add_low_score_rejection(path_str, max_score, best_match.name if best_match else "N/A")
                pbar.update(1)
                continue

            matched_km = key_mappers[best_match]
            base_name = best_match.name
            
            # --- Perform mapping test once for the entire schema ---
            base_context = matched_km.context
            category_counts = defaultdict(lambda: defaultdict(int))
            failures_in_schema = []

            # Use a helper to classify keys and check if they should be tested
            def should_test_key(key_base: str) -> Optional[str]:
                """
                Returns the category if the key should be tested, otherwise None.
                This now uses the base_context to make intelligent decisions.
                """
                # Rule 1: Granular text encoder categorization
                if "lora_te" in key_base:
                    # Check for specific text encoder types
                    if "lora_te1_" in key_base and "CLIP-L" in base_context.components_present:
                        return "CLIP-L"
                    elif "lora_te2_" in key_base and "CLIP-G" in base_context.components_present:
                        return "CLIP-G"
                    elif "lora_t5_" in key_base and "T5-XXL" in base_context.components_present:
                        return "T5-XXL"
                    # Fallback for generic text encoder keys
                    elif any(comp in base_context.components_present for comp in ["CLIP-L", "CLIP-G", "T5-XXL"]):
                        return "Text Encoder (Generic)"
                    return None

                # Rule 2: Handle UNet/DiT keys with granular categorization
                elif "lora_unet" in key_base:
                    # Check for MMDiT/SD3.5 specific keys
                    if "joint_blocks" in key_base and "MMDiT" in base_context.components_present:
                        return "MMDiT"
                    
                    # Check for Chroma-specific guidance keys first (they don't contain double_blocks/single_blocks)
                    if "distilled_guidance_layer" in key_base:
                        return "Chroma Guidance" if base_context.model_type == "Chroma" else None
                    
                    # Check for FLUX/Chroma specific keys
                    elif "double_blocks" in key_base or "single_blocks" in key_base:
                        # FLUX-specific modulation keys - only compatible with FLUX models
                        if "_mod_lin" in key_base or "_modulation_lin" in key_base:
                            return "FLUX Modulation" if base_context.model_type == "FLUX-dev" else None
                        
                        # Generic FLUX/Chroma keys (without modulation/guidance)
                        elif base_context.model_type in ["FLUX-dev", "Chroma"]:
                            return "FLUX/Chroma UNet"
                    
                    # Generic UNet/DiT keys
                    elif any(comp in base_context.components_present for comp in ["UNet", "DiT", "MMDiT"]):
                        return "UNet/DiT"
                    
                    return None
                
                # Rule 3: Other keys can be categorized but always tested
                else:
                    return "Other"

            for raw_key in source_keys:
                key_base = "UNSTRIPPABLE"
                stripped = matched_km._strip_suffix(raw_key)
                if stripped:
                    key_base = stripped[0]
                
                category = should_test_key(key_base)
                # If the key's component is not in the base model, skip it
                if category is None:
                    continue

                category_counts[category]['total'] += 1
                result = matched_km.map_from_lora(raw_key)
                if result:
                    category_counts[category]['mapped'] += 1
                else:
                    failures_in_schema.append(key_base)

            # Apply stats for the entire schema once, which is much more efficient.
            current_base_results = reporter.per_base_results[base_name]
            current_base_results['files_tested'] = current_base_results.get('files_tested', 0) + num_files_in_schema

            for category, counts in category_counts.items():
                cat_data = current_base_results.setdefault(category, {'mapped': 0, 'total': 0})
                cat_data['mapped'] += counts['mapped'] * num_files_in_schema
                cat_data['total'] += counts['total'] * num_files_in_schema
            
            # Log the individual failures for each file that shares this schema
            if failures_in_schema:
                for path_str in paths:
                    for failure in failures_in_schema:
                        reporter.add_failure(failure, path_str)
            
            pbar.update(1)

    # --- Step 6. Print the final report ---
    reporter.print_summary(key_mappers)


if __name__ == "__main__":
    main()
