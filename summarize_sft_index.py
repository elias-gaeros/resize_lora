import argparse
import sys
from collections import defaultdict
from pathlib import Path
import re
import hashlib

from compressed_sft_index import CompressedIndex


# --- Formatting Helpers ---
def format_size(size_bytes):
    if not isinstance(size_bytes, (int, float)):
        return "N/A"
    if size_bytes == 0:
        return "0 B"
    if size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    return f"{size_bytes/1024**3:.2f} GB"


def format_dtype(dtype_str):
    if not isinstance(dtype_str, str):
        return "N/A"
    dtype_str = dtype_str.replace("torch.", "")
    return "".join(re.findall(r"[a-zA-Z]", dtype_str)).upper()[0] + "".join(
        re.findall(r"\d+", dtype_str)
    )


def summarize_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    indices = sorted(list(set(indices)))
    if not indices:
        return ""
    ranges, start = [], indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            end = indices[i - 1]
            ranges.append(f"{start}" if start == end else f"{start}-{end}")
            start = indices[i]
    end = indices[-1]
    ranges.append(f"{start}" if start == end else f"{start}-{end}")
    return f"[{', '.join(ranges)}]"


# --- Adapter Detection System ---


def detect_attention_pattern(node: dict) -> tuple[str, dict] | None:
    """
    Detects if a node represents an attention pattern (QKV projections + output).

    Args:
        node: The node dictionary to analyze

    Returns:
        Tuple of (pattern_type, info) if attention pattern detected, None otherwise
    """
    if not isinstance(node, dict):
        return None

    # Common attention projection names across different model architectures
    qkv_patterns = {
        "q": ["q", "q_proj", "query", "query_proj"],
        "k": ["k", "k_proj", "key", "key_proj"],
        "v": ["v", "v_proj", "value", "value_proj"],
    }

    output_patterns = ["out", "out_proj", "output", "output_proj", "o_proj"]

    # Check what attention components are present
    found_components = {"q": False, "k": False, "v": False, "out": False}
    adapter_types = set()
    adapter_info = {}

    for key, child_node in node.items():
        if key.startswith("_"):
            continue

        # Check if this is a Q, K, V, or output projection
        component = None
        for comp, patterns in qkv_patterns.items():
            if key in patterns:
                component = comp
                break

        if not component and key in output_patterns:
            component = "out"

        if component:
            found_components[component] = True

            # Analyze what type of adapter this component uses (avoid recursion)
            child_type, child_payload = analyze_node_type_basic(child_node)
            if child_type in {
                "lora",
                "dora",
                "loha",
                "locon",
                "lokr",
                "ia3",
                "diff",
                "set_weight",
                "norm",
                "adapter",
            }:
                adapter_types.add(child_type)
                if component not in adapter_info:
                    adapter_info[component] = child_payload

    # Determine what kind of attention pattern we have
    has_qkv = found_components["q"] and found_components["k"] and found_components["v"]
    has_output = found_components["out"]

    if has_qkv and has_output:
        pattern_type = "attention_full"
        description = "Attention (QKV + Output)"
    elif has_qkv:
        pattern_type = "attention_qkv"
        description = "Attention (QKV only)"
    elif has_output and (
        found_components["q"] or found_components["k"] or found_components["v"]
    ):
        pattern_type = "attention_partial"
        partial_components = [
            comp for comp in ["q", "k", "v"] if found_components[comp]
        ]
        description = f"Attention ({'+'.join(partial_components).upper()} + Output)"
    else:
        return None  # Not an attention pattern

    # Summarize adapter information
    if adapter_types:
        if len(adapter_types) == 1:
            adapter_type = list(adapter_types)[0]
            # Get adapter symbol and name
            adapter_symbols = {
                "lora": "◎",
                "dora": "◉",
                "loha": "◈",
                "locon": "◎",
                "lokr": "◇",
                "ia3": "△",
                "diff": "●",
                "set_weight": "■",
                "norm": "▲",
                "adapter": "◯",
            }
            symbol = adapter_symbols.get(adapter_type, "◯")

            # Get variant info from first component
            sample_info = next(iter(adapter_info.values()), {})
            variant = sample_info.get("variant", "")
            if variant and variant != f"Standard {adapter_type.upper()}":
                adapter_desc = variant
            else:
                adapter_desc = adapter_type.upper()

            # Add rank information if available
            rank = sample_info.get("rank", "?")
            dtype = sample_info.get("dtype", "?")
            if adapter_type in ["lora", "dora", "loha", "locon"] and rank != "?":
                adapter_desc += f" (Rank={rank}, {dtype})"
            elif dtype != "?":
                adapter_desc += f" ({dtype})"

            full_description = f"{description}: {adapter_desc}"
        else:
            # Mixed adapter types
            adapter_desc = "+".join(sorted(adapter_types))
            full_description = f"{description}: Mixed ({adapter_desc})"
            symbol = "◯"
    else:
        # No adapters, just regular layers
        full_description = f"{description}: Standard layers"
        symbol = "●"

    return pattern_type, {
        "description": full_description,
        "symbol": symbol,
        "components": found_components,
        "adapter_types": list(adapter_types),
        "adapter_info": adapter_info,
    }


def detect_adapter_type(node_keys: set[str]) -> tuple[str, dict]:
    """
    Comprehensive adapter type detection based on tensor suffixes.

    Args:
        node_keys: Set of tensor names in this node

    Returns:
        Tuple of (adapter_type, payload_info)
    """

    # Helper function to check if any key has the given suffix
    def has_suffix(suffix: str) -> bool:
        return any(key.endswith(suffix) for key in node_keys)

    def get_tensor_with_suffix(suffix: str):
        for key in node_keys:
            if key.endswith(suffix):
                return key
        return None

    # Detection order by specificity (most specific first)

    # 1. DoRA - Must have both LoRA components AND dora_scale
    if (
        has_suffix(".lora_down.weight")
        and has_suffix(".lora_up.weight")
        and has_suffix(".dora_scale")
    ):
        return "dora", {"variant": "Weight-Decomposed LoRA"}

    # 2. LoHa (LyCORIS Hadamard Product) - 4 component matrices
    if (
        has_suffix(".hada_w1_a.weight")
        and has_suffix(".hada_w1_b.weight")
        and has_suffix(".hada_w2_a.weight")
        and has_suffix(".hada_w2_b.weight")
    ):
        return "loha", {"variant": "Hadamard Product LoRA"}

    # 3. LoKr (LyCORIS Kronecker Product) - 2 main matrices, optional t1/t2
    if (
        has_suffix(".lokr_w1")
        or has_suffix(".lokr_w2")
        or has_suffix(".lokr_w1.weight")
        or has_suffix(".lokr_w2.weight")
    ):
        variant = "Kronecker Product LoRA"
        if has_suffix(".t1.weight") or has_suffix(".t2.weight"):
            variant += " (with T matrices)"
        return "lokr", {"variant": variant}

    # 4. IA³ (Infused Adapter) - Simple scaling vector
    if has_suffix(".ia3_input_mask.weight"):
        return "ia3", {"variant": "Infused Adapter (Scaling)"}

    # 5. LoCon (LoRA for Convolution) - Has lora_mid component
    if (
        has_suffix(".lora_down.weight")
        and has_suffix(".lora_up.weight")
        and has_suffix(".lora_mid.weight")
    ):
        return "locon", {"variant": "LoRA for Convolution"}

    # 6. Standard LoRA - Basic up/down structure
    if has_suffix(".lora_down.weight") and has_suffix(".lora_up.weight"):
        return "lora", {"variant": "Standard LoRA"}

    # 7. Alternative LoRA naming (diffusers style)
    if has_suffix(".lora_A.weight") and has_suffix(".lora_B.weight"):
        return "lora", {"variant": "Standard LoRA (alt naming)"}

    # 8. Full/Diff adapters - Direct weight differences
    if has_suffix(".diff.weight"):
        return "diff", {"variant": "Weight Difference"}

    # 9. Set weight adapters - Direct weight replacement
    if has_suffix(".set_weight"):
        return "set_weight", {"variant": "Weight Replacement"}

    # 10. Weight norm adapters
    if has_suffix(".w_norm") or has_suffix(".b_norm"):
        return "norm", {"variant": "Weight Normalization"}

    # 11. Generic adapter detection - has alpha but no specific pattern
    if has_suffix(".alpha"):
        return "adapter", {"variant": "Generic Adapter"}

    return None, {}


def extract_adapter_info(node: dict, adapter_type: str) -> dict:
    """
    Extract detailed information about the adapter (rank, dtype, etc.)

    Args:
        node: The node dictionary containing tensor specs
        adapter_type: The detected adapter type

    Returns:
        Dictionary with adapter information
    """
    info = {"rank": "?", "dtype": "?", "tensors": {}}

    try:
        # Collect all tensor specs in this node
        for key, value in node.items():
            if isinstance(value, dict) and "_info" in value:
                info["tensors"][key] = value["_info"]
            elif (
                isinstance(value, dict)
                and "weight" in value
                and "_info" in value["weight"]
            ):
                info["tensors"][f"{key}.weight"] = value["weight"]["_info"]

        # Extract rank and dtype based on adapter type
        if adapter_type in ["lora", "dora", "locon"]:
            # Look for down/A matrix to get rank
            down_keys = ["lora_down", "lora_A"]
            for down_key in down_keys:
                if down_key in node and "weight" in node[down_key]:
                    down_info = node[down_key]["weight"]["_info"]
                    info["rank"] = down_info["shape"][0]
                    info["dtype"] = format_dtype(down_info["dtype"])
                    break

        elif adapter_type == "loha":
            # LoHa rank is more complex, use w1_a
            if "hada_w1_a" in node and "weight" in node["hada_w1_a"]:
                w1a_info = node["hada_w1_a"]["weight"]["_info"]
                info["rank"] = w1a_info["shape"][0]
                info["dtype"] = format_dtype(w1a_info["dtype"])

        elif adapter_type == "lokr":
            # LoKr rank from w1 or w2
            for w_key in ["lokr_w1", "lokr_w2"]:
                if w_key in node and "_info" in node[w_key]:
                    w_info = node[w_key]["_info"]
                    info["dtype"] = format_dtype(w_info["dtype"])
                    # LoKr rank is more complex, just show dtype
                    break

        elif adapter_type in ["ia3", "diff", "set_weight", "norm"]:
            # For these types, just get dtype from any available tensor
            for tensor_spec in info["tensors"].values():
                info["dtype"] = format_dtype(tensor_spec["dtype"])
                break

    except (KeyError, IndexError, TypeError):
        pass  # Keep default values

    return info


# --- Core Analysis and Tree Building Logic ---


def analyze_node_type_basic(node):
    """
    Basic node classification without attention pattern detection.
    Used to avoid recursion when called from detect_attention_pattern.
    """
    if not isinstance(node, dict):
        return "module", {}

    node_keys = set(node.keys())

    # --- Check for grouped adapter modules first ---
    if "_adapter_group" in node and "_adapter_tensors" in node:
        # This is a grouped adapter module created by the tree building logic
        adapter_tensors = node["_adapter_tensors"]
        adapter_type, adapter_meta = detect_adapter_type(set(adapter_tensors))

        if adapter_type:
            # Extract detailed adapter information
            payload = extract_adapter_info(node, adapter_type)
            payload.update(adapter_meta)
            return adapter_type, payload

    # --- Adapter Detection ---

    # First, try comprehensive adapter detection
    adapter_type, adapter_meta = detect_adapter_type(node_keys)

    if adapter_type:
        # Extract detailed adapter information
        payload = extract_adapter_info(node, adapter_type)
        payload.update(adapter_meta)
        return adapter_type, payload

    # --- Pre-computation: Analyze children to determine node's nature ---
    is_container = False
    # Check for sub-modules that are NOT terminal blocks
    for k, v in node.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and analyze_node_type_basic(v)[0] == "module":
            is_container = True
            break

    # --- Fallback to Standard Detection ---

    # Standard Layer check (must have a 'weight' tensor and not be a container)
    if "weight" in node_keys and not is_container:
        return "layer", {}

    # If it is a container of other modules, it's a 'module'
    if is_container:
        return "module", {}

    # Fallback: If it's not a container and not a recognized layer/adapter, but it has
    # any leaf tensors, classify it as a terminal block (e.g., an Embedding)
    for v in node.values():
        if isinstance(v, dict) and "_info" in v:
            return "embedding", {}

    # If none of the above, it's an empty module or an unhandled structure
    return "module", {}


def analyze_node_type(node):
    """
    node classification with comprehensive adapter format detection.

    This version detects all major adapter formats from ComfyUI documentation
    and also recognizes attention patterns.
    """
    if not isinstance(node, dict):
        return "module", {}

    node_keys = set(node.keys())

    # --- Check for grouped adapter modules first ---
    if "_adapter_group" in node and "_adapter_tensors" in node:
        # This is a grouped adapter module created by the tree building logic
        adapter_tensors = node["_adapter_tensors"]
        adapter_type, adapter_meta = detect_adapter_type(set(adapter_tensors))

        if adapter_type:
            # Extract detailed adapter information
            payload = extract_adapter_info(node, adapter_type)
            payload.update(adapter_meta)
            return adapter_type, payload

    # --- Check for attention patterns ---
    attention_result = detect_attention_pattern(node)
    if attention_result:
        pattern_type, pattern_info = attention_result
        return pattern_type, pattern_info

    # --- Pre-computation: Analyze children to determine node's nature ---
    is_container = False
    # Check for sub-modules that are NOT terminal blocks
    for k, v in node.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and analyze_node_type(v)[0] == "module":
            is_container = True
            break

    # --- Adapter Detection ---

    # First, try comprehensive adapter detection
    adapter_type, adapter_meta = detect_adapter_type(node_keys)

    if adapter_type:
        # Extract detailed adapter information
        payload = extract_adapter_info(node, adapter_type)
        payload.update(adapter_meta)
        return adapter_type, payload

    # --- Fallback to Standard Detection ---

    # Standard Layer check (must have a 'weight' tensor and not be a container)
    if "weight" in node_keys and not is_container:
        return "layer", {}

    # If it is a container of other modules, it's a 'module'
    if is_container:
        return "module", {}

    # Fallback: If it's not a container and not a recognized layer/adapter, but it has
    # any leaf tensors, classify it as a terminal block (e.g., an Embedding)
    for v in node.values():
        if isinstance(v, dict) and "_info" in v:
            return "embedding", {}

    # If none of the above, it's an empty module or an unhandled structure
    return "module", {}


def get_architectural_hash(node):
    """
    Generates a hash based on a module's architecture (names of its children).
    This allows grouping of blocks with internal variations (like different LoRA ranks).
    """
    if not isinstance(node, dict) or analyze_node_type(node)[0] != "module":
        return analyze_node_type(node)[0]
    child_names = sorted([k for k in node.keys() if not k.startswith("_")])
    s_hash = hashlib.sha256(",".join(child_names).encode()).hexdigest()
    return f"mod-{s_hash[:12]}"


def summarize_template_group(group_nodes: list[dict], prefix: str, is_last: bool):
    """
    Analyzes a group of architecturally-identical blocks to summarize their
    internal variations (e.g., which sub-modules are LoRAs, what are their ranks).
    This function is largely unchanged but benefits from the richer input.
    """
    child_prefix = prefix + ("    " if is_last else "│   ")

    if not group_nodes:
        return
    # All nodes in a template group have the same architecture, so use the first as a reference.
    child_names = sorted([k for k in group_nodes[0].keys() if not k.startswith("_")])

    for i, name in enumerate(child_names):
        child_is_last = i == len(child_names) - 1
        connector = "└── " if child_is_last else "├── "

        # Collect all instances of this child from across the group of parent nodes
        child_instances = [node.get(name) for node in group_nodes if node.get(name)]
        if not child_instances:
            continue

        # Analyze the collected instances to see what types they are
        instance_types = [analyze_node_type(inst) for inst in child_instances]
        unique_types = {t[0] for t in instance_types}

        # Case A: Consistently an adapter layer (LoRA, DoRA, LoHa, etc.) or attention pattern
        adapter_types = {
            "lora",
            "dora",
            "loha",
            "locon",
            "lokr",
            "ia3",
            "diff",
            "set_weight",
            "norm",
            "adapter",
        }
        attention_types = {"attention_full", "attention_qkv", "attention_partial"}

        if len(unique_types) == 1 and unique_types.intersection(adapter_types):
            adapter_type = list(unique_types)[0]
            payloads = [t[1] for t in instance_types]

            # Get adapter symbol and name
            adapter_symbols = {
                "lora": "◎",
                "dora": "◉",
                "loha": "◈",
                "locon": "◎",
                "lokr": "◇",
                "ia3": "△",
                "diff": "●",
                "set_weight": "■",
                "norm": "▲",
                "adapter": "◯",
            }
            symbol = adapter_symbols.get(adapter_type, "◯")

            # Format name with variant if available
            display_name = adapter_type.upper()
            variant = payloads[0].get("variant")
            if variant and variant != f"Standard {adapter_type.upper()}":
                display_name = variant

            # Handle rank information
            ranks = [p["rank"] for p in payloads if p.get("rank") != "?"]
            if len(ranks) == 0:
                rank_info = ""
            else:
                rank_str = (
                    f"Rank={ranks[0]}"
                    if len(set(ranks)) <= 1
                    else f"Rank=[{min(ranks)}-{max(ranks)}]"
                )
                dtype = payloads[0].get("dtype", "?")
                rank_info = (
                    f" ({rank_str}, {dtype})"
                    if adapter_type in ["lora", "dora", "loha", "locon"]
                    else f" ({dtype})"
                )

            print(
                f"{child_prefix}{connector}{symbol} {display_name}: {name}{rank_info}"
            )

        # Case A2: Consistently an attention pattern
        elif len(unique_types) == 1 and unique_types.intersection(attention_types):
            attention_type = list(unique_types)[0]
            payloads = [t[1] for t in instance_types]

            # Use the description and symbol from the attention pattern detection
            description = payloads[0].get("description", f"Attention: {name}")
            symbol = payloads[0].get("symbol", "●")

            print(f"{child_prefix}{connector}{symbol} {description}")

        # Case B: Consistently a standard layer
        elif len(unique_types) == 1 and "layer" in unique_types:
            print(f"{child_prefix}{connector}● Layer: {name}")

        # Case C: Consistently a container module (recurse)
        elif len(unique_types) == 1 and "module" in unique_types:
            print(f"{child_prefix}{connector}▶ {name}")
            summarize_template_group(child_instances, child_prefix, child_is_last)

        # Case D: Mixed types (e.g., some instances have LoRA, some don't)
        else:
            type_counts = defaultdict(int)
            for t in instance_types:
                type_counts[t[0]] += 1

            # Format type names more clearly
            formatted_counts = []
            for type_name, count in type_counts.items():
                if type_name in attention_types:
                    formatted_counts.append(f"attention ({count})")
                elif type_name in adapter_types:
                    formatted_counts.append(f"{type_name} ({count})")
                else:
                    formatted_counts.append(f"{type_name} ({count})")

            summary = ", ".join(formatted_counts)
            print(
                f"{child_prefix}{connector}○ Mixed: {name} ({summary} / {len(group_nodes)})"
            )


def get_network_statistics(node: dict) -> dict[str, int]:
    """
    Recursively traverses the model structure to count node types.
    This version ensures a node is counted only once, prioritizing its most
    specific classification.
    """
    stats = defaultdict(int)

    def _traverse(n: dict):
        if not isinstance(n, dict):
            return

        # 1. Classify the current node first.
        node_type, _ = analyze_node_type(n)
        stats[node_type] += 1

        # 2. IMPORTANT: If the node was classified as a terminal block (layer, adapter, etc.),
        #    we STOP recursion. We don't want to traverse its internal 'weight' and 'bias'
        #    and classify them as separate things.
        adapter_types = {
            "lora",
            "dora",
            "loha",
            "locon",
            "lokr",
            "ia3",
            "diff",
            "set_weight",
            "norm",
            "adapter",
        }
        attention_types = {"attention_full", "attention_qkv", "attention_partial"}
        if node_type != "module":
            return

        # 3. If it was classified as a 'module', recurse into its children.
        for child_key, child_node in n.items():
            if not child_key.startswith("_"):
                _traverse(child_node)

    _traverse(node)

    # The root node is always counted as a module by the initial call.
    # We subtract it here because it represents the file itself, not a sub-module.
    if "module" in stats:
        stats["module"] -= 1
        if stats["module"] == 0:
            del stats["module"]  # Clean up if it was the only one.

    return dict(stats)


def get_file_summary_info(
    loader: CompressedIndex,
    path: str,
    samples: int = 0,
    print_metadata: bool | int = False,
):
    """Prints detailed metadata and a sample of keys for a given file path."""
    instance = loader.instances.get(path)
    if not instance:
        return

    print("\n    --- File Details ---")

    # Print Metadata
    metadata = {**instance["user_metadata"], **instance["file_metadata"]}
    if metadata and print_metadata:
        print("    Metadata:")
        items = metadata.items()
        if isinstance(print_metadata, int):
            items = list(items)[:print_metadata]
        for k, v in items:
            # Truncate long values for readability
            v_str = str(v)
            if len(v_str) > 120:
                v_str = v_str[:117] + "..."
            print(f"      - {k}: {v_str}")

    # Print Key Sample
    key_map = loader.get_key_map(instance["schema_id"])
    print(f"    Tensor Key Sample ({len(key_map)} total):")
    for key in key_map[:samples]:
        print(f"      - {key}")
    if len(key_map) > samples:
        print("      - ...")
    print("    --------------------")


def generate_summary_tree(node, name, prefix="", is_last=True):
    """
    Recursively processes the model's structure tree to generate a printable summary.
    This version adds a stronger base case to stop recursion into _info blocks.
    """
    # --- Stronger Base Case ---
    # If a node has an '_info' key, it is a leaf representing a single tensor's
    # specification. We should not traverse inside it. We simply stop here.
    if "_info" in node:
        return

    node_type, payload = analyze_node_type(node)

    # --- Display Logic ---

    # Base case for computational blocks (LoRA, Layer, etc.)
    # We print their summary and STOP recursion.
    if node_type != "module":
        # Create the summary string based on the type
        summary = ""

        # adapter display
        adapter_types = {
            "lora",
            "dora",
            "loha",
            "locon",
            "lokr",
            "ia3",
            "diff",
            "set_weight",
            "norm",
            "adapter",
        }
        attention_types = {"attention_full", "attention_qkv", "attention_partial"}

        if node_type in attention_types:
            # Use the pre-formatted description and symbol for attention patterns
            description = payload.get("description", f"Attention: {name}")
            symbol = payload.get("symbol", "●")
            print(f"{prefix}{'└── ' if is_last else '├── '}{symbol} {description}")

        elif node_type in adapter_types:
            # Get display name with variant
            display_name = node_type.upper()
            variant = payload.get("variant")
            if variant and variant != f"Standard {node_type.upper()}":
                display_name = variant

            # Format summary based on adapter type
            if node_type in ["lora", "dora", "loha", "locon"]:
                rank = payload.get("rank", "?")
                dtype = payload.get("dtype", "?")
                summary = f"Rank={rank}, {dtype}"
            else:
                dtype = payload.get("dtype", "?")
                summary = f"{dtype}" if dtype != "?" else ""

            # Get the appropriate symbol
            symbols = {
                "lora": "◎",
                "dora": "◉",
                "loha": "◈",
                "locon": "◎",
                "lokr": "◇",
                "ia3": "△",
                "diff": "●",
                "set_weight": "■",
                "norm": "▲",
                "adapter": "◯",
            }
            symbol = symbols.get(node_type, "◯")

            # Format final output
            summary_part = f" ({summary})" if summary else ""
            print(
                f"{prefix}{'└── ' if is_last else '├── '}{symbol} {display_name}: {name}{summary_part}"
            )

        else:
            # Standard layer types
            display_name = node_type.capitalize()
            symbols = {"layer": "●", "embedding": "¶"}
            symbol = symbols.get(node_type, "○")

            # Format final output
            summary_part = f" ({summary})" if summary else ""
            print(
                f"{prefix}{'└── ' if is_last else '├── '}{symbol} {display_name}: {name}{summary_part}"
            )

        # For grouped adapter modules, show the individual tensors
        if "_adapter_group" in node and "_adapter_tensors" in node:
            child_prefix = prefix + ("    " if is_last else "│   ")
            adapter_tensors = node["_adapter_tensors"]

            # Show the first few tensors
            for i, tensor_key in enumerate(adapter_tensors[:3]):
                tensor_suffix = tensor_key[len(name) + 1 :]  # Remove module name + '.'
                tensor_is_last = i == min(2, len(adapter_tensors) - 1)
                print(
                    f"{child_prefix}{'└── ' if tensor_is_last else '├── '}● {tensor_suffix}"
                )

            if len(adapter_tensors) > 3:
                print(
                    f"{child_prefix}    └── ... and {len(adapter_tensors) - 3} more tensors"
                )

        return

    # --- Container Module Logic ---

    # Path Contraction for cleaner output
    current_node, current_name = node, name
    while True:
        # Check if the current node is a simple wrapper around another module
        child_keys = [k for k in current_node.keys() if not k.startswith("_")]
        if len(child_keys) == 1:
            child_key = child_keys[0]
            child_node = current_node[child_key]
            # Ensure the child is also a module to continue contraction
            if (
                isinstance(child_node, dict)
                and analyze_node_type(child_node)[0] == "module"
            ):
                current_name = f"{current_name}.{child_key}"
                current_node = child_node
                continue
        break  # Stop if it's not a simple wrapper

    print(f"{prefix}{'└── ' if is_last else '├── '}▶ {current_name}")
    child_prefix = prefix + ("    " if is_last else "│   ")

    # Separate children into sequential (numeric keys) and named modules
    child_modules = {k: v for k, v in current_node.items() if not k.startswith("_")}
    # ... (rest of the function for grouping and rendering children is unchanged) ...
    sequential_groups = defaultdict(lambda: {"nodes": [], "keys": []})
    named_modules = {}

    for key, child in child_modules.items():
        if key.isdigit():
            h = get_architectural_hash(child)
            sequential_groups[h]["nodes"].append(child)
            sequential_groups[h]["keys"].append(key)
        else:
            named_modules[key] = child

    render_items = list(sequential_groups.values()) + list(named_modules.items())

    def sort_key(item):
        key = item[0] if isinstance(item, tuple) else item["keys"][0]
        return (int(key) if key.isdigit() else float("inf"), key)

    render_items.sort(key=sort_key)

    for i, item in enumerate(render_items):
        item_is_last = i == len(render_items) - 1

        if isinstance(item, dict):  # A template group of sequential blocks
            nodes, keys = item["nodes"], item["keys"]
            if len(nodes) > 1:
                indices_str = summarize_indices(sorted([int(k) for k in keys]))
                print(
                    f"{child_prefix}{'└── ' if item_is_last else '├── '}▶ Block Template (x{len(nodes)}) @ {indices_str}"
                )
                summarize_template_group(nodes, child_prefix, item_is_last)
            else:  # A group of one is just a regular module
                generate_summary_tree(nodes[0], keys[0], child_prefix, item_is_last)
        else:  # A named module
            key, child_node = item
            generate_summary_tree(child_node, key, child_prefix, item_is_last)


def main():
    parser = argparse.ArgumentParser(
        description="Create a condensed, hierarchical summary of model structures from a compressed index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "compressed_index_file",
        type=str,
        help="Path to the compressed sft_index.json file.",
    )
    parser.add_argument(
        "--schema",
        type=int,
        default=None,
        help="Only summarize files belonging to a specific schema ID.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print network statistics instead of the detailed tree.",
    )
    parser.add_argument(
        "--min-transitive-instances",
        type=int,
        default=0,
        help="Only display schemas with more than N total transitive instances.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        nargs="?",
        const=10,
        default=0,
        help="Show key samples for the representative file of each schema. If used without value, defaults to 10.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="?",
        const="10",
        default="0",
        help="Number of metadata items to print for the representative file. If used without value, defaults to 10. Use 'all' to show all metadata.",
    )
    args = parser.parse_args()
    if args.metadata == "all":
        args.metadata = True
    else:
        args.metadata = int(args.metadata)

    # ... (loader setup and schema filtering logic is unchanged) ...
    index_path = Path(args.compressed_index_file)
    if not index_path.exists():
        print(
            f"Error: Compressed index file not found at '{index_path}'", file=sys.stderr
        )
        sys.exit(1)

    loader = CompressedIndex(index_path)

    base_to_children = defaultdict(list)
    for i, schema_obj in enumerate(loader._schemas):
        if "base" in schema_obj:
            base_to_children[schema_obj["base"]].append(i)

    def get_all_descendant_ids(schema_id):
        all_ids, queue = {schema_id}, [schema_id]
        while queue:
            parent_id = queue.pop(0)
            for child_id in base_to_children.get(parent_id, []):
                if child_id not in all_ids:
                    all_ids.add(child_id)
                    queue.append(child_id)
        return list(all_ids)

    schema_ids_to_process = []
    if args.schema is not None:
        if 0 <= args.schema < len(loader._schemas):
            schema_ids_to_process.append(args.schema)
        else:
            print(f"Error: Schema ID {args.schema} is out of range.", file=sys.stderr)
            sys.exit(1)
    else:
        schema_ids_to_process = sorted(loader._schema_to_paths.keys())

    # Main loop
    for i, schema_id in enumerate(schema_ids_to_process):
        schema_info = loader.get_schema_info(schema_id)
        direct_paths = loader.get_paths_for_schema(schema_id)
        descendant_ids = get_all_descendant_ids(schema_id)
        transitive_paths = set(direct_paths)
        for did in descendant_ids:
            transitive_paths.update(loader.get_paths_for_schema(did))

        if len(transitive_paths) < args.min_transitive_instances:
            continue

        if i > 0 and any(s in sys.argv for s in ["--schema", "--sample", "--stats"]):
            # Add a separator only if we are printing more than just the default list
            print("\n" + "=" * 80 + "\n")

        print(f"--- Summary for Schema ID: {schema_id} ({schema_info['type']}) ---")
        if schema_info["type"] == "view":
            print(f"    This is a subset of Schema ID: {schema_info['base_id']}")
        print(f"    Tensor Count: {schema_info['key_count']}")
        print(
            f"    File Instances: {len(direct_paths)} (direct), {len(transitive_paths)} (transitive)"
        )

        paths = sorted(list(transitive_paths))
        if not paths:
            print("\n    No files associated with this schema.")
            continue

        representative_path = paths[0]
        raw_tree = loader.get_raw_tree_for_path(representative_path)
        if not raw_tree:
            print(f"    Could not build tree for {representative_path}")
            continue

        # *** INTEGRATE NEW DEBUGGING AND STATS LOGIC ***
        if args.sample or args.metadata:
            get_file_summary_info(
                loader,
                representative_path,
                samples=args.sample,
                print_metadata=args.metadata,
            )

        if args.stats:
            stats = get_network_statistics(raw_tree)
            # Add tensor count for comparison
            stats["tensor_count_in_schema"] = schema_info["key_count"]
            print(
                f"\n    Network Statistics from representative: {Path(representative_path).name}"
            )
            for component, count in sorted(stats.items()):
                print(f"      - {component.replace('_', ' ').capitalize()}s: {count}")
        else:
            print(
                f"\n    Displaying structure from representative file: {Path(representative_path).name}"
            )
            for p in paths[:3]:
                print(f"      - {Path(p).name}")
            if len(paths) > 3:
                print(f"      - ... and {len(paths)-3} more")
            generate_summary_tree(raw_tree, Path(representative_path).name)


if __name__ == "__main__":
    main()
