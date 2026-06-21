import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import sys
import struct

MAX_HEADER_SIZE = 256 * 1024 * 1024
STRING_POOL_REF = "__sft_string_pool_ref__"


def parse_safetensors_header(file_path: Path) -> dict | None:
    """
    Reads and parses the header of a .safetensors file without using the
    safetensors library, for performance reasons.

    Args:
        file_path (Path): The path to the .safetensors file.

    Returns:
        A dictionary representing the parsed JSON header, or None if an error occurs.
    """
    try:
        with open(file_path, "rb") as f:
            # Read the 8-byte header length
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                # File is too small to be a valid safetensors file
                return None

            # Unpack the little-endian unsigned 64-bit integer
            header_len = struct.unpack("<Q", header_len_bytes)[0]

            file_size = file_path.stat().st_size
            if header_len > MAX_HEADER_SIZE or header_len > file_size - 8:
                return None

            # Read the JSON header
            json_header_bytes = f.read(header_len)
            if len(json_header_bytes) < header_len:
                # The file is truncated or the header length is incorrect
                return None

            # Decode the JSON header from UTF-8
            json_header_str = json_header_bytes.decode("utf-8")

            # Parse the JSON string into a Python dictionary
            header = json.loads(json_header_str)
            return header if isinstance(header, dict) else None

    except (struct.error, json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
        # These errors indicate a malformed file or a read error
        print(f"Debug: Failed to parse header for {file_path}: {e}", file=sys.stderr)
        return None


def index_safetensors_file(file_path: Path) -> dict | None:
    """
    Extracts metadata and tensor information from a single .safetensors file
    using a fast, lightweight, direct header parsing method.

    Args:
        file_path (Path): The path to the .safetensors file.

    Returns:
        A dictionary containing the file's information, or None if an error occurs.
    """
    try:
        # Get file system metadata first
        file_stat = file_path.stat()
        file_system_info = {
            "__file_size__": file_stat.st_size,
            "__file_mtime__": file_stat.st_mtime,
            "__file_ctime__": file_stat.st_ctime,
            "__file_inode__": file_stat.st_ino,
        }

        # Use the new lightweight parser
        header = parse_safetensors_header(file_path)
        if header is None:
            print(
                f"Warning: Could not parse header for file {file_path}", file=sys.stderr
            )
            return None

        # The header is a dictionary containing tensor info and potentially '__metadata__'
        # Extract the user metadata if it exists
        metadata = header.pop("__metadata__", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(file_system_info)

        # The remaining keys in the header are the tensor names.
        # We will create the final dictionary structure as requested.
        file_data = {"metadata": metadata}

        tensor_info = {}
        for key, value in header.items():
            # 'value' is a dict like {"dtype": "F16", "shape": [...], "data_offsets": [...]}
            # We only need dtype and shape.
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            tensor_info[key] = {
                "dtype": value.get("dtype", "UNKNOWN"),
                "shape": value.get("shape", []),
            }

        # Unpack the tensor info into the main file data dictionary
        file_data.update(tensor_info)

        return file_data

    except Exception as e:
        # Catch any other unexpected errors during file stat, etc.
        print(f"Warning: Could not process file {file_path}: {e}", file=sys.stderr)
        return None


def build_tree_with_indices(keys: list[str]) -> dict:
    """
    Builds a nested dictionary tree from a list of hierarchically-separated keys.
    Handles both dot-separated keys and underscore-separated keys (common in LoRA).
    A special key `''` is used to store the index of a tensor that is also a parent node.
    """
    root = {}
    key_to_idx_map = {key: i for i, key in enumerate(keys)}
    for key, idx in key_to_idx_map.items():
        # handle both dots and underscores for hierarchy
        parts = parse_hierarchical_key(key)
        current_level = root
        for part in parts:
            current_level = current_level.setdefault(part, {})
        # Use a special key (empty string) to store the index for this node.
        # This ensures current_level remains a dict for potential children.
        current_level[""] = idx
    return root


def parse_hierarchical_key(key: str) -> list[str]:
    """
    Parses a tensor key into lossless, dot-separated hierarchical parts.

    Underscores are meaningful in module names and cannot safely be treated as
    separators. Splitting only on dots guarantees that flattening the tree
    reconstructs every original key exactly.

    Args:
        key: The tensor key to parse

    Returns:
        List of hierarchical parts
    """
    return key.split(".")


def tuplify(obj):
    if isinstance(obj, list):
        return tuple(tuplify(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, tuplify(v)) for k, v in obj.items()))
    return obj


def compress_index(sft_index: dict, trunc_len: Optional[int] = None) -> dict:
    """
    Compresses a safetensors index.
    """

    # --- Pass 1 & 2 are identical to V7 ---
    print("Pass 1: Building global pools for individual specs and strings...")
    spec_pool, string_pool = [], []
    spec_to_idx, string_to_idx = {}, {}
    MIN_STRING_LEN_TO_POOL = 8
    for data in sft_index.values():
        for key, spec in data.items():
            if key == "metadata" or not isinstance(spec, dict):
                continue
            spec_tuple = tuplify(spec)
            if spec_tuple not in spec_to_idx:
                spec_to_idx[spec_tuple] = len(spec_pool)
                spec_pool.append(spec)
        metadata = data.get("metadata", {})
        for key, value in metadata.items():
            if (
                key.startswith("__")
                or not isinstance(value, str)
                or len(value) < MIN_STRING_LEN_TO_POOL
            ):
                continue
            if trunc_len is not None and len(value) > trunc_len:
                value = value[:trunc_len] + "..."
            if value not in string_to_idx:
                string_to_idx[value] = len(string_pool)
                string_pool.append(value)
    print(
        f"Found {len(spec_pool)} unique tensor specs and {len(string_pool)} unique strings."
    )

    print("Pass 2: Grouping files by shareable properties...")
    groups = defaultdict(list)
    for path, data in sft_index.items():
        key_set = frozenset(k for k in data if k != "metadata")
        key_map = sorted(list(key_set))
        spec_indices_tuple = tuple(spec_to_idx[tuplify(data.get(k))] for k in key_map)
        original_metadata = data.get("metadata", {})
        file_meta, user_meta = {}, {}
        for k, v in original_metadata.items():
            if k.startswith("__"):
                file_meta[k] = v
            else:
                user_meta[k] = v
        processed_user_meta = {}
        for k, v in user_meta.items():
            if not isinstance(v, str):
                processed_user_meta[k] = v
                continue
            if trunc_len is not None and len(v) > trunc_len:
                v = v[:trunc_len] + "..."
            if len(v) >= MIN_STRING_LEN_TO_POOL:
                processed_user_meta[k] = {STRING_POOL_REF: string_to_idx[v]}
            else:
                processed_user_meta[k] = v
        user_meta_json = json.dumps(
            processed_user_meta, sort_keys=True, separators=(",", ":")
        )
        group_key = (key_set, spec_indices_tuple, user_meta_json)
        groups[group_key].append({"path": path, "file_meta": file_meta})
    print(f"Found {len(groups)} unique patterns of shareable properties.")

    # --- Pass 3: Assemble the final compressed structure with Optimized Schema Views ---
    print(
        "Pass 3: Assembling final compressed structure with optimized schema views..."
    )
    final_schemas, final_spec_lists, final_user_metadata = [], [], []
    spec_list_map, user_metadata_map = {}, {}
    final_instances = {}

    # 3a. Build the schema pool with subset detection using an inverted index
    key_set_to_schema_id = {}
    base_schema_defs = {}  # Maps a base schema ID to its sorted key list
    key_to_base_schema_ids = defaultdict(list)  # The new inverted index

    all_key_sets = {g[0] for g in groups.keys()}
    sorted_key_sets = sorted(
        all_key_sets, key=lambda keys: (-len(keys), tuple(sorted(keys)))
    )

    print("Building schema pool with optimized subset detection...")
    for key_set in sorted_key_sets:
        found_base_id = None

        # Find candidate supersets using the inverted index
        if key_set:
            candidate_keys = list(key_set)
            # Start with candidates from the first key. Using a set for efficient intersection.
            candidate_base_ids = set(key_to_base_schema_ids.get(candidate_keys[0], []))

            # Intersect with candidates from other keys to narrow down the search space
            for i in range(1, len(candidate_keys)):
                if not candidate_base_ids:
                    break  # Early exit if no candidates remain
                candidate_base_ids.intersection_update(
                    key_to_base_schema_ids.get(candidate_keys[i], [])
                )

            # Now, verify only the few remaining candidates, which are guaranteed to contain all keys
            for base_id in sorted(list(candidate_base_ids)):  # Sort for determinism
                # The check is still required as ground truth, but we do it far fewer times.
                if key_set.issubset(frozenset(base_schema_defs[base_id])):
                    found_base_id = base_id
                    break

        current_schema_id = len(final_schemas)
        if found_base_id is not None:
            # It's a subset, create a "view"
            base_key_map = base_schema_defs[found_base_id]
            base_indices = {key: index for index, key in enumerate(base_key_map)}
            view_indices = [base_indices[key] for key in sorted(key_set)]
            schema_obj = {"base": found_base_id, "view": view_indices}
        else:
            # No superset found, this is a new base schema
            key_map = sorted(list(key_set))
            structure_tree = build_tree_with_indices(key_map)
            schema_obj = {"key_count": len(key_map), "structure_tree": structure_tree}

            # Register it as a potential base for others and update the inverted index
            base_schema_defs[current_schema_id] = key_map
            for key in key_map:
                key_to_base_schema_ids[key].append(current_schema_id)

        final_schemas.append(schema_obj)
        key_set_to_schema_id[key_set] = current_schema_id

    # 3b. Assemble the rest of the structure
    print("Assembling instances...")
    for (key_set, spec_indices_tuple, user_meta_json), items in groups.items():
        schema_id = key_set_to_schema_id[key_set]
        spec_list_id = spec_list_map.setdefault(
            spec_indices_tuple, len(final_spec_lists)
        )
        if spec_list_id == len(final_spec_lists):
            final_spec_lists.append(list(spec_indices_tuple))
        user_meta_id = user_metadata_map.setdefault(
            user_meta_json, len(final_user_metadata)
        )
        if user_meta_id == len(final_user_metadata):
            final_user_metadata.append(json.loads(user_meta_json))
        for item in items:
            path, file_meta = item["path"], item["file_meta"]
            instance = {"s": schema_id, "sl": spec_list_id, "m": user_meta_id}
            if file_meta:
                compact_file_meta = {}
                key_remap = {
                    "__file_size__": "sz",
                    "__file_mtime__": "mt",
                    "__file_ctime__": "ct",
                    "__file_inode__": "in",
                }
                for k, v in file_meta.items():
                    if k in key_remap:
                        compact_file_meta[key_remap[k]] = v
                if compact_file_meta:
                    instance["f"] = compact_file_meta
            final_instances[path] = instance

    return {
        "_METADATA": {"version": "8.1", "source_format": "v7-compatible"},
        "string_pool": string_pool,
        "spec_pool": spec_pool,
        "schemas": final_schemas,
        "spec_list_pool": final_spec_lists,
        "user_metadata_pool": final_user_metadata,
        "instances": final_instances,
    }


class CompressedIndex:
    """
    Loads a compressed safetensors index file into memory and provides
    methods for querying and reconstructing the original data.

    This class is optimized for low memory usage by performing a one-time
    "linking" of data pools during initialization and then discarding the
    raw, index-based data structures. It avoids caching to further reduce
    the memory footprint.
    """

    def __init__(self, compressed_path: Path | str):
        """
        Initializes the loader by reading, linking, and then discarding raw data.

        Args:
            compressed_path: Path to the compressed.json file.
        """
        print(f"Loading and linking compressed index from: {compressed_path}")
        with open(compressed_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self._format_version = raw_data.get("_METADATA", {}).get("version", "8.0")

        # --- Step 1: Link data pools directly, replacing indices ---

        # Load raw pools into local variables, not self, as they are temporary.
        string_pool: List[str] = raw_data["string_pool"]
        self._spec_pool: List[Dict] = raw_data["spec_pool"]  # Keep spec pool on self
        self._schemas: List[Dict] = raw_data["schemas"]  # Keep schemas on self
        spec_list_pool: List[List[int]] = raw_data["spec_list_pool"]
        raw_user_metadata_pool: List[Dict] = raw_data["user_metadata_pool"]
        raw_instances: Dict[str, Dict] = raw_data["instances"]

        # Resolve the user metadata pool in-place. This list will contain the
        # final, fully-resolved metadata objects.
        resolved_user_metadata_pool = [
            self._resolve_metadata_obj(meta_obj, string_pool)
            for meta_obj in raw_user_metadata_pool
        ]

        # --- Step 2: Build the final, linked instances dictionary ---

        self.instances: Dict[str, Dict] = {}
        for path, raw_instance in raw_instances.items():
            # Each instance now holds direct references to the final, shared objects.
            # No copies are made.
            self.instances[path] = {
                # Store IDs, as they are small integers.
                "schema_id": raw_instance["s"],
                # Store a direct reference to the list of spec indices.
                "spec_indices": spec_list_pool[raw_instance["sl"]],
                # Store a direct reference to the fully resolved metadata object.
                "user_metadata": resolved_user_metadata_pool[raw_instance["m"]],
                "file_metadata": self._decompact_file_meta(raw_instance.get("f", {})),
            }

        self._schema_to_paths = defaultdict(list)
        for path, instance in self.instances.items():
            self._schema_to_paths[instance["schema_id"]].append(path)

        print(
            f"Index for {len(self.instances)} files loaded with {len(self._schema_to_paths)} schemas."
        )

    def _decompact_file_meta(self, compact_meta: Dict) -> Dict:
        """Converts compact file metadata keys back to original `__` format."""
        key_remap = {
            "sz": "__file_size__",
            "mt": "__file_mtime__",
            "ct": "__file_ctime__",
            "in": "__file_inode__",
        }
        return {key_remap.get(k, k): v for k, v in compact_meta.items()}

    def _resolve_metadata_obj(self, meta_obj: Any, string_pool: List[str]) -> Any:
        """Recursively replaces integer indices with strings from the string_pool."""
        if isinstance(meta_obj, dict):
            if set(meta_obj) == {STRING_POOL_REF}:
                index = meta_obj[STRING_POOL_REF]
                if not isinstance(index, int) or isinstance(index, bool):
                    raise ValueError("Invalid string-pool reference")
                return string_pool[index]
            return {
                k: self._resolve_metadata_obj(v, string_pool)
                for k, v in meta_obj.items()
            }
        if isinstance(meta_obj, list):
            return [self._resolve_metadata_obj(v, string_pool) for v in meta_obj]
        if isinstance(meta_obj, int) and not isinstance(meta_obj, bool) and self._format_version == "8.0":
            # V8.0 used untagged integer references. Preserve compatibility,
            # although genuine integer metadata in that format is ambiguous.
            return string_pool[meta_obj]
        return meta_obj

    def _flatten_tree(self, tree: Dict, prefix="") -> List[Tuple[str, int]]:
        """
        Recursively flattens a structure_tree back to a list of (key, index) tuples.
        Correctly handles the special `''` key for leaf nodes.
        """
        items = []
        for key, value in tree.items():
            # If the key is our special "leaf" indicator, we've found a full tensor key.
            if key == "":
                # The prefix holds the full key path.
                items.append((prefix, value))
            else:
                # Otherwise, this is a branch. Recurse deeper.
                new_prefix = self._reconstruct_key_path(prefix, key)
                items.extend(self._flatten_tree(value, new_prefix))
        return items

    def _reconstruct_key_path(self, prefix: str, key: str) -> str:
        """
        Reconstructs a key path when building the tree back.
        """
        if not prefix:
            return key

        if self._format_version == "8.0":
            tensor_names = {
                "weight",
                "bias",
                "alpha",
                "dora_scale",
                "lokr_w1",
                "lokr_w2",
                "lokr_w2_a",
                "lokr_w2_b",
                "lora_down",
                "lora_up",
                "lora_A",
                "lora_B",
                "hada_w1_a",
                "hada_w1_b",
                "hada_w2_a",
                "hada_w2_b",
                "t1",
                "t2",
                "ia3_input_mask",
                "diff",
                "diff_b",
                "set_weight",
                "w_norm",
                "b_norm",
            }

            if key in tensor_names:
                return f"{prefix}.{key}"
            return f"{prefix}_{key}"

        return f"{prefix}.{key}"

    def get_key_map(self, schema_id: int) -> List[str]:
        """
        Resolves a schema ID to its deterministically sorted list of tensor keys.
        This method computes the result on each call, avoiding memory caches.

        Args:
            schema_id: The integer index of the schema in the `schemas` pool.

        Returns:
            A list of strings, where each string is a tensor key.
        """
        schema_obj = self._schemas[schema_id]

        if "base" in schema_obj:  # It's a Sub-Schema View
            base_key_map = self.get_key_map(schema_obj["base"])
            view_indices = schema_obj["view"]
            return [base_key_map[i] for i in view_indices]
        else:  # It's a Base Schema
            flat_map = self._flatten_tree(schema_obj["structure_tree"])
            # Sort by index to restore the original deterministic order
            flat_map.sort(key=lambda x: x[1])
            return [key for key, index in flat_map]

    def decompress_path(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Fully reconstructs the original index data for a single file path.

        Args:
            path: The absolute path of the file to reconstruct.

        Returns:
            A dictionary mirroring the original sft_index.json format for that file,
            or None if the path is not found.
        """
        instance = self.instances.get(path)
        if not instance:
            return None

        # 1. Reconstruct metadata by merging the two linked metadata dictionaries.
        reconstructed_data = {
            "metadata": {**instance["user_metadata"], **instance["file_metadata"]}
        }

        # 2. Get the list of tensor keys for this file's architecture.
        key_map = self.get_key_map(instance["schema_id"])

        # 3. Get the list of spec indices for this file's specific variant.
        spec_indices = instance["spec_indices"]

        # 4. Map keys to their specs by referencing the main spec pool.
        if len(key_map) != len(spec_indices):
            raise ValueError(
                f"Mismatch for path {path}: {len(key_map)} keys but {len(spec_indices)} specs."
            )

        for i, key in enumerate(key_map):
            spec_index = spec_indices[i]
            reconstructed_data[key] = self._spec_pool[spec_index]

        return reconstructed_data

    def decompress_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Decompresses the entire index back to its original format.

        Returns:
            A dictionary identical in structure to the original sft_index.json.
        """
        print("Decompressing all entries...")
        reconstructed_index = {}
        # Iterate over self.instances which is already in memory
        for path in self.instances.keys():
            reconstructed_index[path] = self.decompress_path(path)
        return reconstructed_index

    def get_schema_info(self, schema_id: int) -> dict:
        """
        Returns summary information about a specific schema.

        This is useful for understanding if an architecture is a base model
        or a subset (like a LoRA).

        Args:
            schema_id: The integer index of the schema.

        Returns:
            A dictionary containing info like type ('base' or 'view'),
            key_count, and base_id if it is a view.
        """
        if not 0 <= schema_id < len(self._schemas):
            raise IndexError(f"Schema ID {schema_id} is out of bounds.")

        schema_obj = self._schemas[schema_id]
        key_map = self.get_key_map(schema_id)

        if "base" in schema_obj:
            return {
                "type": "view",
                "key_count": len(key_map),
                "base_id": schema_obj["base"],
            }
        else:
            return {
                "type": "base",
                "key_count": len(key_map),
            }

    def get_paths_for_schema(self, schema_id: int) -> list[str]:
        """
        Returns all file paths that belong to a given schema ID.

        This is highly efficient if an inverted index (_schema_to_paths) is
        built during initialization. It allows for analysis of an entire
        architectural family.

        Args:
            schema_id: The integer index of the schema.

        Returns:
            A list of file paths matching the schema.
        """
        # This assumes the inverted index `self._schema_to_paths` was built in __init__.
        return self._schema_to_paths.get(schema_id, [])

    def get_raw_tree_for_path(self, path: str) -> dict | None:
        """
        Constructs the full, nested dictionary tree for a single file path, with
        tensor spec information at the leaves. Groups adapter modules by
        recognizing common adapter tensor patterns.

        Args:
            path: The file path to build the tree for.

        Returns:
            A nested dictionary representing the model's structure, or None if
            the path is not found.
        """
        instance = self.instances.get(path)
        if not instance:
            return None

        key_map = self.get_key_map(instance["schema_id"])
        spec_indices = instance["spec_indices"]

        # Create a temporary key-to-spec mapping for this specific instance
        key_to_spec = {
            key_map[i]: self._spec_pool[spec_indices[i]] for i in range(len(key_map))
        }

        # Group keys by potential adapter modules first
        adapter_groups = self._group_adapter_keys(key_to_spec.keys())

        # Build the tree from the flat key-to-spec map
        root = {}
        processed_keys = set()

        # First, handle grouped adapter modules
        for module_path, tensor_keys in adapter_groups.items():
            if len(tensor_keys) > 1:  # Only group if multiple related tensors
                # Parse the module path hierarchically
                module_parts = parse_hierarchical_key(module_path)
                current_level = root
                for part in module_parts[:-1]:
                    current_level = current_level.setdefault(part, {})

                # Create the adapter module container
                adapter_node = current_level.setdefault(module_parts[-1], {})

                # Add all related tensors to this adapter module
                for tensor_key in tensor_keys:
                    # Parse the full tensor key and determine the suffix within the module
                    tensor_parts = parse_hierarchical_key(tensor_key)
                    module_depth = len(module_parts)

                    # Get the tensor-specific parts (beyond the module path)
                    tensor_suffix_parts = tensor_parts[module_depth:]

                    tensor_current = adapter_node
                    for tensor_part in tensor_suffix_parts[:-1]:
                        tensor_current = tensor_current.setdefault(tensor_part, {})

                    # Place the tensor spec
                    if tensor_suffix_parts:
                        leaf_node = tensor_current.setdefault(
                            tensor_suffix_parts[-1], {}
                        )
                        leaf_node["_info"] = key_to_spec[tensor_key]
                    else:
                        # Edge case: tensor name is exactly the module name
                        tensor_current["_info"] = key_to_spec[tensor_key]

                # Mark this as a grouped adapter module for analysis
                adapter_node["_adapter_group"] = True
                adapter_node["_adapter_tensors"] = tensor_keys

                processed_keys.update(tensor_keys)

        # Then handle remaining individual tensors
        for key, spec_info in key_to_spec.items():
            if key in processed_keys:
                continue

            parts = parse_hierarchical_key(key)
            current_level = root
            for part in parts[:-1]:
                current_level = current_level.setdefault(part, {})
            # Place the final spec under a special '_info' key
            leaf_node = current_level.setdefault(parts[-1], {})
            leaf_node["_info"] = spec_info

        return root

    def _group_adapter_keys(self, keys: list[str]) -> dict[str, list[str]]:
        """
        Groups tensor keys that belong to the same adapter module.

        Args:
            keys: List of tensor keys

        Returns:
            Dictionary mapping module paths to lists of tensor keys
        """
        from collections import defaultdict

        # Common adapter tensor suffixes that indicate related tensors
        adapter_suffixes = {
            ".alpha",
            ".dora_scale",
            ".lora_down.weight",
            ".lora_up.weight",
            ".lora_mid.weight",
            ".lora_A.weight",
            ".lora_B.weight",
            ".hada_w1_a.weight",
            ".hada_w1_b.weight",
            ".hada_w2_a.weight",
            ".hada_w2_b.weight",
            ".lokr_w1",
            ".lokr_w2",
            ".lokr_w1.weight",
            ".lokr_w2.weight",
            ".lokr_w2_a",
            ".lokr_w2_b",  # Additional LoKr variants
            ".t1.weight",
            ".t2.weight",
            ".ia3_input_mask.weight",
            ".diff.weight",
            ".diff_b",
            ".set_weight",
            ".w_norm",
            ".b_norm",
        }

        groups = defaultdict(list)

        for key in keys:
            # Find the longest matching adapter suffix
            for suffix in adapter_suffixes:
                if key.endswith(suffix):
                    module_path = key[: -len(suffix)]
                    groups[module_path].append(key)
                    break

        # Only return groups with multiple keys (indicating a real adapter module)
        result = {
            module_path: tensor_keys
            for module_path, tensor_keys in groups.items()
            if len(tensor_keys) > 1
        }

        return result
