import argparse
import json
from pathlib import Path
import safetensors
from tqdm import tqdm
import sys
import struct

MAX_HEADER_SIZE = 256 * 1024 * 1024


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
        # print(f"Debug: Failed to parse header for {file_path}: {e}", file=sys.stderr)
        return None


# --- Updated index_safetensors_file function ---


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


def main():
    """
    Main function to parse arguments, find files, and create the index.
    """
    parser = argparse.ArgumentParser(
        description="Recursively find all .safetensors files in given directories and create a JSON index of their contents without loading tensors into memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directories",
        type=str,
        nargs="+",
        help="One or more directories to scan recursively for .safetensors files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="sft_index.json",
        help="Path to the output JSON index file. Will be created in the current directory by default.",
    )
    args = parser.parse_args()

    # --- 1. Find all .safetensors files ---
    print("Searching for .safetensors files...")
    files_to_process = {}
    for dir_path_str in args.directories:
        dir_path = Path(dir_path_str)
        if not dir_path.is_dir():
            print(
                f"Warning: '{dir_path_str}' is not a valid directory. Skipping.",
                file=sys.stderr,
            )
            continue
        for file_path in dir_path.rglob("*.safetensors"):
            files_to_process[str(file_path.resolve())] = file_path.resolve()

    if not files_to_process:
        print("No .safetensors files found. Exiting.")
        return

    print(f"Found {len(files_to_process)} files. Indexing...")

    # --- 2. Process each file and build the index ---
    sft_index = {}
    for file_path in tqdm(sorted(files_to_process.values()), desc="Indexing files"):
        file_data = index_safetensors_file(file_path)
        if file_data:
            # Use the absolute path as a unique key in the index
            sft_index[str(file_path.resolve())] = file_data

    # --- 3. Save the index to a JSON file ---
    output_path = Path(args.output).resolve()
    print(f"\nSaving index with {len(sft_index)} entries to {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sft_index, f, indent=2)
        print("Done.")
    except Exception as e:
        print(f"Error: Failed to save JSON file to {output_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
