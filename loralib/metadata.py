# common functions for training

from io import BytesIO
import safetensors.torch
import hashlib

def precalculate_safetensors_hashes(tensors, metadata):
    """
    Precalculates model hashes for a given set of tensors and metadata.

    This function computes hashes compatible with sd-webui-additional-networks.
    It ensures that only training-specific metadata (keys starting with "ss_")
    is used for hash calculation to maintain hash consistency even if other
    user metadata changes.

    Args:
        tensors (dict): A dictionary of tensors to be saved.
        metadata (dict): A dictionary of metadata associated with the tensors.

    Returns:
        tuple: A tuple containing the model_hash and legacy_hash.
    """

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes_data = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes_data) # Create an in-memory binary stream

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash

def addnet_hash_legacy(b):
    """
    Calculates the legacy model hash used by sd-webui-additional-networks.

    This hash is based on a specific portion of the .safetensors file (0x10000 bytes
    starting at offset 0x100000).

    Args:
        b (BytesIO): An in-memory binary stream of the safetensors data.

    Returns:
        str: The first 8 characters of the SHA256 hexdigest.
    """
    m = hashlib.sha256()

    b.seek(0x100000) # Seek to the predefined offset
    m.update(b.read(0x10000)) # Read a fixed number of bytes
    return m.hexdigest()[0:8] # Return the first 8 characters of the hash


def addnet_hash_safetensors(b):
    """
    Calculates the new model hash used by sd-webui-additional-networks for .safetensors files.

    This hash is computed over the tensor data section of the .safetensors file,
    excluding the initial JSON header.

    Args:
        b (BytesIO): An in-memory binary stream of the safetensors data.

    Returns:
        str: The full SHA256 hexdigest.
    """
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024 # 1MB block size for reading

    b.seek(0) # Go to the beginning of the stream
    header = b.read(8) # Read the 8-byte header indicating the length of the JSON metadata
    n = int.from_bytes(header, "little") # Convert header bytes to an integer (length of JSON)

    offset = n + 8 # Calculate the offset to the start of the tensor data
    b.seek(offset) # Seek past the JSON metadata
    # Read the rest of the file in chunks and update the hash
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()
