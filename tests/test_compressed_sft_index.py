import json
import struct

from compressed_sft_index import CompressedIndex, compress_index, parse_safetensors_header


def test_compressed_index_round_trips_keys_and_typed_metadata(tmp_path):
    original = {
        "adapter": {
            "metadata": {
                "count": 42,
                "enabled": True,
                "description": "a sufficiently long pooled string",
                "nested": {"epoch": 3},
            },
            "plain_under_score": {"dtype": "F16", "shape": [1]},
            "lora_unet_block.lora_down.weight": {
                "dtype": "F16",
                "shape": [2, 1],
            },
        }
    }
    compressed = compress_index(original)
    path = tmp_path / "index.json"
    path.write_text(json.dumps(compressed), encoding="utf-8")

    restored = CompressedIndex(path).decompress_all()

    assert restored == original
    assert compressed["_METADATA"]["version"] == "8.1"


def test_parse_header_rejects_length_larger_than_file(tmp_path):
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(struct.pack("<Q", 10_000) + b"{}")

    assert parse_safetensors_header(path) is None


def test_compressed_index_reads_legacy_v8_tree_format(tmp_path):
    path = tmp_path / "legacy.json"
    legacy = {
        "_METADATA": {"version": "8.0", "source_format": "v7-compatible"},
        "string_pool": [],
        "spec_pool": [{"dtype": "F16", "shape": [1]}],
        "schemas": [
            {
                "key_count": 1,
                "structure_tree": {
                    "lora": {
                        "unet": {
                            "block.lora": {"down": {"weight": {"": 0}}}
                        }
                    }
                },
            }
        ],
        "spec_list_pool": [[0]],
        "user_metadata_pool": [{}],
        "instances": {"file.safetensors": {"s": 0, "sl": 0, "m": 0}},
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")

    restored = CompressedIndex(path).decompress_all()

    assert restored == {
        "file.safetensors": {
            "metadata": {},
            "lora_unet_block.lora_down.weight": {"dtype": "F16", "shape": [1]},
        }
    }
