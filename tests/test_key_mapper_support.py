from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from loralib.key_mapper import KeyMapper
from loralib.sources import AdapterFileSource
from test_key_mapper import calculate_compatibility_detailed


def _write_safetensors(path, tensors, metadata=None):
    safetensors.torch.save_file(tensors, str(path), metadata=metadata or {})


def test_key_mapper_builds_and_maps_basic_comfyui_prefixes(tmp_path):
    base_path = tmp_path / "base.safetensors"
    _write_safetensors(
        base_path,
        {
            "model.diffusion_model.input_blocks.1.1.proj_in.weight": torch.zeros(
                (2, 2)
            ),
            "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": torch.zeros(
                (2, 2)
            ),
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.weight": torch.zeros(
                (2, 2)
            ),
        },
    )

    key_mapper = KeyMapper(base_path)

    result = key_mapper.map_from_lora(
        "lora_unet_input_blocks_1_1_proj_in.lora_down.weight"
    )

    assert result is not None
    assert result.canonical_key == "model.diffusion_model.input_blocks.1.1.proj_in.weight"
    assert result.lora_key_base == "lora_unet_input_blocks_1_1_proj_in"
    assert result.suffix == ".lora_down.weight"
    assert result.matched_rule == "DictionaryLookup"


def test_adapter_file_source_reads_keys_and_metadata(tmp_path):
    adapter_path = tmp_path / "adapter.safetensors"
    _write_safetensors(
        adapter_path,
        {
            "block.lora_down.weight": torch.ones((2, 2)),
            "block.lora_up.weight": torch.ones((2, 2)) * 2,
        },
        metadata={"ss_training": "demo"},
    )

    source = AdapterFileSource(adapter_path)

    assert set(source.keys()) == {
        "block.lora_down.weight",
        "block.lora_up.weight",
    }
    assert source.metadata()["ss_training"] == "demo"
    assert torch.equal(source.get_tensor("block.lora_down.weight"), torch.ones((2, 2)))


def test_compatibility_penalty_uses_key_mapper_model_types():
    class FakeKeyMapper:
        def __init__(self):
            self.context = SimpleNamespace(
                model_type="SDXL-Base",
                components_present=set(),
            )
            self._all = {
                "lora_unet_joint_blocks_0_context_block_attn_q_proj": "canonical"
            }

        def _strip_suffix(self, key):
            if key.endswith(".weight"):
                return key[: -len(".weight")], ".weight"
            return None

        def get_all_mappings(self):
            return self._all

    score, details = calculate_compatibility_detailed(
        FakeKeyMapper(),
        {"lora_unet_joint_blocks_0_context_block_attn_q_proj.weight"},
    )

    assert score == pytest.approx(0.5)
    assert details["architecture_penalty"] == pytest.approx(0.5)
    assert details["base_score"] == pytest.approx(1.0)
