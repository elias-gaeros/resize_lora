from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from loralib.key_mapper import KeyMapper
from loralib.key_mapper.generators import MappingGenerator
from loralib.sources import AdapterFileSource
from test_key_mapper import CheckpointAssessor, calculate_compatibility_detailed


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


@pytest.mark.parametrize(
    "suffix",
    [
        ".lora_A.weight",
        ".lora_proj_down",
        ".hada_w1_a.weight",
        ".lokr_w1.weight",
        ".t1.weight",
        ".ia3_input_mask.weight",
    ],
)
def test_key_mapper_maps_direct_comfyui_dit_formats(tmp_path, suffix):
    base_path = tmp_path / "dit.safetensors"
    canonical = "double_blocks.0.img_attn.proj.weight"
    _write_safetensors(base_path, {canonical: torch.zeros((2, 2))})

    key_mapper = KeyMapper(base_path)
    result = key_mapper.map_from_lora(
        f"diffusion_model.double_blocks.0.img_attn.proj{suffix}"
    )

    assert result is not None
    assert result.canonical_key == canonical


def test_key_mapper_rejects_conflicting_generator_aliases(tmp_path):
    base_path = tmp_path / "base.safetensors"
    _write_safetensors(
        base_path,
        {"model.diffusion_model.block.weight": torch.zeros((2, 2))},
    )

    class First(MappingGenerator):
        def generate(self, context, existing_mapping):
            return {"duplicate": "first.weight"}

    class Second(MappingGenerator):
        def generate(self, context, existing_mapping):
            return {"duplicate": "second.weight"}

    with pytest.raises(ValueError, match="conflicting mapping"):
        KeyMapper(base_path, generators=[First(), Second()])


def test_key_mapper_maps_diffusers_attention_aliases(tmp_path):
    base_path = tmp_path / "sdxl.safetensors"
    canonical = (
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0."
        "attn1.to_q.weight"
    )
    _write_safetensors(
        base_path,
        {
            canonical: torch.zeros((2, 2)),
            "model.diffusion_model.input_blocks.3.0.op.weight": torch.zeros((2, 2)),
        },
    )

    key_mapper = KeyMapper(base_path)
    result = key_mapper.map_from_lora(
        "down_blocks.0.attentions.0.transformer_blocks.0."
        "attn1.to_q.lora_down.weight"
    )

    assert result is not None
    assert result.canonical_key == canonical


@pytest.mark.parametrize(
    ("canonical", "adapter_key"),
    [
        (
            "diffusion_model.double_blocks.0.img_attn.proj.weight",
            "diffusion_model.double_blocks.0.img_attn.proj.lora_down.weight",
        ),
        (
            "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
            "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
        ),
    ],
)
def test_key_mapper_detects_wrapped_dit_checkpoints(tmp_path, canonical, adapter_key):
    base_path = tmp_path / "wrapped-dit.safetensors"
    _write_safetensors(base_path, {canonical: torch.zeros((2, 2))})

    key_mapper = KeyMapper(base_path)

    assert "DiT" in key_mapper.context.components_present
    assert key_mapper.map_from_lora(adapter_key).canonical_key == canonical


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
    source.close()
    source.close()


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


def test_checkpoint_assessor_recognizes_svdquant_lora():
    classification, _ = CheckpointAssessor(
        {"diffusion_model.block.lora_proj_down", "diffusion_model.block.lora_proj_up"}
    ).classify()

    assert classification == "ADAPTER"


def test_compatibility_accepts_detected_dit_without_known_model_name():
    class FakeKeyMapper:
        context = SimpleNamespace(
            model_type="Unknown", components_present={"DiT"}
        )

        def _strip_suffix(self, key):
            return key[: -len(".weight")], ".weight"

        def get_all_mappings(self):
            return {"diffusion_model.double_blocks.0.img_attn.proj": "canonical"}

    score, details = calculate_compatibility_detailed(
        FakeKeyMapper(),
        {"diffusion_model.double_blocks.0.img_attn.proj.weight"},
    )

    assert score == pytest.approx(1.0)
    assert details["architecture_penalty"] == 0.0
