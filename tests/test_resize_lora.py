import safetensors.torch
import torch

from loralib import BaseCheckpoint, LoRADict, PairedLoraModel
from resize_lora import ResizeRecipe, process_lora_model

RESIZED = "lora_unet_input_blocks_4_1_proj_in"
DIFF = "lora_unet_input_blocks_1_0_emb_layers_1.diff"
UNMAPPED = "lora_te2_text_projection"


def _lora_layer(name, out_dim, in_dim, rank=4):
    return {
        f"{name}.alpha": torch.tensor(float(rank)),
        f"{name}.lora_down.weight": torch.randn(rank, in_dim),
        f"{name}.lora_up.weight": torch.randn(out_dim, rank),
    }


def _resize(tmp_path, recipe):
    torch.manual_seed(0)
    checkpoint_path = tmp_path / "ckpt.safetensors"
    safetensors.torch.save_file(
        {
            "model.diffusion_model.input_blocks.4.1.proj_in.weight": torch.randn(
                32, 32
            ),
            "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight": torch.randn(
                32, 64
            ),
            "conditioner.embedders.1.model.text_projection": torch.randn(32, 32),
        },
        str(checkpoint_path),
    )
    lora = _lora_layer(RESIZED, 32, 32) | _lora_layer(UNMAPPED, 32, 32)
    lora[DIFF] = torch.randn(32, 64)
    lora_path = tmp_path / "lora.safetensors"
    safetensors.torch.save_file(lora, str(lora_path))  # no metadata

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paired = PairedLoraModel(LoRADict(lora_path), BaseCheckpoint(checkpoint_path))
    process_lora_model(
        paired,
        [ResizeRecipe(recipe)],
        out_dir,
        device="cpu",
        output_dtype=torch.float32,
    )
    (output_path,) = out_dir.iterdir()
    return lora, safetensors.torch.load_file(output_path)


def test_unresizable_tensors_are_copied_unchanged(tmp_path):
    lora, output = _resize(tmp_path, "fro_ckpt=1,thr=-9")

    assert output[f"{RESIZED}.lora_down.weight"].shape[0] == 4
    for key in [DIFF, *(k for k in lora if k.startswith(UNMAPPED))]:
        assert torch.equal(output[key], lora[key])


def test_size_budget_counts_copied_tensors(tmp_path):
    copied_bytes = 4 * (32 * 64 + 2 * 4 * 32 + 1)
    dim_bytes = 4 * (32 + 32)
    target_mib = (copied_bytes + 2.5 * dim_bytes) / (1 << 20)

    _, output = _resize(tmp_path, f"fro_ckpt=1,size={target_mib}")

    # Without the copied bytes, the budget would hold all 4 dims.
    assert output[f"{RESIZED}.lora_down.weight"].shape[0] == 3
