import pytest
import safetensors.torch
import torch

from loralib import BaseCheckpoint, ConcatLoRAsDict, LoRADict, PairedLoraModel
from resize_lora import ResizeRecipe, process_lora_model

RESIZED = "lora_unet_input_blocks_4_1_proj_in"
DIFF = "lora_unet_input_blocks_1_0_emb_layers_1.diff"
UNMAPPED = "lora_te2_text_projection"
MISMATCHED = "lora_unet_time_embed_0"
SDXL_BASE = {
    "model.diffusion_model.input_blocks.4.1.proj_in.weight": (32, 32),
    "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight": (32, 64),
    "model.diffusion_model.time_embed.0.weight": (32, 16),
    "conditioner.embedders.1.model.text_projection": (32, 32),
}


def _lora_layer(name, out_dim, in_dim, rank=4, alpha=None):
    return {
        f"{name}.alpha": torch.tensor(float(rank if alpha is None else alpha)),
        f"{name}.lora_down.weight": torch.randn(rank, in_dim),
        f"{name}.lora_up.weight": torch.randn(out_dim, rank),
    }


def _save(path, tensors):
    safetensors.torch.save_file(tensors, str(path))  # no metadata
    return path


def _run(tmp_path, base, lora_dict, recipe):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paired = PairedLoraModel(lora_dict, BaseCheckpoint(base))
    process_lora_model(
        paired,
        [ResizeRecipe(recipe)],
        out_dir,
        device="cpu",
        output_dtype=torch.float32,
    )
    (output_path,) = out_dir.iterdir()
    return paired, safetensors.torch.load_file(output_path)


def _resize(tmp_path, recipe):
    torch.manual_seed(0)
    base = _save(
        tmp_path / "ckpt.safetensors",
        {k: torch.randn(*shape) for k, shape in SDXL_BASE.items()},
    )
    lora = (
        _lora_layer(RESIZED, 32, 32)
        | _lora_layer(UNMAPPED, 32, 32)
        | _lora_layer(MISMATCHED, 32, 24)
    )
    lora[DIFF] = torch.randn(32, 64)
    lora_path = _save(tmp_path / "lora.safetensors", lora)
    _, output = _run(tmp_path, base, LoRADict(lora_path), recipe)
    return lora, output


def test_unresizable_tensors_are_copied_unchanged(tmp_path):
    lora, output = _resize(tmp_path, "fro_ckpt=1,thr=-9")

    assert output[f"{RESIZED}.lora_down.weight"].shape[0] == 4
    copied = [DIFF, *(k for k in lora if k.startswith((UNMAPPED, MISMATCHED)))]
    for key in copied:
        assert torch.equal(output[key], lora[key])


def test_size_budget_counts_copied_tensors(tmp_path):
    # .diff, then the unmapped and the mismatched rank-4 layers with their alphas
    copied_bytes = 4 * (32 * 64 + 4 * (32 + 32) + 1 + 4 * (32 + 24) + 1)
    dim_bytes = 4 * (32 + 32)
    target_mib = (copied_bytes + 2.5 * dim_bytes) / (1 << 20)

    _, output = _resize(tmp_path, f"fro_ckpt=1,size={target_mib}")

    # Without the copied bytes, the budget would hold all 4 dims.
    assert output[f"{RESIZED}.lora_down.weight"].shape[0] == 3


def test_peft_layer_on_fused_qkv_keeps_its_format(tmp_path):
    torch.manual_seed(0)
    qkv = torch.randn(96, 32)
    base = _save(
        tmp_path / "flux.safetensors", {"double_blocks.0.img_attn.qkv.weight": qkv}
    )
    name = "transformer.transformer_blocks.0.attn.to_k"
    down, up = torch.randn(4, 32), torch.randn(32, 4)
    lora_path = _save(
        tmp_path / "peft.safetensors",
        {f"{name}.lora_A.weight": down, f"{name}.lora_B.weight": up},
    )

    paired, output = _run(tmp_path, base, LoRADict(lora_path), "fro_ckpt=1,thr=-9")

    assert set(output) == {f"{name}.lora_A.weight", f"{name}.lora_B.weight"}
    torch.testing.assert_close(
        output[f"{name}.lora_B.weight"] @ output[f"{name}.lora_A.weight"],
        up @ down,
        rtol=1e-4,
        atol=1e-4,
    )
    norm = paired.checkpoint.frobenius_norm(paired.targets[name])
    assert norm == pytest.approx(torch.linalg.matrix_norm(qkv[32:64]).item())


def test_merge_concatenates_weighted_layers(tmp_path):
    torch.manual_seed(0)
    base = _save(
        tmp_path / "ckpt.safetensors", {next(iter(SDXL_BASE)): torch.randn(32, 32)}
    )
    first = _lora_layer(RESIZED, 32, 32, rank=4, alpha=4)
    second = _lora_layer(RESIZED, 32, 32, rank=2, alpha=1)
    lora_dict = ConcatLoRAsDict(
        [
            (_save(tmp_path / "a.safetensors", first), 0.7),
            (_save(tmp_path / "b.safetensors", second), 0.3),
        ]
    )

    _, output = _run(tmp_path, base, lora_dict, "fro_ckpt=1,thr=-9")

    def delta(sd, scale):
        return (
            scale * sd[f"{RESIZED}.lora_up.weight"] @ sd[f"{RESIZED}.lora_down.weight"]
        )

    rank = output[f"{RESIZED}.lora_down.weight"].shape[0]
    assert rank == 6
    torch.testing.assert_close(
        delta(output, output[f"{RESIZED}.alpha"].item() / rank),
        delta(first, 0.7 * 4 / 4) + delta(second, 0.3 * 1 / 2),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("scale_name", ["scale_weight", "weight_scale"])
def test_fp8_base_weights_are_dequantized(tmp_path, scale_name):
    torch.manual_seed(0)
    weight = torch.randn(32, 32)
    scale = torch.tensor(0.01)
    key = "double_blocks.0.img_attn.proj.weight"
    base = _save(
        tmp_path / "fp8.safetensors",
        {
            key: (weight / scale).to(torch.float8_e4m3fn),
            key.removesuffix("weight") + scale_name: scale,
        },
    )
    checkpoint = BaseCheckpoint(base)

    norm = checkpoint.frobenius_norm(checkpoint.target(f"diffusion_model.{key[:-7]}"))

    assert norm == pytest.approx(torch.linalg.matrix_norm(weight).item(), rel=0.02)
