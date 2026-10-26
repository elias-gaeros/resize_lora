import pytest
import safetensors.torch
import torch

from loralib import (
    BaseCheckpoint,
    ConcatLoRAsDict,
    CtMLoRAsDict,
    LoRADict,
    PairedLoraModel,
)
from resize_lora import MergeSpec, ResizeRecipe, process_lora_model

RESIZED = "lora_unet_input_blocks_4_1_proj_in"
EMB = "lora_unet_input_blocks_1_0_emb_layers_1"
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


def _delta(sd, name):
    up, down = sd[f"{name}.lora_up.weight"], sd[f"{name}.lora_down.weight"]
    return sd[f"{name}.alpha"].item() / down.shape[0] * up @ down


def test_ctm_keeping_every_dimension_is_the_weighted_sum(tmp_path):
    torch.manual_seed(0)
    base = _save(
        tmp_path / "ckpt.safetensors", {next(iter(SDXL_BASE)): torch.randn(32, 32)}
    )
    first = _lora_layer(RESIZED, 32, 32, rank=4, alpha=4)
    second = _lora_layer(RESIZED, 32, 32, rank=2, alpha=1)
    lora_dict = CtMLoRAsDict(
        [
            (_save(tmp_path / "a.safetensors", first), 0.7),
            (_save(tmp_path / "b.safetensors", second), 0.3),
        ]
    )

    _, output = _run(tmp_path, base, lora_dict, "fro_ckpt=1,thr=-9")

    assert output[f"{RESIZED}.lora_down.weight"].shape[0] == 6
    torch.testing.assert_close(
        _delta(output, RESIZED),
        0.7 * _delta(first, RESIZED) + 0.3 * _delta(second, RESIZED),
        rtol=1e-4,
        atol=1e-4,
    )


def test_ctm_resizes_a_layer_only_one_lora_trained_like_that_lora_alone(tmp_path):
    torch.manual_seed(0)
    base = _save(
        tmp_path / "ckpt.safetensors",
        {k: torch.randn(*shape) for k, shape in list(SDXL_BASE.items())[:2]},
    )
    first = _lora_layer(RESIZED, 32, 32) | _lora_layer(EMB, 32, 64)
    second = _lora_layer(RESIZED, 32, 32) | _lora_layer(EMB, 32, 64)
    second[f"{EMB}.lora_up.weight"].zero_()  # untrained, like kohya's last TE layer
    first_path = _save(tmp_path / "a.safetensors", first)
    lora_dict = CtMLoRAsDict(
        [(first_path, 1.0), (_save(tmp_path / "b.safetensors", second), 1.0)],
        ties=0.5,
    )
    recipe = "spn_lora=1,thr=-0.3"
    (tmp_path / "alone").mkdir()
    (tmp_path / "merged").mkdir()

    _, alone = _run(tmp_path / "alone", base, LoRADict(first_path), recipe)
    _, merged = _run(tmp_path / "merged", base, lora_dict, recipe)

    # Same rank, same truncation: not TIES-trimmed, nor NaN from the zero layer
    rank = alone[f"{EMB}.lora_down.weight"].shape[0]
    assert 0 < rank < 4
    assert merged[f"{EMB}.lora_down.weight"].shape[0] == rank
    torch.testing.assert_close(
        _delta(merged, EMB), _delta(alone, EMB), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(
    "beta, kept", [(1.0, [10.0, 9.0, 0.0]), (0.0, [10.0, 0.0, 5.0])]
)
def test_ctm_beta_chooses_which_dimensions_not_how_many(tmp_path, beta, kept):
    # Orthogonal layers: the first has singular values 10 and 9, the second 5.
    eye = torch.eye(32)
    first = {
        f"{RESIZED}.alpha": torch.tensor(2.0),
        f"{RESIZED}.lora_down.weight": eye[:2],
        f"{RESIZED}.lora_up.weight": eye[:, :2] * torch.tensor([10.0, 9.0]),
    }
    second = {
        f"{RESIZED}.alpha": torch.tensor(1.0),
        f"{RESIZED}.lora_down.weight": eye[2:3],
        f"{RESIZED}.lora_up.weight": 5.0 * eye[:, 2:3],
    }
    base = _save(
        tmp_path / "ckpt.safetensors", {next(iter(SDXL_BASE)): torch.randn(32, 32)}
    )
    lora_dict = CtMLoRAsDict(
        [
            (_save(tmp_path / "a.safetensors", first), 1.0),
            (_save(tmp_path / "b.safetensors", second), 1.0),
        ],
        beta=beta,
    )

    # The recipe keeps the norms within 10^-0.1 of the largest: 10 and 9, so 2 dims.
    _, output = _run(tmp_path, base, lora_dict, "spn_lora=1,thr=-0.1")

    # beta=0 gives both LoRAs norm 9.23 to choose the subspaces: 5 becomes 9.23 and
    # 9 becomes 6.17, so the second LoRA's direction replaces the first's weaker one.
    delta = _delta(output, RESIZED)
    torch.testing.assert_close(
        delta[:3, :3], torch.diag(torch.tensor(kept)), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(
    "name, out_dim, in_dim, rank",
    [(EMB, 32, 64, 20), ("lora_unet_time_embed_0", 32, 16, 12)],
)
def test_ctm_is_lossless_when_the_ranks_exceed_a_layer_dimension(
    tmp_path, name, out_dim, in_dim, rank
):
    # The summed rank exceeds one dimension of the layer, so the cores are not square.
    torch.manual_seed(0)
    first = _lora_layer(name, out_dim, in_dim, rank=rank)
    second = _lora_layer(name, out_dim, in_dim, rank=rank)
    lora_dict = CtMLoRAsDict(
        [
            (_save(tmp_path / "a.safetensors", first), 1.0),
            (_save(tmp_path / "b.safetensors", second), 1.0),
        ]
    )
    layer = lora_dict.decompose(name)

    output = layer.statedict()
    truncated = layer.statedict(mask=torch.arange(layer.dim) < 8)

    assert layer.dim == min(out_dim, in_dim)
    torch.testing.assert_close(
        _delta(output, name),
        _delta(first, name) + _delta(second, name),
        rtol=1e-4,
        atol=1e-4,
    )
    assert truncated[f"{name}.lora_down.weight"].shape[0] == 8


def test_ctm_warns_that_subspace_scores_set_only_the_rank(tmp_path, caplog):
    torch.manual_seed(0)
    base = _save(
        tmp_path / "ckpt.safetensors", {next(iter(SDXL_BASE)): torch.randn(32, 32)}
    )
    lora_dict = CtMLoRAsDict(
        [
            (_save(tmp_path / f"{i}.safetensors", _lora_layer(RESIZED, 32, 32)), 1.0)
            for i in range(2)
        ],
        ties=0.25,
    )

    _, output = _run(tmp_path, base, lora_dict, "subspace=1,thr=-9")

    assert "not which" in caplog.text
    assert all(torch.isfinite(v).all() for v in output.values())


def test_merge_warns_when_two_names_modify_one_base_layer(tmp_path, caplog):
    torch.manual_seed(0)
    base = _save(
        tmp_path / "ckpt.safetensors", {next(iter(SDXL_BASE)): torch.randn(32, 32)}
    )
    kohya = _lora_layer(RESIZED, 32, 32)
    comfy = _lora_layer("diffusion_model.input_blocks.4.1.proj_in", 32, 32)
    lora_dict = ConcatLoRAsDict(
        [
            (_save(tmp_path / "a.safetensors", kohya), 1.0),
            (_save(tmp_path / "b.safetensors", comfy), 1.0),
        ]
    )

    paired = PairedLoraModel(lora_dict, BaseCheckpoint(base))

    assert len(set(paired.targets.values())) == 1 < len(paired.targets)
    assert "ComfyUI applies only one" in caplog.text


def test_outputs_with_the_same_name_are_not_overwritten(tmp_path, caplog):
    torch.manual_seed(0)
    base = _save(
        tmp_path / "ckpt.safetensors", {next(iter(SDXL_BASE)): torch.randn(32, 32)}
    )
    lora_path = _save(tmp_path / "lora.safetensors", _lora_layer(RESIZED, 32, 32))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Recipe weights are normalized, so both recipes name their output alike.
    process_lora_model(
        PairedLoraModel(LoRADict(lora_path), BaseCheckpoint(base)),
        [ResizeRecipe("fro_ckpt=1,thr=-9"), ResizeRecipe("fro_ckpt=2,thr=-9")],
        out_dir,
        device="cpu",
    )

    assert len(list(out_dir.iterdir())) == 1
    assert "same name" in caplog.text


@pytest.mark.parametrize("recipe", ["spn_lora=1,thr=-0.3", "fro_lora=1,thr=-0.6"])
def test_rescale_cancels_in_scores_against_the_lora_itself(tmp_path, recipe):
    torch.manual_seed(0)
    base = _save(
        tmp_path / "ckpt.safetensors", {next(iter(SDXL_BASE)): torch.randn(32, 32)}
    )
    lora_path = _save(
        tmp_path / "lora.safetensors", _lora_layer(RESIZED, 32, 32, rank=8)
    )
    (tmp_path / "plain").mkdir()
    (tmp_path / "rescaled").mkdir()

    _, plain = _run(tmp_path / "plain", base, LoRADict(lora_path), recipe)
    _, rescaled = _run(
        tmp_path / "rescaled", base, LoRADict(lora_path), recipe + ",rescale=0.5"
    )

    rank = plain[f"{RESIZED}.lora_down.weight"].shape[0]
    assert 0 < rank < 8
    assert rescaled[f"{RESIZED}.lora_down.weight"].shape[0] == rank
    torch.testing.assert_close(
        _delta(rescaled, RESIZED), 0.5 * _delta(plain, RESIZED), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("merge", [ConcatLoRAsDict, CtMLoRAsDict])
def test_equal_merge_weights_are_a_rescale(tmp_path, merge):
    torch.manual_seed(0)
    base = _save(
        tmp_path / "ckpt.safetensors", {next(iter(SDXL_BASE)): torch.randn(32, 32)}
    )
    paths = [
        _save(tmp_path / f"{i}.safetensors", _lora_layer(RESIZED, 32, 32))
        for i in range(2)
    ]
    recipe = "spn_lora=1,thr=-0.5"
    (tmp_path / "weights").mkdir()
    (tmp_path / "rescale").mkdir()

    _, weighted = _run(
        tmp_path / "weights", base, merge([(p, 0.5) for p in paths]), recipe
    )
    _, rescaled = _run(
        tmp_path / "rescale",
        base,
        merge([(p, 1.0) for p in paths]),
        recipe + ",rescale=0.5",
    )

    rank = weighted[f"{RESIZED}.lora_down.weight"].shape[0]
    assert rank < 8
    assert rescaled[f"{RESIZED}.lora_down.weight"].shape[0] == rank
    torch.testing.assert_close(
        _delta(weighted, RESIZED), _delta(rescaled, RESIZED), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(
    "spec",
    ["mean", "sum,beta=0.5", "ctm,rank=4", "ctm,beta=2", "ctm,ties=0", "ctm,ties=x"],
)
def test_invalid_merge_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        MergeSpec(spec)


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
