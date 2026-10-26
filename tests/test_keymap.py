import json

from loralib import BaseCheckpoint
from loralib.keymap import Target, build_keymap, unet_to_diffusers

OPEN_CLIP = "conditioner.embedders.1.model."
IN_PROJ = OPEN_CLIP + "transformer.resblocks.0.attn.in_proj_weight"
SDXL_UNET = "model.diffusion_model."
SDXL = {
    "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight": (
        8,
        4,
    ),
    IN_PROJ: (12, 4),
    OPEN_CLIP + "transformer.resblocks.0.mlp.c_fc.weight": (16, 4),
    OPEN_CLIP + "text_projection": (4, 6),
    SDXL_UNET + "input_blocks.1.0.in_layers.2.weight": (4, 4, 3, 3),
    SDXL_UNET + "input_blocks.3.0.op.weight": (4, 4, 3, 3),
    SDXL_UNET + "input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight": (8, 8),
    SDXL_UNET + "middle_block.2.emb_layers.1.weight": (8, 16),
    SDXL_UNET + "output_blocks.2.1.conv.weight": (8, 8, 3, 3),
    "first_stage_model.decoder.conv_in.weight": (4, 4, 3, 3),
}


def target(shapes, key, *rows, transpose=False, file=0):
    start, stop = rows or (None, None)
    return Target(key, shapes[key], start, stop, transpose, file)


def test_sdxl_text_encoders_split_open_clip_attention():
    km = build_keymap(SDXL)

    clip_l = (
        "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight"
    )
    assert km["lora_te1_text_model_encoder_layers_0_mlp_fc1"] == target(SDXL, clip_l)
    q = km["lora_te2_text_model_encoder_layers_0_self_attn_q_proj"]
    assert q == target(SDXL, IN_PROJ, 0, 4)
    k = km["text_encoder_2.text_model.encoder.layers.0.self_attn.k_proj"]
    assert k == target(SDXL, IN_PROJ, 4, 8)
    fc1 = km["lora_te2_text_model_encoder_layers_0_mlp_fc1"]
    assert fc1.key.endswith("mlp.c_fc.weight")
    projection = km["lora_te2_text_projection"]
    assert projection == target(SDXL, OPEN_CLIP + "text_projection", transpose=True)
    assert projection.shape == (6, 4)
    assert not any(t.key.startswith("first_stage_model.") for t in km.values())


def test_sd_unet_ldm_and_diffusers_names():
    km = build_keymap(SDXL)

    for ldm, diffusers in [
        ("input_blocks.1.0.in_layers.2", "down_blocks.0.resnets.0.conv1"),
        ("input_blocks.3.0.op", "down_blocks.0.downsamplers.0.conv"),
        ("middle_block.2.emb_layers.1", "mid_block.resnets.1.time_emb_proj"),
        ("output_blocks.2.1.conv", "up_blocks.0.upsamplers.0.conv"),
    ]:
        assert unet_to_diffusers(ldm, 2) == diffusers
        expected = target(SDXL, f"{SDXL_UNET}{ldm}.weight")
        assert km["lora_unet_" + ldm.replace(".", "_")] == expected
        assert km["lora_unet_" + diffusers.replace(".", "_")] == expected
        assert km[f"unet.{diffusers}"] == expected

    attn = target(
        SDXL, SDXL_UNET + "input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight"
    )
    prefix = "down_blocks.1.attentions.0.transformer_blocks.0.attn1"
    assert km[f"{prefix}.to_out.0"] == attn
    assert km[f"unet.{prefix}.processor.to_out"] == attn


QKV = "double_blocks.0.img_attn.qkv.weight"
LINEAR1 = "single_blocks.0.linear1.weight"
FLUX = {
    QKV: (12, 4),
    "double_blocks.0.img_mod.lin.weight": (24, 4),
    LINEAR1: (28, 4),
    "single_blocks.0.linear2.weight": (4, 20),
}


def test_flux_diffusers_names_map_to_fused_rows():
    km = build_keymap(FLUX)

    assert km["diffusion_model.double_blocks.0.img_attn.qkv"] == target(FLUX, QKV)
    assert km["lora_unet_double_blocks_0_img_attn_qkv"] == target(FLUX, QKV)
    assert km["transformer.transformer_blocks.0.attn.to_k"] == target(FLUX, QKV, 4, 8)
    single = "transformer.single_transformer_blocks.0."
    assert km[single + "attn.to_v"] == target(FLUX, LINEAR1, 8, 12)
    assert km[single + "proj_mlp"] == target(FLUX, LINEAR1, 12, 28)
    assert km["diffusion_model.single_blocks.0.linear1_qkv"] == target(
        FLUX, LINEAR1, 0, 12
    )


def test_flux2_gets_no_flux1_diffusers_names():
    km = build_keymap({**FLUX, "double_stream_modulation_img.lin.weight": (24, 4)})

    assert "transformer.transformer_blocks.0.attn.to_k" not in km
    assert "diffusion_model.double_blocks.0.img_attn.qkv" in km


T5_Q = "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight"
SD3 = {
    "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight": (12, 4),
    "model.diffusion_model.joint_blocks.0.x_block.attn2.qkv.weight": (12, 4),
    "model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight": (12, 4),
    "text_encoders.clip_l.transformer.text_model.encoder.layers.0.mlp.fc1.weight": (
        8,
        4,
    ),
    "text_encoders.clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight": (
        8,
        4,
    ),
    T5_Q: (4, 4),
}


def test_sd3_diffusers_and_text_encoder_names():
    km = build_keymap(SD3)
    block = "model.diffusion_model.joint_blocks.0."

    attn2_v = km["transformer.transformer_blocks.0.attn2.to_v"]
    assert attn2_v == target(SD3, block + "x_block.attn2.qkv.weight", 8, 12)
    add_q = km["lora_transformer_transformer_blocks_0_attn_add_q_proj"]
    assert add_q == target(SD3, block + "context_block.attn.qkv.weight", 0, 4)
    fc1 = km["lora_te2_text_model_encoder_layers_0_mlp_fc1"]
    assert fc1.key.startswith("text_encoders.clip_g.")
    assert km["lora_te3_encoder_block_0_layer_0_SelfAttention_q"] == target(SD3, T5_Q)
    assert "lora_te2_encoder_block_0_layer_0_SelfAttention_q" not in km


def test_t5_of_clip_l_models_answers_to_te2_and_te3():
    shapes = {k: v for k, v in SD3.items() if ".clip_g." not in k}
    km = build_keymap(shapes)

    assert km["lora_te2_encoder_block_0_layer_0_SelfAttention_q"] == target(
        shapes, T5_Q
    )
    assert km["lora_te3_encoder_block_0_layer_0_SelfAttention_q"] == target(
        shapes, T5_Q
    )
    fc1 = km["lora_te1_text_model_encoder_layers_0_mlp_fc1"]
    assert fc1.key.startswith("text_encoders.clip_l.")


def _clip_file(width):
    return {"text_model.encoder.layers.0.mlp.fc1.weight": (4 * width, width)} | {
        f"text_model.encoder.layers.0.self_attn.{p}_proj.weight": (width, width)
        for p in "qkvo"
    }


def test_text_encoders_in_separate_files():
    clip_l, clip_g = _clip_file(768), _clip_file(1280)
    t5 = {"encoder.block.0.layer.0.SelfAttention.q.weight": (64, 64)}
    km = build_keymap(FLUX, clip_l, clip_g, t5)

    fc1 = "text_model.encoder.layers.0.mlp.fc1.weight"
    assert km["lora_te1_text_model_encoder_layers_0_mlp_fc1"] == target(
        clip_l, fc1, file=1
    )
    assert km[
        "text_encoders.clip_g.transformer.text_model.encoder.layers.0.mlp.fc1"
    ] == (target(clip_g, fc1, file=2))
    t5_q = target(t5, "encoder.block.0.layer.0.SelfAttention.q.weight", file=3)
    assert (
        km["text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q"]
        == t5_q
    )
    assert (
        km["text_encoders.transformer.encoder.block.0.layer.0.SelfAttention.q"] == t5_q
    )
    assert km["lora_te3_encoder_block_0_layer_0_SelfAttention_q"] == t5_q


def test_lone_qwen3_text_encoder_names():
    qwen = {
        "model.embed_tokens.weight": (100, 2560),
        "model.layers.0.self_attn.q_proj.weight": (4096, 2560),
    }
    km = build_keymap(FLUX, qwen)

    q = target(qwen, "model.layers.0.self_attn.q_proj.weight", file=1)
    assert km["text_encoders.qwen3_4b.transformer.model.layers.0.self_attn.q_proj"] == q
    assert km["text_encoders.transformer.model.layers.0.self_attn.q_proj"] == q
    assert km["lora_te_layers_0_self_attn_q_proj"] == q


def test_checkpoint_from_header_json_files(tmp_path):
    def header_file(name, shapes):
        path = tmp_path / name
        header = {k: {"dtype": "F16", "shape": list(v)} for k, v in shapes.items()}
        path.write_text(json.dumps(header))
        return path

    wan = {"blocks.0.self_attn.q.weight": (8, 8)}
    clip_l = _clip_file(768)
    checkpoint = BaseCheckpoint(
        f"{header_file('wan.json', wan)},{header_file('clip.json', clip_l)}"
    )

    q = target(wan, "blocks.0.self_attn.q.weight")
    assert checkpoint.target("diffusion_model.blocks.0.self_attn.q") == q
    assert checkpoint.target("lora_unet__blocks_0_self_attn_q") == q
    fc1 = checkpoint.target("lora_te_text_model_encoder_layers_0_mlp_fc1")
    assert fc1 == target(clip_l, "text_model.encoder.layers.0.mlp.fc1.weight", file=1)
