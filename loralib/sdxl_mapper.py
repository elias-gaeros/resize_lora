UNET_PREFIX = "model.diffusion_model."
TE_PREFIXES = [
    "conditioner.embedders.0.transformer.text_model.encoder.layers.",
    "conditioner.embedders.1.model.transformer.resblocks.",
]
LORA_UNET_PREFIX = "lora_unet_"
LORA_TE_PREFIXES = [
    "lora_te1_text_model_encoder_layers_",
    "lora_te2_text_model_encoder_layers_",
]


def get_sdxl_lora_keys(base_key):
    layer_name = base_key.removesuffix(".weight")

    if layer_name.startswith(UNET_PREFIX):
        return LORA_UNET_PREFIX + layer_name.removeprefix(UNET_PREFIX).replace(".", "_")
    lora_keys = None
    for te_prefix, lora_te_prefix in zip(TE_PREFIXES, LORA_TE_PREFIXES):
        if layer_name.startswith(te_prefix):
            assert lora_keys is None
            layer_name = lora_te_prefix + layer_name.removeprefix(te_prefix).replace(
                ".", "_"
            )

            if te_prefix is TE_PREFIXES[0]:
                lora_keys = layer_name  # CLIP L is easy
            else:  # CLIP G
                if "attn_in_proj_weight" in layer_name:
                    lora_keys = layer_name.replace("_attn_in_proj_weight", "_self_attn")
                    lora_keys = [
                        f"{lora_keys}_{chunk_name}_proj" for chunk_name in "kqv"
                    ]
                elif layer_name.endswith("_attn_out_proj"):
                    lora_keys = layer_name.replace(
                        "_attn_out_proj", "_self_attn_out_proj"
                    )
                elif "_ln_" in layer_name:
                    lora_keys = layer_name.replace("_ln_", "_layer_norm")
                elif "_mlp_" in layer_name:
                    lora_keys = layer_name.replace("_c_fc", "_fc1").replace(
                        "_c_proj", "_fc2"
                    )
    return lora_keys
