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

# Alternative prefixes and formatters for different LoRA trainers
ALT_LORA_UNET_PREFIX = "down_blocks_"  # Common alternative format


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


def get_alt_lora_keys(base_key):
    """
    Alternative mapping for LoRA keys used by some trainers.
    Maps base model weights to alternative key formats used in some LoRA files.
    """
    layer_name = base_key.removesuffix(".weight")

    if layer_name.startswith(UNET_PREFIX):
        # Remove the standard prefix
        unet_part = layer_name.removeprefix(UNET_PREFIX)

        # Format 1: Blocks with direct mapping (e.g. "down_1" for input_blocks.1)
        if "input_blocks" in unet_part:
            parts = unet_part.split(".")
            try:
                block_num = int(parts[1])
                # Format: down_N
                alt_name = f"down_{block_num}"

                # Add layer details if present
                if len(parts) > 3:
                    # Try different possible formats
                    layer_details = f"{parts[2]}_{parts[3]}"
                    if "in_layers" in layer_details:
                        alt_name += "_in"
                    elif "out_layers" in layer_details:
                        alt_name += "_out"
                    elif "emb_layers" in layer_details:
                        alt_name += "_emb"

                return alt_name
            except (IndexError, ValueError):
                pass

        # Format 2: Middle blocks
        if "middle_block" in unet_part:
            parts = unet_part.split(".")
            try:
                block_num = int(parts[1])
                # Format: mid_N
                alt_name = f"mid_{block_num}"

                # Add layer details if present
                if len(parts) > 3:
                    # Try different possible formats
                    layer_details = f"{parts[2]}_{parts[3]}"
                    if "in_layers" in layer_details:
                        alt_name += "_in"
                    elif "out_layers" in layer_details:
                        alt_name += "_out"
                    elif "emb_layers" in layer_details:
                        alt_name += "_emb"

                return alt_name
            except (IndexError, ValueError):
                pass

        # Format 3: Output blocks
        if "output_blocks" in unet_part:
            parts = unet_part.split(".")
            try:
                block_num = int(parts[1])
                # Format: up_N
                alt_name = f"up_{block_num}"

                # Add layer details if present
                if len(parts) > 3:
                    # Try different possible formats
                    layer_details = f"{parts[2]}_{parts[3]}"
                    if "in_layers" in layer_details:
                        alt_name += "_in"
                    elif "out_layers" in layer_details:
                        alt_name += "_out"
                    elif "emb_layers" in layer_details:
                        alt_name += "_emb"
                    elif "skip_connection" in layer_details:
                        alt_name += "_skip"

                return alt_name
            except (IndexError, ValueError):
                pass

        # Format 4: Simple layer names with numbers
        simpler_name = None
        if "input_blocks" in unet_part:
            parts = unet_part.split(".")
            try:
                simpler_name = f"input_blocks_{parts[1]}"
                if len(parts) > 3:
                    simpler_name += f"_{parts[2]}_{parts[3]}"
            except (IndexError, ValueError):
                pass
        elif "middle_block" in unet_part:
            parts = unet_part.split(".")
            try:
                simpler_name = f"middle_block_{parts[1]}"
                if len(parts) > 3:
                    simpler_name += f"_{parts[2]}_{parts[3]}"
            except (IndexError, ValueError):
                pass
        elif "output_blocks" in unet_part:
            parts = unet_part.split(".")
            try:
                simpler_name = f"output_blocks_{parts[1]}"
                if len(parts) > 3:
                    simpler_name += f"_{parts[2]}_{parts[3]}"
            except (IndexError, ValueError):
                pass

        if simpler_name:
            return simpler_name

    # For CLIP text encoder layers, try some common alternative formats
    for te_prefix, lora_te_prefix in zip(TE_PREFIXES, LORA_TE_PREFIXES):
        if layer_name.startswith(te_prefix):
            # Alternative format for text encoder: text_model_encoder_N
            parts = layer_name.removeprefix(te_prefix).split(".")
            try:
                layer_num = int(parts[0])
                alt_name = f"text_model_encoder_{layer_num}"
                if len(parts) > 1:
                    alt_name += f"_{parts[1]}"
                return alt_name
            except (IndexError, ValueError):
                pass

    # Try standard mapping if alternative doesn't match
    return None  # Return None to avoid duplicating the standard keys


def get_multi_format_lora_keys(base_key):
    """
    Try multiple LoRA key mapping formats and return all possible key names.
    This allows checking multiple naming conventions when loading LoRAs.
    """
    standard_keys = get_sdxl_lora_keys(base_key)
    # alt_keys and direct_format are calculated *before* checking if standard_keys is a list
    alt_keys = get_alt_lora_keys(base_key)
    direct_format = None
    if base_key.startswith(UNET_PREFIX):
        direct_format = (
            base_key.removeprefix(UNET_PREFIX).removesuffix(".weight").replace(".", "_")
        )

    # If standard_keys is already a list of parts, it's authoritative for split layers.
    # Do not append alt_keys/direct_format meant for the whole layer to this list of parts.
    if isinstance(standard_keys, list):
        return standard_keys  # Return the list of parts directly

    # standard_keys is a string or None here (non-split layer according to get_sdxl_lora_keys)
    if standard_keys is None and alt_keys is None and direct_format is None:
        return None

    result = []
    if standard_keys is not None:
        result.append(standard_keys)
    if alt_keys is not None and alt_keys not in result:  # Check for duplicates
        result.append(alt_keys)
    if (
        direct_format is not None and direct_format not in result
    ):  # Check for duplicates
        result.append(direct_format)

    if len(result) == 0:  # Should be caught by the None check above, but for safety
        return None
    if len(result) == 1:
        return result[0]

    return result  # List of alternative names for a single, non-split layer
