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
    Try multiple LoRA key mapping formats.
    Returns:
        None: If no LoRA key mapping is found.
        Tuple (is_split: bool, names: Union[str, List[str]]):
            is_split (bool): True if 'names' are parts of a genuinely split base tensor (e.g., QKV).
                             False if 'names' is a single name or list of aliases for an unsplit tensor.
            names (Union[str, List[str]]): The LoRA key name(s).
    """
    # standard_keys_info can be a list (if base_key is split like QKV), a string, or None.
    standard_keys_info = get_sdxl_lora_keys(base_key)

    # Case 1: get_sdxl_lora_keys indicates a genuine split into parts
    if isinstance(standard_keys_info, list):
        # standard_keys_info is already the list of actual parts (e.g., for QKV).
        # For such layers, we *only* return these part names.
        # Aliases for the "whole" layer from get_alt_lora_keys or direct_format are not applicable here.
        return True, standard_keys_info  # (is_split=True, list_of_part_names)

    # Case 2: Layer is NOT indicated as split by get_sdxl_lora_keys.
    # standard_keys_info is a string (single LoRA name for the whole layer) or None.
    # We will now collect all *alternative names* (aliases) for this *single, unsplit* layer.

    aliases_for_unsplit_layer = []

    # Add the standard name if it exists (it will be a string here)
    if standard_keys_info is not None:  # This would be a string
        aliases_for_unsplit_layer.append(standard_keys_info)

    # get_alt_lora_keys should return a string or None.
    alt_key_single = get_alt_lora_keys(base_key)
    if alt_key_single is not None and alt_key_single not in aliases_for_unsplit_layer:
        aliases_for_unsplit_layer.append(alt_key_single)

    direct_format_single = None
    if base_key.startswith(UNET_PREFIX):
        direct_format_single = (
            base_key.removeprefix(UNET_PREFIX).removesuffix(".weight").replace(".", "_")
        )
    if (
        direct_format_single is not None
        and direct_format_single not in aliases_for_unsplit_layer
    ):
        aliases_for_unsplit_layer.append(direct_format_single)

    if not aliases_for_unsplit_layer:
        return None  # No known LoRA name (standard, alt, or direct) for this base_key

    if len(aliases_for_unsplit_layer) == 1:
        # Only one name found, return it as a string
        return (
            False,
            aliases_for_unsplit_layer[0],
        )  # (is_split=False, single_name_string)
    else:
        # Multiple alias names found for the same unsplit layer
        return (
            False,
            aliases_for_unsplit_layer,
        )  # (is_split=False, list_of_alias_strings)
