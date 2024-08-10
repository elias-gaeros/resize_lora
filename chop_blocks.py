import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

from safetensors.numpy import safe_open, save_file

logger = logging.getLogger(__name__)


def analyze_lora_layers(
    sft_fd: safe_open,
) -> tuple[list[tuple[tuple[str, int], set[str]]], set[str]]:
    """
    Analyze the LoRA layers in a SafeTensors file.

    Args:
        sft_fd (safe_open): An open SafeTensors file.

    Returns:
        A tuple containing:
        - A list of tuples, each containing a (section, index) pair and a set of associated keys.
        - A set of pass-through keys (non-LoRA layers).
    """
    RE_LORA_NAME = re.compile(
        r"lora_unet_((?:input|middle|output|down|mid|up)_blocks?)(?:(?:_(\d+))?_attentions)?_(\d+)_.*"
    )

    pass_through_keys: set[str] = set()
    block2keys: dict[tuple[str, int], set[str]] = defaultdict(set)

    for k in sft_fd.keys():
        m = RE_LORA_NAME.fullmatch(k)
        if not m:
            pass_through_keys.add(k)
            continue
        section, idx1, idx2 = m.groups()
        if idx1 is None:
            idx = idx2
        else:
            idx = f"{idx1}{idx2}"

        block2keys[(section, idx)].add(k)

    if not block2keys:
        raise ValueError(
            "No UNet layers found in the LoRA checkpoint (Maybe not a SDXL model?)"
        )
    block2keys_sorted = sorted(block2keys.items())
    return block2keys_sorted, pass_through_keys


def print_block_layout(
    block2keys: list[tuple[tuple[str, int], set[str]]],
    weights: list[float] | None = None,
) -> None:
    """
    Print the layout of LoRA blocks, optionally with weights.

    Args:
        block2keys: A list of tuples, each containing a (section, index) pair and a set of associated keys.
        weights: Optional list of weights corresponding to each block.
    """
    logger.info("Blocks layout:")
    if weights is None:
        for i, ((section, idx), v) in enumerate(block2keys):
            logger.info(f"\t[{i:>2d}] {section:>13}.{idx} layers={len(v):<3}")
        section2shortname = {
            # SDXL names:
            "input_blocks": "INP",
            "middle_block": "MID",
            "output_blocks": "OUT",
            # SD1 names
            "down_blocks": "INP",
            "mid_block": "MID",
            "up_blocks": "OUT",
        }
        vector_string = ",".join(
            f"{section2shortname[section]}{idx:>02}" for (section, idx), _ in block2keys
        )
        logger.info(f'Vector string format: "1,{vector_string}"')
        vector_string = ",".join("0" * len(block2keys))
        logger.info(f'Example (drops all blocks): "1,{vector_string}"')
    else:
        for i, (((section, idx), v), weight) in enumerate(zip(block2keys, weights)):
            if abs(weight) > 1e-6:
                if abs(weight - 1) < 1e-6:
                    weight = 1
                logger.info(
                    f"\t[{i:>2d}] {section:>13}.{idx} layers={len(v):<3} weight={weight}"
                )
            else:
                logger.info(
                    f"\t[{i:>2d}] {section:>13}.{idx} layers={len(v):<3} (removed)"
                )


def filter_blocks(sft_fd: safe_open, vector_string: str) -> dict[str, "numpy.ndarray"]:
    """
    Filter LoRA blocks based on a vector string.

    Args:
        sft_fd (safe_open): An open SafeTensors file.
        vector_string (str): A string representing weights for each block.

    Returns:
        A dictionary containing the filtered state dict, or None if an error occurs.
    """
    global_weight, *weights_vector = map(float, vector_string.split(","))

    block2keys, pass_through_keys = analyze_lora_layers(sft_fd)
    if len(weights_vector) != len(block2keys):
        logger.error(f"expected {len(block2keys)} weights, got {len(weights_vector)}")
        print_block_layout(block2keys)
        return None

    if logger.getEffectiveLevel() >= logging.INFO:
        print_block_layout(block2keys, weights_vector)

    state_dict = {}
    for weight, (_, keys) in zip(weights_vector, block2keys):
        weight *= global_weight
        if abs(weight) < 1e-6:
            continue
        for k in keys:
            tensor = sft_fd.get_tensor(k)
            if abs(weight - 1.0) > 1e-6:
                tensor *= weight
            state_dict[k] = tensor

    logger.info(
        "Keeping %d keys from the UNet, %d passing through (text encoders)",
        len(state_dict),
        len(pass_through_keys),
    )
    for k in pass_through_keys:
        state_dict[k] = sft_fd.get_tensor(k)

    return state_dict


def setup_logging(verbosity: int) -> None:
    """
    Set up logging based on verbosity level and quiet flag.

    Args:
        verbosity (int): The verbosity level (0-2).
        quiet (bool): If True, suppress all output except errors.
    """
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[max(0, min(verbosity, 2))]
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")


def main() -> None:
    """
    Main function to handle CLI arguments and execute the appropriate actions.
    """
    parser = argparse.ArgumentParser(
        description="Analyze and filter LoRA layers in SafeTensors files."
    )
    parser.add_argument("input_file", type=Path, help="Input SafeTensors file")
    parser.add_argument(
        "vector_string", nargs="?", help="Vector string for filtering blocks"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output file path")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (can be repeated)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Suppress all output except errors",
    )
    args = parser.parse_args()
    setup_logging(args.verbose - args.quiet)

    with safe_open(args.input_file, framework="np") as sft_fd:
        if args.vector_string:
            # Filter blocks and save the result
            filtered_state_dict = filter_blocks(sft_fd, args.vector_string)
            if filtered_state_dict is None:
                exit(1)

            # Determine output path
            output_path = args.output or args.input_file.with_stem(
                f"{args.input_file.stem}-chop"
            )

            metadata = sft_fd.metadata()
            metadata["block_vector_string"] = args.vector_string
            save_file(filtered_state_dict, output_path, metadata=metadata)
            logging.info(f"Filtered LoRA saved to {output_path}")
        else:
            # Analyze LoRA layers
            block2keys, pass_through_keys = analyze_lora_layers(sft_fd)
            print_block_layout(block2keys)
            logging.info(f"Pass through layers: {len(pass_through_keys)}")


if __name__ == "__main__":
    main()
