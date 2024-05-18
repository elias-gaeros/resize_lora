import argparse
import logging
from pathlib import Path

from tqdm import tqdm
import torch as pt
import safetensors.torch

from loralib import PairedLoraModel, BaseCheckpoint, JsonCache


def parse_score_recipe(recipe):
    weights = {"spn_lora": 0.0, "subspace": 0.0, "spn_ckpt": 0.0}
    size = None
    threshold = None

    for part in recipe.split(","):
        key, _, value = part.partition("=")
        if key in weights:
            weights[key] = float(value)
        elif key == "size":
            size = int(value)
        elif key == "thr":
            threshold = float(value)
        else:
            raise ValueError("Unknown key in recipe: %s" % key)

    if size is None and threshold is None:
        raise ValueError("Either 'size' or 'thr' must be specified in the recipe")

    return weights, size, threshold


def process_lora_model(
    paired: PairedLoraModel,
    recipes,
    output_folder,
    device=None,
    compute_dtype=pt.float32,
    output_dtype=pt.float16,
):
    print_scores = logging.root.isEnabledFor(logging.INFO)

    compute_kwargs = dict(dtype=compute_dtype, device=device, non_blocking=True)
    checkpoint = paired.checkpoint

    dlora_layers = []
    for key in tqdm(paired.keys(), desc="SVD"):
        dlora = paired.decompose_layer(key, **compute_kwargs).to(device="cpu")
        if dlora.S[0].abs().item() < 1e-6:
            logging.warning(
                "LoRA layer %s is all zeroes! dim=%d", dlora.name, dlora.S.shape[0]
            )
            continue
        dlora_layers.append(dlora)

    for recipe in recipes:
        score_weights, target_size, threshold = parse_score_recipe(recipe)
        needs_flat_sizes = target_size is not None
        needs_flat_scores = print_scores or needs_flat_sizes

        score_layers = {}
        all_scores = []
        all_sizes = []
        for dlora in tqdm(dlora_layers, desc=f"Scoring {recipe}"):
            W_base = None
            scores = pt.zeros_like(dlora.S)
            if abs(score_weights["subspace"]) > 1e-6:
                W_base = checkpoint.get_weights(dlora.name).to(**compute_kwargs)
                scores += score_weights["subspace"] * pt.log10(
                    dlora.S / dlora.compute_subspace_scales(W_base).abs().cpu()
                )
            if abs(score_weights["spn_ckpt"]) > 1e-6:
                scores += score_weights["spn_ckpt"] * pt.log10(
                    dlora.S
                    / checkpoint.spectral_norm(
                        dlora.name, weights=W_base, **compute_kwargs
                    )
                )
            if abs(score_weights["spn_lora"]) > 1e-6:
                scores += score_weights["spn_lora"] * pt.log10(dlora.S / dlora.S[0])
            score_layers[dlora.name] = scores

            if needs_flat_scores:
                all_scores.append(scores)
            if needs_flat_sizes:
                dim_size = dlora.dim_size(output_dtype.itemsize)
                all_sizes.append(
                    pt.tile(pt.scalar_tensor(dim_size, dtype=pt.int32), (dlora.dim,))
                )

        if needs_flat_scores:
            all_scores = pt.cat(all_scores)
        if needs_flat_sizes:
            all_scores, order = all_scores.sort(descending=True)
            all_sizes = pt.cat(all_sizes)
            all_sizes = all_sizes[order]
            threshold = all_scores[
                pt.searchsorted(all_sizes.cumsum(0), target_size << 20).item()
            ]
            logging.info("Selected threshold: %.3f", threshold)

        if print_scores:
            quantile_fracs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
            quantile_string = " ".join(
                f"{frac:.2f}:{q:.3f}"
                for frac, q in zip(
                    quantile_fracs, all_scores.quantile(pt.tensor(quantile_fracs))
                )
            )
            logging.info(f"Score quantiles: {quantile_string}")

        sd = {}
        for dlora in dlora_layers:
            mask = score_layers[dlora.name] > threshold
            sd.update(dlora.statedict(mask=mask, dtype=output_dtype))
            logging.debug(
                "dim: %s->%s\t%s", mask.shape[0], mask.sum().item(), dlora.name
            )

        recipe_fn = recipe.replace(",", "_").replace("=", "")
        output_path = output_folder / (
            f"{paired.lora_path.stem}_th{-threshold:.2f}_{recipe_fn}.safetensors"
        )
        metadata = paired.lora_fd.metadata()
        metadata["resize_threshold"] = f"{threshold:.3f}"
        metadata["resize_recipe"] = recipe
        logging.info("Saving %s", output_path)
        safetensors.torch.save_file(
            sd,
            output_path,
            metadata=paired.lora_fd.metadata(),
        )


def main():
    compute_device = "cuda" if pt.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Resizes multiple LoRAs with specified parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    parser.add_argument(
        "lora_model_paths",
        type=str,
        nargs="+",
        help="Paths to the Lora model files",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
        type=str,
        help="Folder to save the output files",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=compute_device,
        help="Device to run the computations on",
    )
    parser.add_argument(
        "-r",
        "--score_recipes",
        type=str,
        default="spn_ckpt=1,thr=-1",
        help="Score recipes separated by ':' in the format spn_ckpt=X,spn_lora=Y,subspace=Z,size=S:spn_ckpt=...",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (e.g., -v for INFO, -vv for DEBUG)",
    )
    args = parser.parse_args()

    log_level = logging.WARNING - (10 * args.verbose)
    logging.basicConfig(level=log_level)
    output_folder = Path(args.output_folder)
    score_recipes = args.score_recipes.split(":")

    spectral_norms_cache = JsonCache(
        Path(__file__).parent / "spectral_norms_cache.json"
    )
    checkpoint = BaseCheckpoint(args.checkpoint_path, cache=spectral_norms_cache)

    for lora_model_path in args.lora_model_paths:
        logging.info(f"Processing LoRA model: {lora_model_path}")
        paired = PairedLoraModel(lora_model_path, checkpoint)
        process_lora_model(
            paired=paired,
            recipes=score_recipes,
            output_folder=output_folder,
            device=args.device,
        )


if __name__ == "__main__":
    main()
