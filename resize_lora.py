from math import log10
import argparse
import logging
from pathlib import Path
import json

from tqdm import tqdm
import torch
import safetensors.torch

from loralib import PairedLoraModel, BaseCheckpoint, JsonCache

logger = logging.root


def parse_score_recipe(recipe):
    weights = {
        "spn_lora": 0.0,
        "spn_ckpt": 0.0,
        "subspace": 0.0,
        "fro_lora": 0.0,
        "fro_ckpt": 0.0,
    }
    size = None
    threshold = None

    for part in recipe.split(","):
        key, _, value = part.partition("=")
        if key in weights:
            weights[key] = float(value)
        elif key == "size":
            size = float(value)
        elif key == "thr":
            threshold = float(value)
        else:
            raise ValueError("Unknown key in recipe: %s" % key)

    if size is None and threshold is None:
        raise ValueError("Either 'size' or 'thr' must be specified in the recipe")

    wsum = sum(weights.values())
    if wsum == 0.0:
        raise ValueError("At least one score type must be specified")
    weights = {k: v / wsum for k, v in weights.items()}

    return weights, size, threshold


def process_lora_model(
    lora_model: PairedLoraModel,
    recipes,
    output_folder,
    device=None,
    compute_dtype=torch.float32,
    output_dtype=torch.float16,
):
    print_scores = logger.isEnabledFor(logging.INFO)
    print_layers = logger.isEnabledFor(logging.DEBUG)

    compute_kwargs = dict(dtype=compute_dtype, device=device, non_blocking=True)
    checkpoint = lora_model.checkpoint

    lora_layers = []
    for key in tqdm(lora_model.keys(), desc="SVD"):
        decomposed_lora = lora_model.decompose_layer(key, **compute_kwargs).to(
            device="cpu"
        )
        if decomposed_lora.S[0].abs().item() < 1e-6:
            logger.warning(
                "LoRA layer %s is all zeroes! dim=%d",
                decomposed_lora.name,
                decomposed_lora.S.shape[0],
            )
            continue
        lora_layers.append(decomposed_lora)

    for recipe in recipes:
        score_weights, target_size, threshold = parse_score_recipe(recipe)
        needs_flat_scores = print_scores or target_size is not None

        score_layers = {}
        all_scores = []
        all_sizes = []
        for decomposed_lora in tqdm(lora_layers, desc=f"Scoring {recipe}"):
            layer_name = decomposed_lora.name
            S = decomposed_lora.S
            scores = torch.log10(S)
            if abs(score_weights["subspace"]) > 1e-6:
                W_base = checkpoint.get_weights(layer_name).to(**compute_kwargs)
                scores -= score_weights["subspace"] * torch.log10(
                    decomposed_lora.compute_subspace_scales(W_base).abs().cpu()
                )
            if abs(score_weights["spn_ckpt"]) > 1e-6:
                scores -= score_weights["spn_ckpt"] * log10(
                    checkpoint.spectral_norm(layer_name, **compute_kwargs)
                )
            if abs(score_weights["spn_lora"]) > 1e-6:
                scores -= score_weights["spn_lora"] * torch.log10(S[0])
            if abs(score_weights["fro_ckpt"]) > 1e-6:
                scores -= score_weights["fro_ckpt"] * log10(
                    checkpoint.frobenius_norm(layer_name, dtype=torch.float32)
                )
            if abs(score_weights["fro_lora"]) > 1e-6:
                scores -= score_weights["fro_lora"] * torch.log10(
                    torch.linalg.vector_norm(S)
                )
            score_layers[layer_name] = scores

            if needs_flat_scores:
                all_scores.append(scores)
            if target_size is not None:
                dim_size = decomposed_lora.dim_size(output_dtype.itemsize)
                all_sizes.append(
                    torch.tile(
                        torch.scalar_tensor(dim_size, dtype=torch.int32),
                        (decomposed_lora.dim,),
                    )
                )

        if needs_flat_scores:
            all_scores = torch.cat(all_scores)
        if target_size is not None:
            all_scores, order = all_scores.sort(descending=True)
            cum_sizes = torch.cat(all_sizes)[order].cumsum(0)
            target_size_bytes = target_size * (1 << 20)
            if target_size_bytes < cum_sizes[-1]:
                threshold = all_scores[
                    torch.searchsorted(cum_sizes, target_size_bytes).item()
                ]
            else:
                threshold = -torch.inf
            logger.info("Selected threshold: %.3f", threshold)

        sd = {}
        for decomposed_lora in lora_layers:
            mask = score_layers[decomposed_lora.name] > threshold
            sd.update(decomposed_lora.statedict(mask=mask, dtype=output_dtype))
            if print_layers:
                S = decomposed_lora.S
                err = 0.0 if torch.all(mask) else torch.linalg.vector_norm(S[~mask])
                re_lora = err / torch.linalg.vector_norm(S)
                re_ckpt = err / checkpoint.frobenius_norm(
                    decomposed_lora.name, dtype=torch.float32
                )
                logger.debug(
                    f"dim:{S.shape[0]:>3} :{mask.sum().item():<3}"
                    f" rle_lora:{100. * re_lora:>6.2f}% rle_ckpt:{100. * re_ckpt:>6.2f}%"
                    f" {decomposed_lora.name}"
                )

        if print_scores:
            quantile_fracs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
            quantile_string = " ".join(
                f"{frac:.2f}:{q:.3f}"
                for frac, q in zip(
                    quantile_fracs, all_scores.quantile(torch.tensor(quantile_fracs))
                )
            )
            logger.info(f"Score quantiles: {quantile_string}")

        metadata = lora_model.lora_fd.metadata()
        metadata = lora_model.lora_fd.metadata()
        metadata["resize_threshold"] = f"{threshold:.3f}"

        recipe_fn = [f"{k}{v:.2f}" for k, v in score_weights.items() if v != 0]
        if target_size is not None:
            recipe_fn.append(f"size{target_size:.1f}")
            metadata["resize_size"] = f"{target_size:.1f}"
        metadata["resize_weights"] = json.dumps(score_weights)
        recipe_fn = "_".join(recipe_fn)
        output_path = output_folder / (
            f"{lora_model.lora_path.stem}_th{threshold:.2f}_{recipe_fn}.safetensors"
        )

        logger.info("Saving %s", output_path)
        safetensors.torch.save_file(
            sd,
            output_path,
            metadata=lora_model.lora_fd.metadata(),
        )


def main():
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "-t",
        "--output_dtype",
        type=str,
        choices=["16", "32"],
        default="16",
        help="Output dtype: 16 for float16, 32 for float32",
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
        default="spn_ckpt=1,thr=-1.2",
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
    output_dtype = torch.float16 if args.output_dtype == "16" else torch.float32
    score_recipes = args.score_recipes.split(":")

    norms_cache = JsonCache(Path(__file__).parent / "norms_cache.json")
    checkpoint = BaseCheckpoint(args.checkpoint_path, cache=norms_cache)

    for lora_model_path in args.lora_model_paths:
        logger.info(f"Processing LoRA model: {lora_model_path}")
        paired = PairedLoraModel(lora_model_path, checkpoint)
        process_lora_model(
            lora_model=paired,
            recipes=score_recipes,
            output_folder=output_folder,
            output_dtype=output_dtype,
            device=args.device,
        )

    norms_cache.save(discard=True)


if __name__ == "__main__":
    main()
