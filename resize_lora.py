from math import log10
import argparse
import logging
from pathlib import Path
import json

from tqdm import tqdm
import torch
import safetensors.torch

from loralib import (
    PairedLoraModel,
    BaseCheckpoint,
    JsonCache,
    DecomposedLoRA,
    LoRADict,
    ConcatLoRAsDict,
)

logger = logging.root


class ResizeRecipe:
    def __init__(self, recipe_str):
        self.recipe_str = recipe_str
        self.weights = parsed = {
            "spn_lora": 0.0,
            "spn_ckpt": 0.0,
            "subspace": 0.0,
            "fro_lora": 0.0,
            "fro_ckpt": 0.0,
            "params": 0.0,
        }
        self.target_size = None
        self.threshold = None
        self.rescale = 1.0

        for part in recipe_str.split(","):
            key, _, value = part.partition("=")
            if value:
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(
                        f"Could not parse {key}={value} in recipe {recipe_str}"
                    )
            if key in parsed:
                parsed[key] = 1.0 if value == "" else value
                continue
            if value == "":
                raise ValueError(
                    f"Empty value not accepted for key {key} in recipe {recipe_str}"
                )
            match key:
                case "size":
                    self.target_size = value
                case "thr":
                    self.threshold = value
                case "rescale":
                    self.rescale = value
                case _:
                    raise ValueError(f"Unknown key {key} in recipe {recipe_str}")

        wsum = sum(parsed.values())
        if wsum == 0.0:
            raise ValueError("At least one score type must be specified")
        self.weights = parsed = {k: v / wsum for k, v in parsed.items()}

        if self.target_size is None and self.threshold is None:
            raise ValueError("Either 'size' or 'thr' must be specified in the recipe")

    def __str__(self):
        return self.recipe_str

    def score_dims(self, decomposed_lora, checkpoint, **compute_kwargs):
        weights = self.weights
        layer_name = decomposed_lora.name
        S = decomposed_lora.S
        scores = torch.log10(S)
        if self.rescale is not None:
            scores += log10(self.rescale)
        if abs(weights["subspace"]) > 1e-6:
            W_base = checkpoint.get_weights(layer_name).to(**compute_kwargs)
            scores -= weights["subspace"] * torch.log10(
                decomposed_lora.compute_subspace_scales(W_base).abs().cpu()
            )
        if abs(weights["spn_ckpt"]) > 1e-6:
            scores -= weights["spn_ckpt"] * log10(
                checkpoint.spectral_norm(layer_name, **compute_kwargs)
            )
        if abs(weights["spn_lora"]) > 1e-6:
            scores -= weights["spn_lora"] * torch.log10(S[0])
        if abs(weights["fro_ckpt"]) > 1e-6:
            scores -= weights["fro_ckpt"] * log10(
                checkpoint.frobenius_norm(layer_name, dtype=torch.float32)
            )
        if abs(weights["fro_lora"]) > 1e-6:
            scores -= weights["fro_lora"] * torch.log10(torch.linalg.vector_norm(S))
        if abs(weights["params"]) > 1e-6:
            scores -= weights["params"] * log10(decomposed_lora.dim_size(1))
        return scores

    def resize_lora(
        self,
        lora_layers: list[DecomposedLoRA],
        checkpoint,
        compute_kwargs=dict(dtype=torch.float32),
        output_dtype=torch.float16,
        output_elem_size=None,
    ):
        print_scores = logger.isEnabledFor(logging.INFO)
        print_layers = logger.isEnabledFor(logging.DEBUG)
        needs_flat_scores = print_scores or self.target_size is not None
        if output_elem_size is None:
            if output_dtype == torch.float32:
                output_elem_size = 4
            elif output_dtype == torch.float16:
                output_elem_size = 2
            else:
                # Only works for torch>=2.1
                output_elem_size = output_dtype.itemsize

        # Score all fims
        scores = [
            self.score_dims(decomposed_lora, checkpoint, **compute_kwargs)
            for decomposed_lora in lora_layers
        ]

        if needs_flat_scores:
            flat_scores = torch.cat(scores)
        # Select a threshold (greedy knapsack)
        if self.target_size is not None:
            flat_scores, order = flat_scores.sort(descending=True)
            cum_sizes = torch.repeat_interleave(
                *torch.tensor(
                    [
                        (layer.dim_size(output_elem_size), layer.dim)
                        for layer in lora_layers
                    ],
                    dtype=torch.int32,
                ).T
            )[order].cumsum(0)
            target_size_bytes = self.target_size * (1 << 20)
            if target_size_bytes < cum_sizes[-1]:
                threshold = flat_scores[
                    torch.searchsorted(cum_sizes, target_size_bytes).item()
                ].item()
            else:
                threshold = -torch.inf
            logger.info("Selected threshold: %.3f", threshold)
        else:
            threshold = self.threshold

        sd = {}
        for decomposed_lora, layer_scores in zip(lora_layers, scores):
            mask = layer_scores > threshold
            sd.update(
                decomposed_lora.statedict(
                    mask=mask, dtype=output_dtype, rescale=self.rescale
                )
            )
            if print_layers:
                S = decomposed_lora.S * self.rescale
                err = 0.0 if torch.all(mask) else torch.linalg.vector_norm(S[~mask])
                re_lora = err / torch.linalg.vector_norm(S)
                re_ckpt = err / checkpoint.frobenius_norm(
                    decomposed_lora.name, dtype=torch.float32
                )
                logger.debug(
                    f"dim:{S.shape[0]:>3}->{mask.sum().item():<3}"
                    f" rle_lora:{100. * re_lora:>6.2f}% rle_ckpt:{100. * re_ckpt:>6.2f}%"
                    f" {decomposed_lora.name}"
                )

        if print_scores:
            quantile_fracs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
            quantile_string = " ".join(
                f"{frac:.2f}:{q:.3f}"
                for frac, q in zip(
                    quantile_fracs, flat_scores.quantile(torch.tensor(quantile_fracs))
                )
            )
            logger.info(f"Score quantiles: {quantile_string}")

        return sd, threshold


def process_lora_model(
    lora_model: PairedLoraModel,
    recipes: list[ResizeRecipe],
    output_folder,
    device=None,
    compute_dtype=torch.float32,
    output_dtype=torch.float16,
):
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
        sd, threshold = recipe.resize_lora(
            tqdm(lora_layers, desc=f"Scoring {recipe}"),
            checkpoint,
            compute_kwargs=compute_kwargs,
            output_dtype=output_dtype,
        )

        params = recipe.__dict__.copy()
        params["threshold"] = threshold
        metadata = lora_model.lora_dict.metadata()
        metadata["resize_params"] = json.dumps(params)

        recipe_fn = [
            f"{k.replace('_', '')}{format_float(v)}"
            for k, v in sorted(recipe.weights.items())
            if v != 0.0
        ]
        if recipe.rescale != 1.0:
            recipe_fn.append(f"scale{format_float(recipe.rescale)}")
        if recipe.target_size is not None:
            recipe_fn.append(f"size{format_float(recipe.target_size)}")
        recipe_fn = "_".join(recipe_fn)
        output_path = output_folder / (
            f"{lora_model.lora_dict.name}_{recipe_fn}_th{format_float(threshold)}.safetensors"
        )

        logger.info("Saving %s", output_path)
        safetensors.torch.save_file(
            sd,
            output_path,
            metadata=metadata,
        )


def load_lora_or_merge(path, **to_kwargs):
    if "," in path:
        members = []
        for path in path.split(","):
            path, _, weight = path.partition(":")
            weight = float(weight) if weight else 1.0
            members.append((path, weight))
        return ConcatLoRAsDict(members, **to_kwargs)
    else:
        return LoRADict(path, **to_kwargs)


def format_float(v, p=2):
    return f"{v:.{p}f}".rstrip("0").rstrip(".")


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
        type=str,
        default=None,
        help="Folder to save the output files, use the same folder as the input lora if not specified",
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
        default="fro_ckpt=1,thr=-3.5",
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

    output_dtype = torch.float16 if args.output_dtype == "16" else torch.float32
    score_recipes = [ResizeRecipe(recipe) for recipe in args.score_recipes.split(":")]

    norms_cache = JsonCache(Path(__file__).parent / "norms_cache.json")
    checkpoint = BaseCheckpoint(args.checkpoint_path, cache=norms_cache)

    for lora_model_path in args.lora_model_paths:
        logger.info(f"Processing LoRA model: {lora_model_path}")
        lora_dict = load_lora_or_merge(
            lora_model_path, device=args.device, dtype=torch.float32
        )
        output_folder = lora_dict.path.parent
        if args.output_folder:
            output_folder = Path(args.output_folder)
        paired = PairedLoraModel(lora_dict, checkpoint)
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
