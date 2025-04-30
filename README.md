# LoRA Resize Tool

This tool compresses SDXL LoRA models by selectively reducing their rank (the number of internal dimensions). It goes beyond simple truncation by using methods to score the importance of each dimension, comparing the LoRA's characteristics against the original base model checkpoint. This context-aware approach helps preserve the LoRA's essential effects while significantly reducing file size.

This tool is primarily designed for **SDXL LoRA/LoCon models**. Support for other types (SD1.5, DoRA, LoKr, LoHa, etc.) is not currently included.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Command](#basic-command)
  - [Arguments](#arguments)
  - [Advanced Examples](#advanced-examples)
- [Understanding Recipes](#understanding-recipes)
  - [The Goal: Scoring and Selecting Dimensions](#the-goal-scoring-and-selecting-dimensions)
  - [Selection Methods: `thr` vs. `size`](#selection-methods-thr-vs-size)
  - [Scoring Methods: Defining "Importance"](#scoring-methods-defining-importance)
    - [Weighting the Comparisons](#weighting-the-comparisons)
    - [How Comparisons are Combined](#how-comparisons-are-combined)
    - [Available Methods](#available-methods)
  - [The `rescale` Factor](#the-rescale-factor)
- [Output Filename Convention](#output-filename-convention)
- [Understanding Verbose Output (`-vv`)](#understanding-verbose-output--vv)
- [Technical Details: Spectral vs. Frobenius Norm](#technical-details-spectral-vs-frobenius-norm)
- [Why is it so fast?](#why-is-it-so-fast)
- [Evaluation](#evaluation)
- [License](#license)
- [Contributing](#contributing)

## Features

- **SDXL Focused**: Optimized for SDXL LoRA and LoCon models.
- **Base Model Aware**: Uses weights from the base model checkpoint for more robust dimension scoring.
- **Flexible Scoring**: Combine multiple metrics (comparing LoRA to itself, to the base model, considering parameter efficiency) using weighted recipes.
- **Target Size or Threshold**: Prune dimensions based on a specific target file size (`size=`) or a direct importance score threshold (`thr=`).
- **Batch Processing**: Process multiple LoRA files in a single command.
- **Multiple Recipes**: Apply several different compression recipes to the same LoRA simultaneously, facilitating experimentation without redundant SVD calculations.
- **Optimized for Speed**: Uses efficient SVD techniques and caches base model norm calculations to disk (`norms_cache.json`) for faster subsequent runs.

## Installation

Clone the repository:

```sh
git clone https://github.com/elias-gaeros/resize_lora.git
cd resize_lora
```

### Requirements

- Python 3.8+
- `torch`
- `safetensors`
- `tqdm`

Install required packages (preferably in a virtual environment):

```sh
pip install torch safetensors tqdm
```

## Usage

### Basic Command

```sh
python resize_lora.py /path/to/sdxl_base_v1.0.safetensors /path/to/my_lora.safetensors -o /path/to/output/folder
```

This command uses the default recipe (`fro_ckpt=1,thr=-3.5`). It compresses the LoRA by keeping only the dimensions whose "strength" (singular value) is greater than roughly 1/16th ($10^{-1.2}$) of the "maximum strength" (spectral norm) of the corresponding layer in the base SDXL model.

### Arguments

- `checkpoint_path` (positional): Path to the base model checkpoint file (e.g., SDXL 1.0 base).
- `lora_model_paths` (positional): Path(s) to the LoRA model files.
  - You can specify multiple files: `lora1.safetensors lora2.safetensors`.
  - You can use wildcards (shell expanded): `loras/*.safetensors`.
  - You can specify a weighted merge: `"lora1:0.7,lora2:0.3"` (use quotes if needed by your shell). The tool will merge these LoRAs _before_ resizing.
- `-o`, `--output_folder` (required): Folder where the resized LoRA files will be saved.
- `-t`, `--output_dtype`: Output precision (`16` for float16, `32` for float32). Default: `16`.
- `-d`, `--device`: Device for computations (`cuda`, `cpu`, etc.). Default: `cuda` if available, otherwise `cpu`.
- `-r`, `--score_recipes`: Defines how to score and prune dimensions. Separate multiple recipes with colons (`:`). Default: `fro_ckpt=1,thr=-3.5`. See [Understanding Recipes](#understanding-recipes) below.
- `-v`, `--verbose`: Increase output detail.
  - `-v`: INFO level (shows progress, final thresholds, quantiles).
  - `-vv`: DEBUG level (shows detailed per-layer resizing info).

### Advanced Examples

1.  **Process multiple LoRAs with the default recipe:**

    ```sh
    python resize_lora.py sdxl_base.safetensors loras/*.safetensors -o resized_loras
    ```

2.  **Apply multiple recipes to a single LoRA:**

    ```sh
    python resize_lora.py sdxl_base.safetensors my_lora.safetensors -o experimental_resizes \
        -r "fro_ckpt=1,thr=-3.5:spn_lora=1,thr=-0.7:spn_ckpt=1,size=32"
    ```

    This generates three versions of `my_lora.safetensors`:

    - One using the default base-model comparison (`spn_ckpt`) with a threshold of $`10^{-1.2}`$.
    - One using a self-comparison (`spn_lora`) similar to Kohya's `sv_ratio`, keeping dimensions stronger than $`10^{-0.7} \approx 1/5`$th of the LoRA layer's own maximum.
    - One aiming for a 32 MiB file size, using the `spn_ckpt` scoring method to decide which dimensions offer the best "value" per parameter.

3.  **Merge two LoRAs then resize the result using a custom recipe and verbose output:**
    ```sh
    python resize_lora.py sdxl_base.safetensors "style_lora:0.6,char_lora:0.4" -o merged_resized -vv \
        -r "fro_ckpt=0.8,params=0.2,size=50"
    ```
    This merges `style_lora` (60% weight) and `char_lora` (40% weight), then resizes the merged result to approximately 50 MiB. The dimension scoring prioritizes comparison to the base model's Frobenius norm (`fro_ckpt`, 80% importance) while slightly favoring parameter efficiency (`params`, 20% importance). `-vv` shows detailed logs.

## Understanding Recipes

Recipes tell the script _how_ to decide which internal dimensions of the LoRA are important enough to keep. Each recipe consists of two parts: a **Selection Method** (`size` or `thr`) and one or more **Scoring Methods** (like `spn_ckpt`, `params`, etc.).

### The Goal: Scoring and Selecting Dimensions

Internally, a LoRA layer modifies the output of a standard neural network layer. This modification can be broken down (using Singular Value Decomposition, SVD) into a set of independent "directions" or "dimensions", each with an associated "strength" (its singular value, $\sigma_i$). Resizing works by discarding the dimensions with low strength/importance.

The core idea here is to calculate an **importance score** for each dimension. This score starts with the dimension's raw strength ($\sigma_i$) but is then adjusted based on comparisons defined by the scoring methods. These scores are calculated on a logarithmic (base 10) scale for numerical stability and easier interpretation.

### Selection Methods: `thr` vs. `size`

You must specify exactly one of these per recipe:

- `thr=<value>`: **Threshold Selection**

  - This sets a direct cutoff on the final log10 importance score. Any dimension whose score is _greater_ than `<value>` is kept.
  - Think of the threshold in terms of fractions:
    - `thr=-1.0`: Keep dimensions with strength > 1/10th of the reference magnitude(s).
    - `thr=-1.2`: Keep dimensions > $`10^{-1.2} \approx 1/16`$th of the reference.
    - `thr=-2.0`: Keep dimensions > $`10^{-2.0} \approx 1/100`$th of the reference.
  - Negative thresholds are typical when comparing to norms (like `spn_ckpt` or `fro_ckpt`).

- `size=<value>`: **Target Size Selection**
  - This sets a target output file size in MiB.
  - The script doesn't magically know the final size beforehand. Instead, it calculates the importance score for _all_ dimensions (using the specified scoring methods).
  - It then figures out the parameter cost (bytes needed) for each dimension.
  - It performs a greedy selection: Keep the dimensions with the highest score-per-byte until the total size budget (`<value>` MiB) is met.
  - This process effectively _calculates_ an internal threshold (`thr`) that achieves the target size. This calculated threshold is reported in the output filename and logs.
  - This is useful when you have a specific file size budget.

### Scoring Methods: Defining "Importance"

These methods determine _how_ we judge the importance of each LoRA dimension (represented by its singular value, $\sigma_i$). The core idea is **comparison**: we measure the strength of a LoRA dimension not in isolation, but _relative_ to one or more **reference points**. A dimension scoring highly is considered strong _compared to_ these chosen references.

Think of it like evaluating how significant a small change is. A $1 increase in price is negligible for a car but significant for a candy bar. Similarly, a LoRA dimension's strength might be considered important if it's large compared to the base model's effect (`spn_ckpt`), or large compared to the LoRA's _own_ strongest effect (`spn_lora`), or efficient in terms of parameters (`params`).

#### Weighting the Comparisons

You don't have to rely on just one comparison. Recipes allow you to _weight_ the importance of different reference points. For example `spn_ckpt=0.7,params=0.3` This means: "When deciding which dimensions to keep, I care 70% about how strong they are compared to the base model layer's peak strength (`spn_ckpt`), and 30% about how parameter-efficient they are (`params`)."

Weights are specified after the method name (e.g., `spn_ckpt=0.7`). If you omit the weight (e.g., just `spn_ckpt`), it defaults to 1.0. The tool automatically normalizes all specified weights so they sum to 1.0 (e.g., `fro_ckpt=3,spn_lora=1` is treated as `fro_ckpt=0.75,spn_lora=0.25`).

#### How Comparisons are Combined

The final importance score combines these weighted comparisons. Intuitively, a dimension gets a higher score if it's strong relative to the reference points you've weighted highly. Mathematically, the tool calculates a **weighted geometric mean** of the ratios $(\sigma_i / \text{Reference})$. This means dimensions that are consistently strong across your chosen comparisons will score best.

For numerical stability and easier thresholding, calculations are done using logarithms. The final score is essentially `log10(sigma_i) - w1*log10(Ref1) - w2*log10(Ref2)...`where`Ref`are the reference magnitudes and `w`s are the normalized weights you set for each method. As the final score, the `thr=` threshold is also specified in log10 space: `thr=-1` is a 1/10th cutoff, `thr=-2` is a 1/100th cutoff, etc.

Changing scoring methods and their weights will have a strong influence on the optimal threshold. For this reason you should probably target a fixec `size=` when comparing different recipes.

#### Available Methods

Here are the available methods and the reference magnitude they compare $\sigma_i$ against:

- `spn_ckpt`:
  - **Reference:** Max strength (spectral norm, $\sigma_\text{max}$) of the corresponding **base model layer**.
  - **Intuition:** How strong is the LoRA dimension relative to the base model's _peak_ effect in that layer? Good for general compression, aligns LoRA significance with the base model's scale.
- `spn_lora`:
  - **Reference:** Max strength (spectral norm, $\sigma_\text{max}$) of the **LoRA layer itself**.
  - **Intuition:** How strong is this dimension relative to the _strongest_ dimension within the _same_ LoRA layer? Keeps dimensions that are internally important to the LoRA's function. (Similar to Kohya's `sv_ratio`).
- `fro_ckpt`:
  - **Reference:** Overall magnitude (Frobenius norm, $\|W\|_F$) of the corresponding **base model layer**.
  - **Intuition:** How strong is the LoRA dimension relative to the base model's _total_ effect magnitude in that layer? Similar to `spn_ckpt` but considers the "average" strength across all base layer dimensions. Larger layers have more spread out singular values, so `fro_ckpt` penalizes them more than `spn_ckpt`.
- `fro_lora`:
  - **Reference:** Overall magnitude (Frobenius norm, $\|BA\|_F$) of the **LoRA layer itself**.
  - **Intuition:** How strong is this dimension relative to the _total_ effect magnitude of this LoRA layer?
- `subspace` (Experimental):
  - **Reference:** How much the **base model layer acts** along this specific LoRA dimension's direction ($`| \langle \mathbf{u}_i, W_\text{ckpt} \mathbf{v}_i \rangle |`$).
  - **Intuition:** Is the LoRA dimension strong _even after accounting for_ how the base model already operates in that specific direction? Penalizes dimensions where the LoRA effect might be redundant with the base model.
- `params`:
  - **Reference:** Parameter cost per rank ($n + m$ for an $n \times m$ layer).
  - **Intuition:** This doesn't compare strength-to-strength, but rather penalizes dimensions slightly if they reside in layers that are inherently parameter-hungry (per rank). Use small weights (e.g., `0.1`, `0.2`) primarily with the `size=` selection method to favor keeping dimensions in more "efficient" layers when constrained by a file size budget.
  - **Note:** `fro_ckpt` already implicitly penalizes layers with lot of parameters per dimension see [Spectral vs. Frobenius Norm](#technical-details-spectral-vs-frobenius-norm).

### The `rescale` Factor

- `rescale=<value>`: Multiplies all singular values $\sigma_i$ by `<value>` _before_ any scoring calculations. Default: `1.0`.
  - Useful if you want to globally adjust the LoRA's strength during resizing. For example, `rescale=0.8` would resize a slightly weaker version of the LoRA, and thus prune more dimensions. This factor is baked into the final weights.

## Output Filename Convention

The script generates informative filenames for the resized LoRAs:

`<lora_name>_<recipe_details>_th<threshold>.safetensors`

Where:

- `<lora_name>`: The name of the original LoRA file (or the merged name if applicable).
- `<recipe_details>`: A summary of the recipe used:
  - Scoring methods and their weights: `spnckpt1`, `frckpt0.8`, `params0.2`, etc. (Names are shortened, weights included if not 1.0).
  - `scale<value>` is added if `rescale` is not 1.0.
  - `size<value>` is added if the `size` selection method was used.
- `_th<threshold>`: The _final_ log10 threshold used for pruning.
  - If you specified `thr=X` in the recipe, this will be `_thX`.
  - If you specified `size=Y`, this will be the threshold _calculated_ by the script to meet that size target (e.g., `_th-3.142`).

Example: Running with `-r spn_ckpt=1,size=32` might produce `my_lora_spnckpt1_size32_th-2.871.safetensors`.

## Understanding Verbose Output (`-vv`)

When running with `-vv` (DEBUG level), you'll see lines like this for many layers during the "Scoring" phase:

```
DEBUG:root:dim:  8->5   rle_lora:  3.19% rle_ckpt:  0.03% lora_te1_text_model_encoder_layers_0_self_attn_out_proj
DEBUG:root:dim:256->0   rle_lora:100.00% rle_ckpt:  0.00% lora_unet_middle_block_0_in_layers_2
```

- `dim: <old> -> <new>`: Shows the original rank (number of dimensions) of the LoRA layer and the new rank after pruning based on the recipe's threshold. `256->0` means the entire LoRA layer was removed.
- `rle_lora: X.XX%`: **Relative LoRA Error**. This estimates the error introduced by removing dimensions, relative to the original LoRA layer's total magnitude (Frobenius norm). It's calculated as `norm(discarded_singular_values) / norm(all_singular_values)`. A value of `100.00%` means all dimensions were discarded (or the layer was zero to begin with). Lower is generally better.
- `rle_ckpt: Y.YY%`: **Relative Checkpoint Error**. This estimates the error relative to the base model checkpoint layer's magnitude (Frobenius norm). It's calculated as `norm(discarded_singular_values) / norm(base_layer_weights)`. This indicates how much the pruning changes the model's output _relative to the scale of the original model layer_. Often, this percentage is much smaller than `rle_lora`, suggesting the removed dimensions were small compared to the base model's weights, even if they were a larger fraction of the LoRA's own weights.
- `<layer_name>`: The name of the specific LoRA layer being processed.

You will also see `WARNING` messages for layers detected as being entirely zero in the original LoRA file. These layers are skipped. With SDXL, A LoRA layer is instanciated for some text encoder blocks but they are never trained. So this warning should be ignored in most cases.

Finally, INFO level (`-v` or `-vv`) shows `Score quantiles`, giving a statistical overview of the calculated importance scores across all dimensions before thresholding.

## Technical Details: Spectral vs. Frobenius Norm

The Frobenius norm $`\|W\|_F`$ and the spectral norm $`\sigma_{\max}(W)`$ are two ways to measure the "magnitude" of a matrix $`W \in \mathbb{R}^{n \times m}`$.

- **Spectral Norm:** $`\sigma_{\max}(W)`$ is the largest singular value. It represents the maximum scaling factor the matrix applies to any input vector.
- **Frobenius Norm:** $`\|W\|_F = \sqrt{\sum_{i=1}^{\min(n,m)} \sigma_i^2(W)}`$. It's like the Euclidean distance if you unroll the matrix into a vector.

For typical neural network weight matrices where singular values decrease slowly, there's an approximate relationship:

```math
\|W\|^2_F = \sum_i \sigma_i^2(W) \propto \min(n, m) \cdot \sigma_{\max}^2(W)
```

This means $`\|W\|_F \approx \sqrt{\min(n, m)} \cdot \sigma_{\max}(W)`$.

Comparing `fro_ckpt` and `spn_ckpt`:

- Scoring with `fro_ckpt=1` uses $\sigma_i(BA) / \|W_\text{ckpt}\|_\text{F}$ as the core ratio.
- Scoring with `spn_ckpt=1` uses $\sigma_i(BA) / \sigma_\text{max}(W_\text{ckpt})$.

**Therefore, `fro_ckpt` score is similar to `spn_ckpt` divided by $`\sqrt{\min(n,m)}`$.**

Due to the relationship above, using `fro_ckpt` implicitly penalizes layers with larger dimensions (higher $`\min(n,m)`$) more than `spn_ckpt` does, somewhat similar to adding a small weight to the `params` score.

## Why is it so fast?

### Efficient SVD Calculation

Calculating the Singular Value Decomposition (SVD) is central to analyzing the LoRA layers. A naive approach would involve reconstructing the full-rank matrix modification $\Delta W = BA$ (where $B$ is `lora_up.weight` and $A$ is `lora_down.weight`) before performing the SVD. If the layer dimensions are large (e.g., $n \times m$), this intermediate matrix $BA$ can be large and computationally expensive to form and decompose, even if the LoRA rank $r$ is small ($B$ is $n \times r$, $A$ is $r \times m$).

This tool avoids forming the full $BA$ matrix. Instead, it leverages the fact that we only need the SVD of $BA$. This can be computed more efficiently by operating directly on the smaller $B$ and $A$ matrices, often using QR decompositions:

1.  Compute the QR decomposition of $B$: $B = Q_B R_B$.
2.  Compute the QR decomposition of $A^T$: $A^T = Q_A R_A$, which means $A = R_A^T Q_A^T$.
3.  Form the much smaller $r \times r$ matrix $M = R_B R_A^T$.
4.  Compute the SVD of this small matrix: $M = U_M S V_M^T$.
5.  The singular values $S$ of $BA$ are the same as the singular values of $M$. The full singular vectors can be reconstructed if needed as $U = Q_B U_M$ and $V = Q_A V_M$.

This approach significantly reduces computational cost, as the expensive SVD is performed on a small $r \times r$ matrix, replacing the SVD on the potentially large $n \times m$ matrix $BA$.

### Base Model Statistics Caching

Several scoring methods (`spn_ckpt`, `fro_ckpt`, `subspace`) rely on properties of the corresponding layers in the base model checkpoint, specifically their spectral norm ($\sigma_{\max}(W_{\text{ckpt}})$) or Frobenius norm ($\|W_{\text{ckpt}}\|_F$). Computing these norms, especially the spectral norm (which requires an SVD or iterative methods), can be time-consuming.

To avoid redundant calculations, the tool automatically caches these base model layer statistics:

- When a norm for a specific base layer is needed for the first time, it is computed.
- The result is stored in a JSON file (e.g., `norms_cache.json`) associated with the `BaseCheckpoint`.
- On subsequent runs, or when processing other LoRA files against the _same_ base checkpoint, the tool reads the required norms directly from the cache file.

This caching mechanism speeds up processing after the first run, particularly when applying multiple recipes or resizing batches of LoRAs that share the same base model.

## Evaluation

(TBD) General procedure:

1. Compress a LoRA with default parameters `-r spn_ckpt=1,thr=-1.2`, note the file size.
2. Compress the same original LoRA with `-r spn_lora=1,size=<file size of the first output>`, note the threshold.
3. Compress the same original LoRA using [`kohya-ss/sd-scripts/networks/resize_lora.py`](https://github.com/kohya-ss/sd-scripts/blob/main/networks/resize_lora.py)` --dynamic_method="sv_ratio" --dynamic_param=<10**-threshold noted from 2.>`.

LoRAs from steps 2 and 3 should give similar results. Evaluation of `spn_ckpt` scoring compares step 1 against steps 2 and 3.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please open an issue to discuss changes or submit a pull request on GitHub.
