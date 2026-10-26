# LoRA Resize Tool

This tool compresses LoRA models by selectively reducing their rank (the number of internal dimensions). It goes beyond simple truncation by using methods to score the importance of each dimension, comparing the LoRA's characteristics against the original base model checkpoint. This context-aware approach helps preserve the LoRA's essential effects while significantly reducing file size.

It resizes the low-rank layers (LoRA, LoCon, DoRA) of the LoRA files ComfyUI loads, and pairs them with the base model the same way ComfyUI does:

- **Base models:** SD1.x, SD2.x, SDXL and its finetunes, SD3 and SD3.5, Flux.1, Chroma, Flux.2 Klein, Wan, and other diffusion models whose LoRAs use the checkpoint's own layer names.
- **Layer names:** ComfyUI's own (`diffusion_model.…`, `text_encoders.…`), kohya (`lora_unet_…`, `lora_te1_…`), diffusers and PEFT (`unet.…`, `transformer.…`, `text_encoder.…`) for SD/SDXL UNets, SD3 and Flux.1, and LyCORIS (`lycoris_…`).
- **Tensor names:** `lora_up`/`lora_down`, `lora_B`/`lora_A`, `lora.up`/`lora.down` and the other low-rank suffixes ComfyUI accepts, with or without `.alpha`. The output keeps the input's names and suffixes.

Diffusers names for other architectures (HunyuanVideo, Qwen-Image, Z-Image, AuraFlow, PixArt) are not mapped. LoHa, LoKr, OFT and other adapters that are not a product of two matrices are not resized.

Tensors the tool cannot resize (LyCORIS `.diff` layers, LoHa/LoKr factors, layers without a base-layer mapping or whose shape differs from it) are copied to the output unchanged and listed in a warning. The `size=` budget includes their bytes; `rescale` does not apply to them. When merging LoRAs, these tensors and DoRA magnitudes (`dora_scale`) are dropped instead. A LoRA with no low-rank layer, or with fewer than half of its low-rank layers matching the base model, is skipped with an error.

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
- [Merging LoRAs](#merging-loras)
  - [Weights](#weights)
  - [Merge modes](#merge-modes)
  - [Differences from the original CtM](#differences-from-the-original-ctm)
  - [Choosing CtM parameters](#choosing-ctm-parameters)
  - [How CtM works](#how-ctm-works)
- [Output Filename Convention](#output-filename-convention)
- [Understanding Verbose Output (`-vv`)](#understanding-verbose-output--vv)
- [Technical Details: Spectral vs. Frobenius Norm](#technical-details-spectral-vs-frobenius-norm)
- [Why is it so fast?](#why-is-it-so-fast)
- [Indexing and Key-Mapping Tools](#indexing-and-key-mapping-tools)
- [Evaluation](#evaluation)
- [License](#license)
- [Contributing](#contributing)

## Features

- **ComfyUI Compatible**: Accepts the LoRA layer names ComfyUI loads, for every base model listed above.
- **Base Model Aware**: Uses weights from the base model checkpoint for more robust dimension scoring.
- **Flexible Scoring**: Combine multiple metrics (comparing LoRA to itself, to the base model, considering parameter efficiency) using weighted recipes.
- **Target Size or Threshold**: Prune dimensions based on a specific target file size (`size=`) or a direct importance score threshold (`thr=`).
- **Batch Processing**: Process multiple LoRA files in a single command.
- **Multiple Recipes**: Apply several different compression recipes to the same LoRA simultaneously, facilitating experimentation without redundant SVD calculations.
- **LoRA Merging**: Merge several LoRAs before resizing, by weighted sum or by Compress then Merge (CtM).
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

This command uses the default recipe (`fro_ckpt=1,thr=-3.5`). It compresses the LoRA by keeping only the dimensions whose "strength" (singular value) is greater than roughly 1/16th ($10^{-1.2}$) of the "maximum strength" (spectral norm) of the corresponding layer in the base model.

### Arguments

- `checkpoint_path` (positional): Path to the base model checkpoint file (e.g., SDXL 1.0 base).
  - A base model split across files takes them all, separated by commas: `flux1-dev.safetensors,clip_l.safetensors,t5xxl_fp16.safetensors`. Text encoder layers of a LoRA are only resized when their text encoder is given; otherwise they are copied unchanged.
  - Full checkpoints, ComfyUI diffusion-model files and "fp8 scaled" files are accepted. GGUF, SVDQuant and 4-bit (block-scaled) files are not.
- `lora_model_paths` (positional): Path(s) to the LoRA model files.
  - You can specify multiple files: `lora1.safetensors lora2.safetensors`.
  - You can use wildcards (shell expanded): `loras/*.safetensors`.
  - You can specify a weighted merge: `"lora1:0.7,lora2:0.3"` (use quotes if needed by your shell). The tool merges these LoRAs _before_ resizing, with the method `-m` selects.
- `-o`, `--output_folder` (required): Folder where the resized LoRA files will be saved.
- `-t`, `--output_dtype`: Output precision (`16` for float16, `32` for float32). Default: `16`.
- `-d`, `--device`: Device for computations (`cuda`, `cpu`, etc.). Default: `cuda` if available, otherwise `cpu`.
- `-r`, `--score_recipes`: Defines how to score and prune dimensions. Separate multiple recipes with colons (`:`). Default: `fro_ckpt=1,thr=-3.5`. See [Understanding Recipes](#understanding-recipes) below.
- `-m`, `--merge`: How comma-separated LoRAs are merged: `sum` or `ctm[,beta=B][,ties=D]`. Separate several methods with colons (`:`). Default: `sum`. See [Merging LoRAs](#merging-loras).
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

4.  **Compare a sum merge with Compress then Merge at the same size:**
    ```sh
    python resize_lora.py sdxl_base.safetensors "style_lora,char_lora" -o merged -v \
        -m "sum:ctm:ctm,ties=0.25" -r "fro_ckpt=1,size=64"
    ```
    This writes three 64 MiB merges: the weighted sum, CtM with summed coordinates, and CtM with TIES. See [Merging LoRAs](#merging-loras).

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
  - Useful if you want to globally adjust the LoRA's strength during resizing. For example, `rescale=0.8` would resize a slightly weaker version of the LoRA, and thus prune more dimensions when the recipe scores against the base model. `spn_lora` and `fro_lora` compare the LoRA to itself, so the factor cancels out in them. This factor is baked into the final weights.

## Merging LoRAs

A comma-separated list of LoRAs merges them into one LoRA before the recipes resize it:

```sh
python resize_lora.py base.safetensors "style_lora:0.7,char_lora:0.7" -o merged -r "fro_ckpt=1,size=64"
```

### Weights

The number after each `:` multiplies that LoRA's change to the base model, like the strength it is loaded with in ComfyUI; an omitted weight is 1. LoRAs that each work at strength 1 can be too strong together, because their changes add up. Lowering every weight, for example to `0.7` each, weakens all of them, as lowering both strengths does when the LoRAs are loaded together. Unequal weights favour one LoRA, e.g. `"style_lora:0.4,char_lora:0.9"`.

With `-m sum`, the merge makes the same change to the base model as loading the LoRAs together at these strengths, before the recipe resizes it. The same weight on every LoRA gives the same result as weights of 1 with the recipe's `rescale=` set to that weight. With `-m ctm`, the weights also set how much of each layer's directions each LoRA gets, unless `beta=0`.

### Merge modes

`-m` selects how the LoRAs are combined. Several modes separated by `:` write one output per mode and recipe. Each merged file records the mode, its parameters, and the merged LoRAs' names and weights as JSON in the `merge_params` metadata key.

- **`sum`** (default) merges, then compresses. It adds the weighted LoRAs exactly, then the recipe keeps the strongest directions of the sum. These are the directions where one LoRA is strong or where the LoRAs agree: a direction only the weaker LoRA uses can be pruned in favour of the stronger LoRA's, and changes of opposite sign cancel before the recipe sees them.
- **`ctm`** compresses, then merges ([Compress then Merge](https://openreview.net/forum?id=p32nWlgwYC), He et al., ICML 2026). For each layer, it first chooses the directions that keep the most of each LoRA taken separately, not of their sum, then merges the LoRAs within these directions. The recipe still decides how many directions each layer keeps; CtM decides which. Within them the LoRAs are added, or, with `ties=`, TIES-merged: each LoRA keeps only its largest values, each value takes the sign of their sum, and the values with that sign are averaged, so that LoRAs pulling in opposite directions do not cancel out.

Layers are matched by name. When two LoRAs name the same base layer differently (for example kohya and PEFT names), the output keeps both layers, but ComfyUI applies only one of them, so one LoRA's change to that layer is lost; the tool warns when this happens. A layer that modifies a slice of a fused weight (diffusers `to_q` on Flux's `qkv`) and one that modifies the whole weight are both applied.

### Differences from the original CtM

The [original implementation](https://github.com/ZhengbaoHe/compress-then-merge) (`merge_space=low_rank_core` in `task_merger.py`) merges task LoRAs of CLIP and language models. This version differs as follows:

- **Rank.** The original gives every layer the same rank (`--lora_rank`, 16 in the paper). Here the recipe chooses each layer's rank from the merge's component norms (see [How CtM works](#how-ctm-works)), with the same scores, thresholds and `size=` budget as for a single LoRA.
- **Strength.** The original scales the merge by one coefficient after merging (`--scaling_coeffs`). Here each LoRA's weight scales it before merging, and the recipe's `rescale=` scales the result.
- **Inputs.** The original reads the PEFT `lora_A` and `lora_B` matrices of linear layers without their alpha scaling, and every LoRA must have every layer. Here alphas are applied, convolution (LoCon) layers and every naming scheme this tool reads are accepted, and a layer that only one LoRA has, or that the others leave untrained (zero), is resized like that LoRA alone. When the LoRAs' summed rank exceeds a dimension of the layer, the larger side keeps its extra directions (see [How CtM works](#how-ctm-works)), so that keeping every dimension loses nothing.
- **Computation.** The shared bases come from QR decompositions instead of SVDs; they span the same subspaces, and the result does not depend on the choice of basis. The Tucker-2 decomposition runs in PyTorch, in double precision on the CPU, instead of tensorly's `partial_tucker`, with the same algorithm. It iterates until the kept energy grows by less than $10^{-12}$ of itself; the original stops when the relative reconstruction error changes by less than $10^{-6}$, which leaves the directions about $10^{-3}$ from convergence.
- **Merges.** Only the sum and TIES merges of the coordinates are implemented. The original's other merges and baselines (DARE, KnOTS, LoRA-LEGO, RobustMerge, Iso-C, Iso-CTS) are not.

### Choosing CtM parameters

- `beta=B`, between 0 and 1, default `0.25`: how evenly the LoRAs share each layer's directions. 1 shares them by the LoRAs' norms (weights included); 0 gives every LoRA the same share, whatever its norm and weight. It changes which directions are kept, not how many.
- `ties=D`, between 0 (excluded) and 1: the fraction of values each LoRA keeps in TIES; of $d$ values it keeps the $\lfloor D d \rfloor + 1$ largest, as the original does, and all of them for $D = 1$. Omitted, the LoRAs are added.

The paper's parameters, `--beta=0.25 --topK=25 --scaling_coeffs=0.75`, translate to `ctm,beta=0.25,ties=0.25` and `rescale=0.75`. They were tuned on CLIP and language-model tasks, so they are a starting point, not values tuned for diffusion LoRAs. Start by comparing the modes at the same `size=`, with the weights you would load the LoRAs with and verbose output:

```sh
python resize_lora.py base.safetensors "style_lora:0.7,char_lora:0.7" -o merged -v \
    -m "sum:ctm:ctm,ties=0.25" -r "fro_ckpt=1,size=64"
```

TIES averages the values the LoRAs agree on instead of adding them, so it can be weaker than `sum`: for two Chroma LoRAs with every dimension kept, the norm of the TIES merge was 78 to 82% of the sum's for `ties` from 0.25 to 1. Raise the weights or `rescale=` to compensate.

With `-v`, each CtM output logs, for each LoRA, the part of its norm that falls outside the chosen directions over all layers. For the two Chroma LoRAs (rank 32 each) merged at `size=64`:

```
INFO:root:CtM subspace error over all layers: style_lora 33.39%, char_lora 23.76%
```

- **One LoRA's error is much larger than the others'.** That LoRA is crowded out. Lower `beta`, down to 0, to give it more of the directions; the other LoRAs' errors rise accordingly.
- **Every error is large.** The size budget is too small for the merge: `beta` only moves error between LoRAs. Raise `size=` or lower `thr=`. With the default `thr=-3.5` recipe instead of `size=64`, the same two LoRAs lost under 1% each.

With `-vv`, the `ctm` line of each layer gives the same error per layer (see [Understanding Verbose Output](#understanding-verbose-output--vv)), which shows the layers where the LoRAs compete for directions. The error does not include what TIES changes; judge `ties` and the overall strength from generated images.

### How CtM works

For one layer, let $T$ be the number of LoRAs that have it, $B_t$ the up matrix of LoRA $t$ multiplied by its weight and by its alpha over its rank $r_t$, $A_t$ its down matrix, and $R = \sum_t r_t$.

1. **Shared bases.** The QR decompositions $[B_1 \cdots B_T] = Q_B R_B$ and $[A_1; \ldots; A_T]^\top = Q_A R_A$ give orthonormal bases shared by all the LoRAs. In these bases LoRA $t$ is an $a \times b$ core $M_t$ with $B_t A_t = Q_B M_t Q_A^\top$, without loss: $a$ and $b$ are $R$, or the layer's output and input dimensions where they are smaller than $R$. Let $n = \min(a, b)$.
2. **Component norms.** The higher-order SVD of the cores gives output directions $p_1, \ldots, p_a$ (left singular vectors of $[M_1 \cdots M_T]$) and input directions $q_1, \ldots, q_b$ (left singular vectors of $[M_1^\top \cdots M_T^\top]$). Let $E_j = \sum_t \|P_j^\top M_t Q_j\|_F^2$ be the energy of all the LoRAs that $P_j$ and $Q_j$ keep, where $P_j$ holds the first $j + a - n$ output directions and $Q_j$ the first $j + b - n$ input directions: when the cores are not square, the larger side starts with its $|a - b|$ extra directions. The component norms are $s_j = \sqrt{E_j - E_{j-1}}$ for $j = 1, \ldots, n$, sorted in decreasing order: the norm each pair of directions adds. For a single LoRA they are its singular values, and $\sum_j s_j^2 = \sum_t \|M_t\|_F^2$.
3. **Rank.** The recipe scores the $s_j$ in place of the singular values, so `thr=`, `size=` and the scoring methods choose the layer's rank $k$ as they do for a single LoRA. With `subspace` scoring, the merge keeps the number of dimensions the recipe selects, not necessarily the selected ones.
4. **Subspaces.** A Tucker-2 decomposition selects matrices $P_B$ and $P_A$ with $k + a - n$ and $k + b - n$ orthonormal columns (both $k$ when $a = b$) that maximize the energy kept from all the LoRAs, $\sum_t \|P_B^\top M_t P_A\|_F^2$. For this selection only, $\beta$ (`beta`) replaces each norm $\|M_t\|_F$ by $\beta \|M_t\|_F + (1 - \beta)\,\overline{\|M\|_F}$, where $\overline{\|M\|_F}$ is the mean norm of the $T$ LoRAs. $\beta = 1$ keeps the norms; $\beta = 0$ gives every LoRA the same norm, so that a weak LoRA is not crowded out by a strong one. `beta` changes which directions the merge keeps, not how many.
5. **Merge.** The coordinates $C_t = P_B^\top M_t P_A$, of size $(k + a - n) \times (k + b - n)$, are summed, or TIES-merged when `ties` is set: each $C_t$ keeps its $\lfloor D d \rfloor + 1$ largest-magnitude entries out of $d$, where $D$ is `ties`, each entry takes the sign of the sum of the kept values, and the kept values that have this sign are averaged.
6. **Output.** The merged layer is $Q_B P_B C \, (Q_A P_A)^\top$, where $C$ is the merged coordinate matrix. One side of $C$ has size $k$, so the layer has rank at most $k$.

A layer that only one LoRA has, or that the other LoRAs leave at zero (untrained), is resized like that LoRA alone, without TIES. When the recipe keeps all $n$ dimensions and `ties` is not set, `ctm` gives the same layers as `sum`.

## Output Filename Convention

The script generates informative filenames for the resized LoRAs:

`<lora_name>_<recipe_details>_th<threshold>.safetensors`

Where:

- `<lora_name>`: The name of the original LoRA file. For a merge, the first word of each merged file's name, joined by `+`, with the weights that are not 1; `-m ctm` appends its parameters, e.g. `_ctm-b0.25-ties0.25`.
- `<recipe_details>`: A summary of the recipe used:
  - Scoring methods and their weights: `spnckpt1`, `frckpt0.8`, `params0.2`, etc. (Names are shortened, weights included if not 1.0).
  - `scale<value>` is added if `rescale` is not 1.0.
  - `size<value>` is added if the `size` selection method was used.
- `_th<threshold>`: The _final_ log10 threshold used for pruning.
  - If you specified `thr=X` in the recipe, this will be `_thX`.
  - If you specified `size=Y`, this will be the threshold _calculated_ by the script to meet that size target (e.g., `_th-3.142`).

Example: Running with `-r spn_ckpt=1,size=32` might produce `my_lora_spnckpt1_size32_th-2.871.safetensors`.

When two outputs of one run have the same name, for example from recipes whose weights normalize to the same values or from `-m "ctm:ctm,beta=0.25"`, only the first is saved and the others are reported as errors.

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

With `-m ctm`, `-vv` also shows one line per merged layer during the "Scoring" phase:

```
DEBUG:root:ctm dim: 64->32  subspace rle: style_lora:9.81% char_lora:12.40% lora_unet_input_blocks_4_1_proj_in
```

- `dim: <R> -> <k>`: the sum of the LoRAs' ranks, and the rank the recipe chose for the merged layer.
- `subspace rle`: for each LoRA, the part of its layer that falls outside the CtM subspaces, relative to its layer: $\|M_t - P_B P_B^\top M_t P_A P_A^\top\|_F / \|M_t\|_F$, with the symbols of [How CtM works](#how-ctm-works). It does not include the changes TIES makes.

INFO level shows the same error over all layers for each recipe, e.g. `CtM subspace error over all layers: style_lora 33.39%, char_lora 23.76%`. A LoRA with a much larger error than the others is crowded out of the subspaces; a lower `beta` gives it more room. With `-m ctm`, `rle_lora` estimates the error relative to all the merged LoRAs from the component norms, $\sqrt{\sum_{j > k} s_j^2 / \sum_j s_j^2}$; the `ctm` line gives each LoRA's actual subspace error.

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

## Indexing and Key-Mapping Tools

`check_mapping.py` reports how the low-rank layers of LoRA files pair with base models,
reading headers only. Each LoRA is paired with the base that maps most of its layers:

```sh
python check_mapping.py sdxl_base.safetensors flux1-dev.safetensors,clip_l.safetensors \
    -l /path/to/loras -v
```

Each output line gives the paired, shape-mismatched and unmapped layer counts, the
number of tensors that would be copied unchanged, and the chosen base. `-v` lists the
unpaired layer names with digits replaced by `N`. A base can also be a JSON copy of a
safetensors header, which lets a corpus be checked without downloading the weights.

The other tools inspect safetensors schemas. Create a raw header-only index without
loading tensor payloads:

```sh
python index_safetensors.py /path/to/models /path/to/loras -o sft_index.json
```

Compress the index for the summary tool:

```sh
python - <<'PY'
import json
from pathlib import Path
from compressed_sft_index import compress_index

source = json.loads(Path("sft_index.json").read_text(encoding="utf-8"))
compressed = compress_index(source)
Path("compressed_sft_index.json").write_text(
    json.dumps(compressed), encoding="utf-8"
)
PY
python summarize_sft_index.py compressed_sft_index.json --stats --sample 5
```

To create a small, diverse, duplicate-free key fixture from a large LoRA corpus:

```sh
python extract_lora_keys.py /path/to/loras -o lora_key_fixtures.json --limit 24
```

`--limit 0` retains every distinct key schema. A fixture contains names and source
paths only, not tensor data. The source paths may still disclose local filenames, so
review the JSON before publishing it.

See [the key-map design](docs/refs/keymapper%20design.md) and
[the data-driven QA workflow](docs/refs/data_driven_dev.md) for implementation details
and current validation scope.

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
