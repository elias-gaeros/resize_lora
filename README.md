# LoRA Resize Tool

This tool compresses SDXL LoRA models while experimenting with new methods for selecting singular values. The proposed method scores the singular values of the LoRA layers relative to the spectral norm of the corresponding layers from the base model checkpoint. This score, called `spn_ckpt`, is computed as $\sigma_i(BA) / \sigma_{\text{max}}(W_{\text{ckpt}})$, where $\sigma_i(BA)$ represents the singular values of the LoRA layer, and $\sigma_{\text{max}}(W_{\text{ckpt}})$ is the maximum singular value (spectral norm) of the base layer from the checkpoint.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Command](#basic-command)
  - [Arguments](#arguments)
  - [Recipe Format](#recipe-format)
  - [Advanced Example](#advanced-example)
- [Evaluation](#evaluation)
- [License](#license)
- [Contributing](#contributing)

## Features

- Support SDXL LoRAs/LoCons only for now (This means **no SD1.5, and no DoRA, LoKr, LoHa**, etc).
- **Base model weights for thresholding**: Uses weights from the base model for thresholding the relative singular values of the LoRA.
- **Multiple scoring Methods**: Various scoring methods can be weighted together.
- **Target size or score threshold**: Specify either a target size or a score threshold.
- **Batch processing**: Process multiple LoRA models in one command.
- **Multiple recipes**: Apply multiple recipes resulting in multiple outputs per input LoRA. This allows to reuse the SVD decomposition for experimenting with a range of parameters.
- **Optimized for speed**: Uses 3 SVDs directly on the LoRA's weights instead of the full rank matrix. Spectral norms from the checkpoint are cached to the filesystem for faster processing.

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

## Usage

### Basic Command

```sh
python resize_lora.py /path/to/checkpoint.safetensors /path/to/lora.safetensors -o /path/to/output/folder
```

Filters out the singular values that are 10 times smaller or more than the spectral norm of the base layer from the checkpoint.

### Arguments

- `checkpoint_path` (positional): Path to the checkpoint file.
- `lora_model_paths` (positional): Paths to the LoRA model files (can specify multiple files).
- `-o`, `--output_folder` (required): Folder to save the output files.
- `-t`, `--output_dtype`: Output bit-width, either 32 or 16 (default: 16).
- `-d`, `--device`: Device to run the computations on (default: 'cuda' if available, otherwise 'cpu').
- `-r`, `--score_recipes`: Score/threshold recipes separated by colons (`:`). Each recipe specifies weights and either a target size or threshold. The default is `spn_ckpt=1,thr=-1.2`.
- `-v`, `--verbose`: Increase verbosity level (e.g., `-v` for `INFO`, `-vv` for `DEBUG`).

### Recipe Format

The score recipe string is a comma-separated list of key-value pairs. It specifies the weights for the geometric average of various score metrics and the threshold or the target file size. Valid keys are:

- `spn_ckpt`: Weight for the score relative to the spectral norm of the layer from the checkpoint. The score is $\sigma_i(BA) / \sigma_\text{max}\left(W_\text{ckpt}\right)$.
- `spn_lora`: Weight for the score relative to the spectral norm of the layer from the LoRA model. The score is $\sigma_i(BA) / \sigma_\text{max}\left(BA\right)$.
- `fro_ckpt`: Weight for the score relative to the Frobenius norm of the layer from the checkpoint. The score is $\sigma_i(BA) / \|W_\text{ckpt}\|_\text{F}$.
- `fro_lora`: Weight for the score relative to the Frobenius norm of the layer from the LoRA model. The score is $\sigma_i(BA) / \|BA\|_\text{F}$.
- `subspace`: Weight for the subspace score, computed as $\sigma_i(BA) / \left\langle \mathbf{u}_i, W_\text{ckpt} \mathbf{v}_i \right\rangle$.
- `size`: Target output size in MiBs (required if `thr` is not specified).
- `thr`: Log base 10 threshold for scores (required if `size` is not specified). A threshold of -1 (the default) will select the singular values $> 10^{-1} \times \text{reference}$, at least 1/10th of the reference. Typically negative except for `subspace` denominator.

`spn_lora` scoring should be equivalent to `kohya-ss/sd-scripts/networks/resize_lora.py --dynamic_method="sv_ratio"`.

Unlike `spn_lora`, `spn_ckpt` can remove LoRA layers completely. It is expected to generalize better across LoRAs and doesn't depend on the amount of training.

`subspace` is experimental and hasn't performed well alone. It favors singular subspaces that gets contracted by the base weights. It remains to be seen if low weights can improve other methods.

### Advanced Example

```sh
python resize_lora.py /path/to/checkpoint.safetensors /path/to/loras/*.safetensors -o /path/to/output/folder \
    -v -r spn_lora=1,thr=-0.7:spn_ckpt=1,thr=-1.2:subspace=0.5,spn_ckpt=0.5,size=32
```

Process multiple LoRAs from a folder. For each, outputs:

- `spn_lora=1,thr=-0.7`: a LoRA with singular values greater than $10^{-0.7} \simeq $ 1/5th of the largest one. This is equivalent to `kohya-ss/sd-scripts/networks/resize_lora.py --dynamic_method="sv_ratio" --dynamic_param=5`.
- `spn_ckpt=1,thr=-1.2`: a LoRA with singular values greater than $10^{-1.2} \simeq $ 1/16th of the largest singular value of the base layer (spectral norm). This is the default and recommended setting for a strong compression with relatively limited effects on the quality.
- `subspace=0.5,spn_ckpt=0.5,size=32`: a 32MiB LoRA with a dynamic threshold applied on the geometric mean of:
  - The singular values divided by the scaling of base layer on the singular subspaces.
  - The singular values divided by the spectral norm of the base layer.

## Evaluation

(TBD) General procedure:

1. Compress a LoRA with default parameters `-r spn_ckpt=1,thr=-1.2`, note the file size.
2. Compress the same original LoRA with `-r spn_lora=1,size=<file size of the first output>`, note the threshold.
3. Compress the same original LoRA using [`kohya-ss/sd-scripts/networks/resize_lora.py`](https://github.com/kohya-ss/sd-scripts/blob/main/networks/resize_lora.py)` --dynamic_method="sv_ratio" --dynamic_param=<10**-threshold noted from 2.>`.

LoRAs from steps 2 and 3 should give similar results. Evaluation of `spn_ckpt` scoring compares step 1 against steps 2 and 3.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.
