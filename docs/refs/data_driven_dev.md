# Data-Driven QA for Adapter Key Mapping

## Goal

Unit tests cover known transformations, but real adapter collections expose additional
naming conventions and partial schemas. The corpus workflow inventories those schemas,
deduplicates them, and measures mapping coverage against explicit base checkpoints.

Schema extraction and semantic mapping are separate checks. Finding every key in a
safetensors header does not prove that those keys map to the correct base layer.

## Build an Index

Create a raw index. Tensor payloads are not loaded, but metadata and local absolute
paths are included:

```sh
python index_safetensors.py /path/to/models /path/to/loras -o sft_index.json
```

Convert it to the pooled format used by the analysis tools:

```sh
python - <<'PY'
import json
from pathlib import Path
from compressed_sft_index import compress_index

raw = json.loads(Path("sft_index.json").read_text(encoding="utf-8"))
Path("compressed_sft_index.json").write_text(
    json.dumps(compress_index(raw)), encoding="utf-8"
)
PY
```

Version 8.1 preserves dotted and underscored keys losslessly and tags string-pool
references so integer and boolean metadata round-trip without reinterpretation.
`CompressedIndex` can still read legacy 8.0 files, whose underscore reconstruction and
untagged integer metadata are inherently ambiguous.

## Inspect and Test

Summarize schemas before testing mappings:

```sh
python summarize_sft_index.py compressed_sft_index.json --stats --sample 10
```

Pair the adapters with base checkpoints appropriate for them:

```sh
python check_mapping.py /path/to/sdxl-base.safetensors \
    /path/to/flux1-dev.safetensors,/path/to/clip_l.safetensors -l /path/to/loras -v
```

`check_mapping.py` runs the pairing that `resize_lora.py` uses, on headers only. For
each LoRA it chooses the base that pairs the most low-rank layers, then prints the
paired, shape-mismatched and unmapped layer counts and the number of tensors that
resizing would copy unchanged. `-v` lists the unpaired layer names with digits replaced
by `N`. A base can be a JSON copy of a safetensors header, so a corpus can be checked
against base models whose weights are not on disk; the header of a remote file can be
read with two HTTP range requests (the 8-byte length, then the JSON).

A pairing requires a matching name and matching input and output sizes, so a shape
mismatch indicates a wrong rule or a wrong base. The check does not apply the adapter
or compare generated output.

## Extract Review Fixtures

For code review and regression tests, produce a smaller fixture that contains one copy
of each selected key schema:

```sh
python extract_lora_keys.py /path/to/loras \
    -o lora_key_fixtures.json --limit 24
```

Selection first removes global duplicate schemas, then prioritizes family and detected
format coverage. `--limit 0` writes every unique schema. Fingerprints are SHA-256 hashes
of sorted tensor names; tensor shapes and metadata do not affect deduplication.

## Validation Snapshot

On 2026-10-26, `check_mapping.py` paired 493 readable files from
`~kade/wolfy/models/loras/`, `~kade/wolfy/models/loras-piezo/` and
`/mnt/old_root/home/kade/diffusion/comfy/loras/` with the headers of Chroma1-HD,
Flux.2 Klein base 4B, Flux.1 dev and schnell, SDXL base 1.0, SD1.5, SD3.5 Large and
Medium, and Wan 2.1 T2V 1.3B, T2V 14B and I2V 14B. 9 further files could not be opened.

- 125,538 of 125,682 low-rank layers paired, with no shape mismatch.
- The 144 unpaired layers are the CLIP-L layers of two kohya Flux LoRAs. With a CLIP-L
  header added to the Flux base, both files pair completely.
- 77 files contain no plain low-rank layer: LoKr (noob, pony, SD3.5, IL), Klein files
  with `.R`/`.scale` tensors, and one with `lora_proj_down`/`lora_proj_up`.
- The SD UNet ldm-to-diffusers names were compared with sd-mecha's model configs: all
  794 SDXL and 282 SD1 UNet matrices convert to the diffusers names with equal shapes.

On 2026-06-21, the corpus at `~kade/wolfy/models/loras/` contained 402 safetensors
files. Header extraction found 42 unique key schemas across seven top-level families.
The raw-index to compressed-index to decompressed-index round-trip was exactly equal for
all 402 files.

The 42 schemas included standard LoRA, PEFT LoRA, SVDQuant LoRA, DoRA, and LoKr marker
sets. One Klein schema containing `.R` and normalization `.scale` tensors remained
`unknown`. This validates indexing and deduplication on that corpus; it is not a mapping
accuracy percentage because matching Klein and LTX base checkpoints were not part of
that run.

## Acceptance Criteria

A mapping change should satisfy all of the following:

1. `pytest -q` passes with a focused regression for the changed convention.
2. `python -m compileall -q .` succeeds.
3. Raw/compressed index round-trip equality remains true on the review corpus.
4. `check_mapping.py` coverage does not regress and reports no shape mismatch for
   relevant LoRAs and base checkpoints.
5. New unmatched formats are documented rather than counted as successful mappings.
