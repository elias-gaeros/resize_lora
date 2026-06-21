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

Run the mapping harness with base checkpoints appropriate for the adapters:

```sh
python test_key_mapper.py compressed_sft_index.json \
    /path/to/sdxl-base.safetensors /path/to/flux-base.safetensors -v
```

The harness:

- classifies schemas using adapter suffix and checkpoint-prefix heuristics;
- builds one `KeyMapper` per supplied base checkpoint;
- scores each adapter schema by the fraction of unique module names mapped;
- applies simple cross-architecture penalties;
- reports frequent unmatched patterns and example rejections.

The score is diagnostic. It does not load adapter tensors, validate shapes, apply the
adapter, or compare generated output. Shared schemas are tested once, so the report is
weighted by schema unless file counts are inspected separately.

Use `--debug-key RAW_KEY` to inspect suffix stripping and lookup for one key, or
`--trace-key MODULE_NAME` to inspect the limited generator tracing currently available.

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
4. Mapping coverage does not regress for relevant schemas and base checkpoints.
5. New unmatched formats are documented rather than counted as successful mappings.
