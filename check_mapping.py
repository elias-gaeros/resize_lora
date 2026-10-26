"""Report how the low-rank layers of LoRA files map onto base checkpoints.

Reads headers only. Each LoRA is paired with the base checkpoint that maps most of its
layers, using the same pairing as resize_lora.py. A base can be a safetensors file or a
JSON copy of its header.
"""

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

from loralib import BaseCheckpoint, LoRADict, PairedLoraModel


def pattern(name):
    return re.sub(r"\d+", "N", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bases", nargs="+", type=Path, help="Base checkpoints")
    parser.add_argument(
        "-l",
        "--loras",
        nargs="+",
        type=Path,
        required=True,
        help="LoRA files or folders",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="List unpaired layer patterns"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    bases = [BaseCheckpoint(path) for path in args.bases]
    lora_paths = [
        p
        for root in args.loras
        for p in (sorted(root.rglob("*.safetensors")) if root.is_dir() else [root])
    ]

    totals = Counter()
    unpaired = Counter()
    print("paired/layers  mismatched  unmapped  copied  base  lora")
    for path in lora_paths:
        try:
            lora = LoRADict(path)
        except (OSError, PermissionError) as e:
            print(f"unreadable: {path} ({e.__class__.__name__})")
            continue
        # The best base pairs the most layers, then has the fewest shape mismatches.
        paired = max(
            (PairedLoraModel(lora, base) for base in bases),
            key=lambda p: (len(p.targets), -len(p.mismatched)),
        )
        n_layers = len(lora.layers)
        row = (
            len(paired.targets),
            len(paired.mismatched),
            len(paired.unmapped),
            len(paired.passthrough_keys),
        )
        base_name = paired.checkpoint.path.stem if n_layers else "-"
        unpaired.update(pattern(n) for n in paired.unmapped)
        unpaired.update(pattern(m.split(" ")[0]) for m in paired.mismatched)
        totals.update(
            files=1, layers=n_layers, paired=row[0], mismatched=row[1], unmapped=row[2]
        )
        if n_layers and row[0] == n_layers:
            totals["fully_paired_files"] += 1
        print(
            f"{row[0]:>6}/{n_layers:<6} {row[1]:>10} {row[2]:>9} {row[3]:>7}  "
            f"{base_name}  {path}"
        )

    print(
        f"\n{totals['files']} files, {totals['fully_paired_files']} with every low-rank "
        f"layer paired; layers: {totals['paired']} paired, {totals['mismatched']} "
        f"shape mismatches, {totals['unmapped']} unmapped, of {totals['layers']}"
    )
    if args.verbose and unpaired:
        print("\nUnpaired layer patterns:")
        for name, count in unpaired.most_common():
            print(f"{count:>7}  {name}")


if __name__ == "__main__":
    main()
