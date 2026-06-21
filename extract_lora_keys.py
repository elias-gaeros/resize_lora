"""Extract diverse, duplicate-free tensor-key fixtures from LoRA files."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from index_safetensors import parse_safetensors_header


FORMAT_MARKERS = {
    "lora": (".lora_down.weight", ".lora_up.weight"),
    "peft_lora": (".lora_A.weight", ".lora_B.weight"),
    "svdquant_lora": (".lora_proj_down", ".lora_proj_up"),
    "dora": (".dora_scale",),
    "loha": (".hada_w1_a.weight", ".hada_w1_b.weight"),
    "lokr": (".lokr_w1", ".lokr_w1.weight"),
    "oft": (".oft_blocks",),
    "full_diff": (".diff", ".diff_b", ".diff_w"),
}


@dataclass(frozen=True)
class KeyFixture:
    family: str
    source: str
    fingerprint: str
    formats: tuple[str, ...]
    keys: tuple[str, ...]


def fingerprint_keys(keys: Iterable[str]) -> str:
    payload = json.dumps(sorted(keys), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def detect_formats(keys: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(keys)
    formats = {
        name
        for name, suffixes in FORMAT_MARKERS.items()
        if any(key.endswith(suffixes) for key in keys)
    }
    return tuple(sorted(formats or {"unknown"}))


def read_fixture(path: Path, root: Path) -> KeyFixture | None:
    header = parse_safetensors_header(path)
    if header is None:
        return None
    keys = tuple(sorted(key for key in header if key != "__metadata__"))
    if not keys:
        return None
    try:
        relative = path.resolve().relative_to(root.resolve())
        family = relative.parts[0] if len(relative.parts) > 1 else root.name
        source = relative.as_posix()
    except ValueError:
        family = path.parent.name
        source = str(path.resolve())
    return KeyFixture(family, source, fingerprint_keys(keys), detect_formats(keys), keys)


def scan_roots(roots: Iterable[Path]) -> list[KeyFixture]:
    fixtures = []
    seen_paths = set()
    for root in roots:
        for path in sorted(root.rglob("*.safetensors")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            fixture = read_fixture(resolved, root)
            if fixture is not None:
                fixtures.append(fixture)
    return fixtures


def select_diverse(fixtures: Iterable[KeyFixture], limit: int = 0) -> list[KeyFixture]:
    """Select unique schemas while prioritizing family and format coverage."""
    fixtures = list(fixtures)
    family_schema_counts: dict[str, set[str]] = {}
    by_fingerprint: dict[str, list[KeyFixture]] = {}
    for fixture in fixtures:
        family_schema_counts.setdefault(fixture.family, set()).add(fixture.fingerprint)
        by_fingerprint.setdefault(fixture.fingerprint, []).append(fixture)

    unique = []
    for candidates in by_fingerprint.values():
        # Keep shared schemas in the family that has fewer alternatives.
        unique.append(
            min(
                candidates,
                key=lambda item: (
                    len(family_schema_counts[item.family]),
                    item.family,
                    item.source,
                ),
            )
        )

    remaining = sorted(unique, key=lambda item: (item.family, item.source))
    selected: list[KeyFixture] = []
    covered_families: set[str] = set()
    covered_formats: set[str] = set()
    family_selections: dict[str, int] = {}
    target = len(remaining) if limit <= 0 else min(limit, len(remaining))

    while remaining and len(selected) < target:
        best = max(
            remaining,
            key=lambda item: (
                item.family not in covered_families,
                len(set(item.formats) - covered_formats),
                -family_selections.get(item.family, 0),
                -len(family_schema_counts[item.family]),
            ),
        )
        remaining.remove(best)
        selected.append(best)
        covered_families.add(best.family)
        covered_formats.update(best.formats)
        family_selections[best.family] = family_selections.get(best.family, 0) + 1

    assert len({item.fingerprint for item in selected}) == len(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=24,
        help="Maximum fixtures to write; use 0 for every unique schema.",
    )
    args = parser.parse_args()

    fixtures = select_diverse(scan_roots(args.roots), args.limit)
    output = {
        "version": 1,
        "fixture_count": len(fixtures),
        "fixtures": [asdict(fixture) for fixture in fixtures],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(fixtures)} unique key fixtures to {args.output}")


if __name__ == "__main__":
    main()
