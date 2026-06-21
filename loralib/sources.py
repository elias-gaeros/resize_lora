from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import safetensors.torch


class AdapterFileSource:
    """Thin safetensors-backed adapter source used by the key-mapper harness."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd = safetensors.torch.safe_open(self.path, framework="pt", device="cpu")

    def keys(self) -> Iterable[str]:
        return self._fd.keys()

    def __iter__(self) -> Iterator[tuple[str, object]]:
        for key in self._fd.keys():
            yield key, self._fd.get_tensor(key)

    def get_tensor(self, key: str):
        return self._fd.get_tensor(key)

    def metadata(self) -> dict:
        return self._fd.metadata() or {}

    def __len__(self) -> int:
        return len(list(self._fd.keys()))

    def close(self) -> None:
        close = getattr(self._fd, "close", None)
        if callable(close):
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
