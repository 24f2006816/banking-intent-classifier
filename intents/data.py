"""Load Banking77 (PolyAI): 13,083 customer-support queries labelled with 77 banking intents."""

from __future__ import annotations

import csv
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
CACHE = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Split:
    texts: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.texts)


def _download(name: str, cache: Path) -> Path:
    path = cache / f"{name}.csv"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE_URL}/{name}.csv", path)  # noqa: S310 - fixed https URL
    return path


def read_csv(path: Path) -> Split:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return Split([r["text"].strip() for r in rows], [r["category"] for r in rows])


def load_banking77(cache: Path = CACHE) -> tuple[Split, Split]:
    return read_csv(_download("train", cache)), read_csv(_download("test", cache))


def per_class_subset(split: Split, per_class: int | None, seed: int = 0) -> Split:
    """Keep at most `per_class` examples of each intent (None keeps everything)."""
    if per_class is None:
        return split
    by_label: dict[str, list[int]] = {}
    for i, label in enumerate(split.labels):
        by_label.setdefault(label, []).append(i)
    rng = random.Random(seed)  # noqa: S311 - sampling, not security
    keep = sorted(i for idx in by_label.values() for i in rng.sample(idx, min(per_class, len(idx))))
    return Split([split.texts[i] for i in keep], [split.labels[i] for i in keep])
