"""Read YOLO-format ground-truth labels as :class:`Detections` (used by evaluation and tests)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from siteguard.types import CLASS_NAMES, Detections


def label_path(image_path: str | Path) -> Path:
    p = Path(image_path)
    parts = list(p.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(image_path: str | Path, width: int, height: int, names: list[str] | None = None) -> Detections:
    lp = label_path(image_path)
    rows = [r.split() for r in lp.read_text().splitlines() if r.strip()] if lp.exists() else []
    if not rows:
        return Detections.empty(names)
    a = np.array(rows, dtype=float)
    cx, cy, w, h = a[:, 1] * width, a[:, 2] * height, a[:, 3] * width, a[:, 4] * height
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return Detections(xyxy, np.ones(len(a)), a[:, 0].astype(int), names or CLASS_NAMES)
