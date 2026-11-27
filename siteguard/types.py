"""Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASS_NAMES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
    "forklift",
    "no_vest",
]


@dataclass
class Detections:
    """Detections for one image, in pixel coordinates (x1, y1, x2, y2)."""

    xyxy: np.ndarray  # (N, 4) float32
    confidence: np.ndarray  # (N,) float32
    class_id: np.ndarray  # (N,) int
    names: list[str]

    def __post_init__(self):
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        self.class_id = np.asarray(self.class_id, dtype=int).reshape(-1)

    def __len__(self) -> int:
        return len(self.xyxy)

    @classmethod
    def empty(cls, names: list[str] | None = None) -> Detections:
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int), names or CLASS_NAMES)

    def class_name(self, i: int) -> str:
        return self.names[int(self.class_id[i])]

    def where(self, name: str) -> np.ndarray:
        if name not in self.names:
            return np.zeros(0, dtype=int)
        return np.flatnonzero(self.class_id == self.names.index(name))

    def filter(self, mask: np.ndarray) -> Detections:
        return Detections(self.xyxy[mask], self.confidence[mask], self.class_id[mask], self.names)

    def to_list(self) -> list[dict]:
        return [
            {
                "class": self.class_name(i),
                "confidence": round(float(self.confidence[i]), 3),
                "box": [round(float(v), 1) for v in self.xyxy[i]],
            }
            for i in range(len(self))
        ]
