import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from siteguard.types import CLASS_NAMES, Detections  # noqa: E402

N = CLASS_NAMES.index


def make_det(*objs) -> Detections:
    """objs: (class_name, x1, y1, x2, y2[, conf])"""
    if not objs:
        return Detections.empty()
    boxes = [o[1:5] for o in objs]
    conf = [o[5] if len(o) > 5 else 0.9 for o in objs]
    return Detections(np.array(boxes, float), np.array(conf), np.array([N(o[0]) for o in objs]), CLASS_NAMES)


class FakeDetector:
    """Returns a fixed set of detections - lets API tests run without a model."""

    names = CLASS_NAMES
    imgsz = 640

    def __init__(self, det: Detections | None = None):
        self.det = (
            det
            if det is not None
            else make_det(
                ("Person", 100, 100, 200, 400),
                ("helmet", 120, 95, 180, 140),
                ("vest", 110, 170, 190, 280),
                ("Person", 300, 100, 400, 400),
                ("no_helmet", 320, 100, 380, 150),
                ("vest", 310, 170, 390, 280),
            )
        )
        self.calls = 0

    def predict(self, image_bgr, conf=0.25, iou=0.5):
        self.calls += 1
        return self.det.filter(self.det.confidence >= conf)


@pytest.fixture
def fake_detector():
    return FakeDetector()
