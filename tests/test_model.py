"""Tests against the shipped model artefacts (skipped if they are missing)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from siteguard import ComplianceConfig, evaluate
from siteguard.detector import OnnxDetector
from siteguard.types import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "models" / "siteguard.onnx"
PT = ROOT / "models" / "ppe_yolo11n_640.pt"
SITE = ROOT / "tests" / "assets" / "site.jpg"

pytestmark = pytest.mark.skipif(not ONNX.exists(), reason="models/siteguard.onnx not built")


@pytest.fixture(scope="module")
def onnx_det():
    return OnnxDetector(ONNX)


def test_onnx_metadata(onnx_det):
    assert onnx_det.names == CLASS_NAMES
    assert onnx_det.dynamic and onnx_det.imgsz == 960


def test_site_photo(onnx_det):
    img = cv2.imread(str(SITE))
    det = onnx_det.predict(img)
    assert len(det.where("Person")) >= 2
    rep = evaluate(det, ComplianceConfig.from_file(ROOT / "configs" / "thresholds_960.json"))
    assert len(rep.people) >= 2
    assert sum(rep.counts.values()) == len(rep.people)


def test_blank_image_has_no_workers(onnx_det):
    det = onnx_det.predict(np.full((720, 1280, 3), 127, np.uint8))
    assert len(det.where("Person")) == 0


def test_onnx_matches_pytorch(onnx_det):
    pytest.importorskip("ultralytics")
    if not PT.exists():
        pytest.skip("PyTorch weights not available")
    from siteguard.detector import UltralyticsDetector

    img = cv2.imread(str(SITE))
    a = UltralyticsDetector(PT, imgsz=960).predict(img)
    b = onnx_det.predict(img)
    assert len(a) == len(b)
    order_a = np.lexsort((a.xyxy[:, 0], a.class_id))
    order_b = np.lexsort((b.xyxy[:, 0], b.class_id))
    assert (a.class_id[order_a] == b.class_id[order_b]).all()
    assert np.abs(a.xyxy[order_a] - b.xyxy[order_b]).max() < 2.0  # pixels
    assert np.abs(a.confidence[order_a] - b.confidence[order_b]).max() < 1e-3


def test_metrics_file_is_consistent():
    m = json.loads((ROOT / "models" / "metrics.json").read_text())
    assert 0 < m["test"]["mAP50"] <= 1
    assert m["export"]["parity"]["pytorch_boxes_found_by_onnx"] >= 0.99
    for size in ("640", "960"):
        assert set(m["compliance"][size]) == {"default", "tuned"}
