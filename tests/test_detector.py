import numpy as np
import pytest

from siteguard.detector import decode_yolo, letterbox, load_detector
from siteguard.labels import label_path, read_labels


def test_letterbox_keeps_aspect_and_centres():
    img = np.zeros((200, 400, 3), np.uint8)
    out, r, (px, py) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert r == pytest.approx(1.6)
    assert px == 0 and py == 160
    assert (out[0, 0] == 114).all()


def _head(boxes_cxcywh, scores, nc=12):
    """Build a fake (1, 4+nc, N) YOLO head output."""
    n = len(boxes_cxcywh)
    out = np.zeros((1, 4 + nc, n), np.float32)
    for i, ((cx, cy, w, h), (cls, s)) in enumerate(zip(boxes_cxcywh, scores, strict=True)):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + cls, i] = s
    return out


def test_decode_threshold_and_nms():
    out = _head(
        [(100, 100, 50, 50), (102, 101, 50, 50), (300, 300, 40, 40), (500, 500, 10, 10)],
        [(5, 0.9), (5, 0.8), (0, 0.7), (0, 0.1)],
    )
    xyxy, scores, cls = decode_yolo(out, conf=0.25, iou=0.5)
    assert len(xyxy) == 2  # duplicate suppressed, low-score dropped
    assert sorted(cls.tolist()) == [0, 5]
    assert np.allclose(xyxy[np.argmax(scores)], [75, 75, 125, 125])


def test_nms_is_class_aware():
    # a helmet box fully inside a Person box must survive NMS
    out = _head([(100, 100, 60, 60), (100, 100, 60, 60)], [(5, 0.9), (0, 0.8)])
    _, _, cls = decode_yolo(out, conf=0.25, iou=0.5)
    assert sorted(cls.tolist()) == [0, 5]


def test_decode_empty():
    xyxy, scores, cls = decode_yolo(np.zeros((1, 16, 10), np.float32), 0.25, 0.5)
    assert xyxy.shape == (0, 4) and len(scores) == 0


def test_load_detector_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_detector(tmp_path / "missing.onnx")
    bad = tmp_path / "model.tflite"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_detector(bad)


def test_read_labels(tmp_path):
    img = tmp_path / "ds" / "images" / "val" / "a.jpg"
    img.parent.mkdir(parents=True)
    lab = label_path(img)
    lab.parent.mkdir(parents=True)
    lab.write_text("5 0.5 0.5 0.2 0.4\n0 0.5 0.32 0.1 0.08\n")
    det = read_labels(img, 1000, 500)
    assert det.class_name(0) == "Person"
    assert np.allclose(det.xyxy[0], [400, 150, 600, 350])
