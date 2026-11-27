import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from conftest import FakeDetector


@pytest.fixture
def client(monkeypatch, tmp_path):
    fake = FakeDetector()
    model = tmp_path / "siteguard.onnx"
    model.write_bytes(b"fake")
    monkeypatch.setenv("SITEGUARD_MODEL", str(model))
    monkeypatch.setattr(api_main, "load_detector", lambda path, **kw: fake)
    with TestClient(api_main.app) as c:
        c.fake = fake
        yield c


def jpeg(w=640, h=480) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 127, np.uint8))
    return buf.tobytes()


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    m = client.get("/v1/model").json()
    assert "helmet" in m["classes"] and m["input_size"] == 640


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "SiteGuard" in r.text


def test_detect_json(client):
    r = client.post("/v1/detect", files={"file": ("a.jpg", jpeg(), "image/jpeg")}, params={"include_detections": True})
    assert r.status_code == 200
    body = r.json()
    s = body["compliance"]["summary"]
    assert s == {
        "compliant": 1,
        "violation": 1,
        "unverified": 0,
        "people": 2,
        "unassigned_violations": 0,
        "has_violation": True,
    }
    assert body["image"] == {"width": 640, "height": 480}
    assert len(body["detections"]) == 6
    assert "image_base64" not in body


def test_detect_with_image_and_options(client):
    r = client.post(
        "/v1/detect",
        files={"file": ("a.jpg", jpeg(), "image/jpeg")},
        params={"return_image": True, "required": "helmet"},
    )
    body = r.json()
    img = cv2.imdecode(np.frombuffer(base64.b64decode(body["image_base64"]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (480, 640, 3)
    assert set(body["compliance"]["people"][0]["items"]) == {"helmet"}


def test_strict_mode(client):
    r = client.post(
        "/v1/detect",
        files={"file": ("a.jpg", jpeg(), "image/jpeg")},
        params={"required": "helmet,vest,boots", "strict": True},
    )
    assert r.json()["compliance"]["summary"]["violation"] == 2  # boots not visible -> violation


def test_annotated_endpoint(client):
    r = client.post("/v1/detect/annotated", files={"file": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["x-violations"] == "1" and r.headers["x-workers"] == "2"


@pytest.mark.parametrize(
    "files,params,code",
    [
        ({"file": ("a.txt", b"not an image", "text/plain")}, {}, 400),
        ({"file": ("a.jpg", b"", "image/jpeg")}, {}, 400),
        ({"file": ("a.jpg", jpeg(), "image/jpeg")}, {"required": "cape"}, 422),
        ({"file": ("a.jpg", jpeg(), "image/jpeg")}, {"conf": 2}, 422),
        ({}, {}, 422),
    ],
)
def test_bad_requests(client, files, params, code):
    assert client.post("/v1/detect", files=files, params=params).status_code == code


def test_too_large(client, monkeypatch):
    monkeypatch.setattr(api_main, "MAX_BYTES", 100)
    r = client.post("/v1/detect", files={"file": ("a.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 413
