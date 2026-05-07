"""Integration tests for the FastAPI segmentation endpoint.

These tests mock the model to avoid requiring actual weights.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def make_jpeg_bytes(w: int = 64, h: int = 64) -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    with patch("api.main.BiRefNetModel") as MockModel:
        instance = MagicMock()
        instance.name = "BiRefNetModel"
        # predict returns a float32 64x64 mask
        instance.predict.return_value = np.ones((64, 64), dtype=np.float32) * 0.9
        instance.load.return_value = None
        MockModel.return_value = instance

        from api.main import app, _pipelines
        _pipelines.clear()  # reset between test runs

        with TestClient(app) as c:
            yield c


class TestAPI:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_segment_returns_image(self, client):
        jpeg = make_jpeg_bytes()
        resp = client.post(
            "/segment",
            files={"file": ("test.jpg", jpeg, "image/jpeg")},
            data={"model": "birefnet"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_segment_unsupported_type(self, client):
        resp = client.post(
            "/segment",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 415
