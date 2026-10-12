import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_trained_classic_service_and_catalog_mismatch(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from decisionbench.datasets import load_catalog
    from services.server import create_app

    root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "services.train_classic",
            "--train",
            str(root / "data/fixtures/train.jsonl"),
            "--catalog",
            str(root / "data/fixtures/catalog.json"),
            "--out",
            str(model_path),
        ],
        check=True,
        capture_output=True,
        env=os.environ,
    )
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    catalog = load_catalog(root / "data/fixtures/catalog.json")
    with TestClient(create_app("classic")) as client:
        assert client.get("/health").json()["ready"]
        payload = {
            "sample_id": "new-id",
            "text": "Cancel my alarm",
            "catalog": {"version": catalog.version, "labels": catalog.labels},
        }
        result = client.post("/predict", json=payload)
        assert result.status_code == 200
        assert result.json()["label"] == "alarm_cancel"
        assert result.json()["confidence"] is None
        assert result.json()["score_type"] == "svm_margin_uncalibrated"
        payload["catalog"]["labels"]["music_play"] = "Play music"
        assert client.post("/predict", json=payload).status_code == 409
        assert "decisionbench_predictions_total" in client.get("/metrics").text
