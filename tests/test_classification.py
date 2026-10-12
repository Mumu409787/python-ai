import json

import httpx
import pytest

from python_ai.classification import classify
from python_ai.config import PROJECT_ROOT, get_settings
from python_ai.main import app


def test_export_matches_saved_sklearn_test_probabilities():
    # 这些概率是独立 sklearn 推理生成；验证 API 数值实现与训练模型一致。
    report = json.loads(
        (PROJECT_ROOT / "training/artifacts/evaluation.json").read_text(encoding="utf-8")
    )
    for row in report["predictions"]:
        result = classify(row["text"])
        assert result.confidence == pytest.approx(row["confidence"], abs=1e-10)
        if row["accepted"]:
            assert result.label == row["predicted"]
        else:
            assert result.status == "low_confidence" and result.label is None


def test_missing_model_disabled_and_unknown_input(monkeypatch, tmp_path):
    assert classify("🦄🛸").status == "low_confidence"
    monkeypatch.setattr(get_settings(), "classifier_model_path", tmp_path / "missing.json")
    assert classify("问题").status == "unavailable"
    monkeypatch.setattr(get_settings(), "classifier_enabled", False)
    assert classify("问题").status == "disabled"


def test_corrupt_numeric_model_degrades_without_invalid_response(monkeypatch, tmp_path):
    path = tmp_path / "model.json"
    model = json.loads((PROJECT_ROOT / "training/artifacts/question_classifier.json").read_text())
    model["idf"][0] = float("nan")
    path.write_text(json.dumps(model), encoding="utf-8")
    monkeypatch.setattr(get_settings(), "classifier_model_path", path)
    assert classify("42901").status == "unavailable"


async def test_local_classifier_api_without_llm():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post("/api/chat/classify", json={"message": "访问令牌多久过期"})
        assert response.json()["data"]["label"] == "auth"
        assert (await c.post("/api/chat/classify", json={"message": " "})).json()["code"] == 1001
