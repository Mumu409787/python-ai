"""CPU 问题主题分类：固定数据划分、验证集选阈值、独立测试、导出安全 JSON。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent


def load_data():
    source = ROOT / "data" / "questions.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    seen, group_splits = set(), {}
    for row in rows:
        normalized = re.sub(r"[\W_]+", "", row["text"].lower())
        if normalized in seen:
            raise ValueError(f"重复问题：{row['id']}")
        seen.add(normalized)
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("无效数据划分")
        previous = group_splits.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError("同组问题不能跨数据集")
    return rows, hashlib.sha256(source.read_bytes()).hexdigest()


def main():
    rows, fingerprint = load_data()
    subsets = {
        split: [r for r in rows if r["split"] == split] for split in ("train", "validation", "test")
    }
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)),
            ("classifier", LogisticRegression(C=4.0, max_iter=1000, random_state=42)),
        ]
    )
    train = subsets["train"]
    model.fit([r["text"] for r in train], [r["label"] for r in train])
    validation = subsets["validation"]
    probs = model.predict_proba([r["text"] for r in validation])
    labels = model.classes_.tolist()
    predictions = np.asarray(labels)[probs.argmax(axis=1)]
    gold = np.asarray([r["label"] for r in validation])
    choices = []
    for threshold in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]:
        accepted = probs.max(axis=1) >= threshold
        count = int(accepted.sum())
        accuracy = float((predictions[accepted] == gold[accepted]).mean()) if count else None
        choices.append({"threshold": threshold, "accepted": count, "accuracy": accuracy})
    qualifying = [x for x in choices if x["accepted"] and x["accuracy"] >= 0.9]
    threshold = max(qualifying, key=lambda x: x["accepted"])["threshold"] if qualifying else 0.8
    tfidf, classifier = model.named_steps["tfidf"], model.named_steps["classifier"]
    artifact = {
        "formatVersion": 1,
        "modelType": "char_tfidf_logistic_regression",
        "dataSha256": fingerprint,
        "classes": labels,
        "threshold": threshold,
        "ngramRange": [1, 3],
        "sublinearTf": True,
        "vocabulary": tfidf.vocabulary_,
        "idf": tfidf.idf_.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(),
    }
    target = ROOT / "artifacts"
    target.mkdir(exist_ok=True)
    (target / "question_classifier.json").write_text(
        json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(model, target / "question_classifier.joblib")
    test = subsets["test"]
    test_texts, y = [r["text"] for r in test], [r["label"] for r in test]
    predicted = model.predict(test_texts)
    confidence = model.predict_proba(test_texts).max(axis=1)
    accepted = confidence >= threshold
    metrics = classification_report(y, predicted, labels=labels, output_dict=True, zero_division=0)
    report = {
        "dataSha256": fingerprint,
        "seed": 42,
        "datasetOrigin": "手工编写的教学样本，非生产数据",
        "sizes": {s: dict(Counter(r["label"] for r in data)) for s, data in subsets.items()},
        "validationThresholdSearch": choices,
        "selectedThreshold": threshold,
        "testAccuracy": accuracy_score(y, predicted),
        "testMacroF1": metrics["macro avg"]["f1-score"],
        "testCoverage": float(accepted.mean()),
        "testAcceptedAccuracy": float(np.asarray(y)[accepted].__eq__(predicted[accepted]).mean())
        if accepted.any()
        else None,
        "classificationReport": metrics,
        "confusionMatrix": confusion_matrix(y, predicted, labels=labels).tolist(),
        "predictions": [
            {
                "id": r["id"],
                "text": r["text"],
                "expected": r["label"],
                "predicted": str(p),
                "confidence": float(c),
                "accepted": bool(c >= threshold),
            }
            for r, p, c in zip(test, predicted, confidence, strict=True)
        ],
    }
    (target / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "sizes",
                    "selectedThreshold",
                    "testAccuracy",
                    "testMacroF1",
                    "testCoverage",
                    "testAcceptedAccuracy",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
