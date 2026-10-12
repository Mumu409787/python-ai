"""只加载训练环境导出的 JSON 数值，避免在 API 环境安装 sklearn 或读取 pickle。"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from .config import get_settings
from .schemas import QuestionCategory

logger = logging.getLogger(__name__)
NAMES = {"auth": "认证", "rate_limit": "限流", "failure": "故障", "other": "其他"}


@lru_cache(maxsize=4)
def _load(path: str, modified: int):
    with open(path, encoding="utf-8") as file:
        model = json.load(file)
    if model["formatVersion"] != 1 or model["ngramRange"] != [1, 3]:
        raise ValueError("不支持的分类模型格式")
    size = len(model["vocabulary"])
    if sorted(model["vocabulary"].values()) != list(range(size)):
        raise ValueError("无效词表")
    coefficients = np.asarray(model["coefficients"])
    if coefficients.shape != (len(model["classes"]), size):
        raise ValueError("无效分类系数")
    for name, shape in (("idf", (size,)), ("intercepts", (len(model["classes"]),))):
        values = np.asarray(model[name], dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError("无效分类参数")
    if not np.isfinite(coefficients).all() or not 0 <= model["threshold"] <= 1:
        raise ValueError("无效分类参数")
    return model


def classify(question: str) -> QuestionCategory:
    s = get_settings()
    if not s.classifier_enabled:
        return QuestionCategory(status="disabled")
    try:
        path = s.classifier_model_path
        model = _load(str(path), path.stat().st_mtime_ns)
        text = re.sub(r"\s\s+", " ", question.lower())
        counts = Counter(text[i : i + n] for n in range(1, 4) for i in range(len(text) - n + 1))
        vector = np.zeros(len(model["idf"]))
        for term, count in counts.items():
            index = model["vocabulary"].get(term)
            if index is not None:
                vector[index] = (1 + math.log(count)) * model["idf"][index]
        norm = np.linalg.norm(vector)
        if norm == 0:
            return QuestionCategory(status="low_confidence", confidence=0.0)
        vector /= norm
        scores = np.asarray(model["coefficients"]) @ vector + np.asarray(model["intercepts"])
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        best = int(probs.argmax())
        confidence = float(probs[best])
        label = model["classes"][best]
        if confidence < model["threshold"]:
            return QuestionCategory(status="low_confidence", confidence=confidence)
        return QuestionCategory(
            status="classified", label=label, name=NAMES[label], confidence=confidence
        )
    except Exception as exc:
        logger.warning("问题分类不可用：%s", type(exc).__name__)
        return QuestionCategory(status="unavailable")
