"""Text -> {pattern, confidence}: the official M2 output (plan, section 4).

    classify_text("Only 2 left!")                          # rule baseline
    classify_text("Only 2 left!", classifier=clf)          # rules + DeBERTa, fused
"""
from __future__ import annotations

from typing import Dict, Optional, Union

from ..detection.fusion import fuse
from ..detection.pattern_detector import DETECTORS
from ..detection.rules_config import PATTERN_NONE
from .preprocessing import segment_text


def classify_text(text: str, classifier=None, model_result: Optional[dict] = None) -> Dict[str, Union[str, float]]:
    """Strongest pattern in `text`, e.g. {"pattern": "CONFIRM_SHAMING", "confidence": 0.96, "source": "rules+model"}.

    Pass `classifier` (a DebertaClassifier) or a precomputed `model_result` to use the model.
    """
    segments = segment_text(text or "")
    rule_conf: Dict[str, float] = {}
    for detector in DETECTORS.values():
        for match in detector(segments):
            rule_conf[match.pattern] = max(match.rule_confidence, rule_conf.get(match.pattern, 0.0))

    if model_result is None and classifier is not None and text:
        model_result = classifier.predict([text])[0]

    patterns = set(rule_conf)
    if model_result and model_result["pattern"] != PATTERN_NONE:
        patterns.add(model_result["pattern"])

    best = None
    best_pattern = PATTERN_NONE
    for p in patterns:
        decision = fuse(p, rule_conf.get(p), model_result)
        if decision and (best is None or decision.confidence > best.confidence):
            best, best_pattern = decision, p
    if best is None:
        return {"pattern": PATTERN_NONE, "confidence": 0.0, "source": "model" if model_result else "rules"}
    return {"pattern": best_pattern, "confidence": best.confidence, "source": best.source}
