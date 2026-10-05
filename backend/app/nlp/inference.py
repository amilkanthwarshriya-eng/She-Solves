"""Text -> {pattern, confidence}: the official M2 output (plan, section 4).

PHASE 1 (current): rule baseline only  ->  "source": "rules".
PHASE 2 (not built yet): DeBERTa-v3-base classifier (nlp/classifier.py).
PHASE 3 (not built yet): fuse rule confidence + DeBERTa confidence.
"""
from __future__ import annotations

from typing import Dict, Union

from ..detection.pattern_detector import DETECTORS
from ..detection.rules_config import PATTERN_NONE
from .preprocessing import segment_text


def classify_text(text: str) -> Dict[str, Union[str, float]]:
    """Return the strongest pattern found in `text`, e.g.
    {"pattern": "CONFIRM_SHAMING", "confidence": 0.95, "source": "rules"}."""
    segments = segment_text(text or "")
    best = None
    for detector in DETECTORS.values():
        for match in detector(segments):
            if best is None or match.rule_confidence > best.rule_confidence:
                best = match
    if best is None:
        return {"pattern": PATTERN_NONE, "confidence": 0.0, "source": "rules"}
    return {"pattern": best.pattern, "confidence": best.rule_confidence, "source": "rules"}
