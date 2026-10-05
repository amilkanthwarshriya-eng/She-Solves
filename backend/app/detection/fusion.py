"""Rule confidence + DeBERTa confidence -> one final confidence (Phase 3).

For ONE pattern on ONE sentence:

  rules hit + model agrees      conf = 0.4*rule + 0.6*model
  rules hit + model disagrees   conf = 0.4*rule + 0.6*(model's probability for that pattern)  (lowered)
  rules miss + model flags it   only if model conf >= 0.90, conf = 0.9 * model     (model_only)
  no model available            conf = rule confidence                             (fallback)

All numbers are constants below, so they are easy to explain and to tune.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

RULE_WEIGHT = 0.4
MODEL_WEIGHT = 0.6
MODEL_ONLY_MIN_CONFIDENCE = 0.90
MODEL_ONLY_DISCOUNT = 0.90
MAX_CONFIDENCE = 0.99

SOURCE_RULES = "rules"
SOURCE_AGREE = "rules+model"
SOURCE_DISAGREE = "rules (model disagrees)"
SOURCE_MODEL_ONLY = "model_only"


@dataclass(frozen=True)
class FusedDecision:
    confidence: float
    source: str
    rule_confidence: Optional[float]
    model_confidence: Optional[float]   # model's probability for THIS pattern


def fuse(pattern: str, rule_conf: Optional[float], model: Optional[Dict]) -> Optional[FusedDecision]:
    """Decide whether `pattern` is reported and with what confidence. None = not reported."""
    if model is None:
        if rule_conf is None:
            return None
        return FusedDecision(round(rule_conf, 2), SOURCE_RULES, rule_conf, None)

    top, top_conf = model["pattern"], float(model["confidence"])
    probs = model.get("probabilities", {})
    model_p = top_conf if top == pattern else float(probs.get(pattern, 0.0))

    if rule_conf is not None:
        conf = RULE_WEIGHT * rule_conf + MODEL_WEIGHT * model_p
        source = SOURCE_AGREE if top == pattern else SOURCE_DISAGREE
        return FusedDecision(round(min(conf, MAX_CONFIDENCE), 2), source, rule_conf, round(model_p, 4))

    if top == pattern and top_conf >= MODEL_ONLY_MIN_CONFIDENCE:
        conf = top_conf * MODEL_ONLY_DISCOUNT
        return FusedDecision(round(min(conf, MAX_CONFIDENCE), 2), SOURCE_MODEL_ONLY, None, round(top_conf, 4))
    return None
