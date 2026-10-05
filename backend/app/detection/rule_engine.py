"""Central, config-driven rule engine - the entry point other members import.

    from app.detection.rule_engine import analyze_text
    findings = analyze_text(page_text, page="checkout")     # list of contract-shaped dicts

Flow:  text -> preprocess -> detectors (per DETECTORS) -> config (name/severity/recommendation)
       -> Finding (shared contract) -> de-duplicate
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Union

from ..nlp.preprocessing import normalize_text, segment_text
from .evidence_engine import build_evidence
from .models import DEFAULT_STATUS, Finding
from .pattern_detector import DETECTORS
from .rules_config import RULES_CONFIG, get_rule

# Every registered detector must have a rule in the config.
_unknown = set(DETECTORS) - set(RULES_CONFIG)
if _unknown:
    raise RuntimeError(f"Detectors registered for rules missing from config: {sorted(_unknown)}")


def analyze_findings(text: Union[str, Iterable[str], None], page: Optional[str] = None) -> List[Finding]:
    """Run every registered detector and return Finding objects."""
    if text is None:
        return []
    if not isinstance(text, str):          # tolerate a list of text snippets
        text = "\n".join(str(t) for t in text)

    segments = segment_text(text)
    findings: List[Finding] = []
    seen = set()
    for rule_id, detector in DETECTORS.items():
        cfg = get_rule(rule_id)
        for match in detector(segments):
            key = (rule_id, normalize_text(match.evidence_text))
            if key in seen:                 # same text repeated on a page -> one finding
                continue
            seen.add(key)
            findings.append(
                Finding(
                    rule_id=cfg.rule_id,
                    name=cfg.name,
                    severity=cfg.severity,              # fixed per rule, from config
                    confidence=match.rule_confidence,
                    status=DEFAULT_STATUS,
                    evidence=build_evidence(match.evidence_text, page=page),
                    recommendation=cfg.recommendation,
                    explanation=match.explanation,
                    pattern=cfg.pattern,
                )
            )
    return findings


def analyze_text(text: Union[str, Iterable[str], None], page: Optional[str] = None) -> List[dict]:
    """Primary integration point: page text in, list of finding dicts out."""
    return [f.to_dict() for f in analyze_findings(text, page=page)]


def get_rules_config() -> List[dict]:
    """All five rules (DP01-DP05) as plain dicts, e.g. for the API or the dashboard."""
    return [
        {"rule_id": r.rule_id, "name": r.name, "method": r.method, "severity": r.severity,
         "pattern": r.pattern, "description": r.description, "recommendation": r.recommendation,
         "detector_owner": r.detector_owner, "m2_detects": r.m2_detects}
        for r in RULES_CONFIG.values()
    ]
