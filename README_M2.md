# DarkPatternGuard — Member 2 (M2): NLP + Detection Intelligence

**Status: Phase 1 (rule baseline) complete and aligned to the official plan.
Phase 2 (DeBERTa-v3-base) and Phase 3 (rule + model confidence fusion) are NOT built yet.**

## What M2 owns (plan §17)
`backend/app/detection/` · `backend/app/nlp/` · `ml/nlp/`
Output: *text / evidence -> pattern + confidence*, wrapped into the shared finding contract.

## Status vs the plan
| Item | Status |
|---|---|
| Rule engine, config-driven DP01–DP05 (§4) | Done (`rules_config.py`) |
| DP02 False Urgency — language side | Done (rules). DOM-signal input from M1: not yet |
| DP03 Confirm Shaming | Done (rules) |
| Shared JSON contract (§7), M2's fields | Done (`models.py`, `evidence_engine.py`) |
| `{pattern, confidence}` model-level output | Done for rules (`nlp/inference.py`) |
| DeBERTa-v3-base classifier | **Not trained yet.** Backend wrapper `nlp/classifier.py` not written |
| Rule + DeBERTa confidence fusion | **Missing** |
| Labeled dataset (100–200 samples) | Seed version done (180 samples, `ml/nlp/make_dataset.py`). Add real examples |
| Training + evaluation scripts | Written (`ml/nlp/train.py`, `evaluate.py`, `inference.py`). **Training not run yet**, needs Colab/GPU |
| Rule baseline score (test split) | accuracy 0.82, macro-F1 0.79 (rules miss many shaming phrasings) |

## Structure
```
backend/app/
  detection/  rules_config.py   DP01-DP05 config (name, method, severity, recommendation, owner)
              rule_engine.py    analyze_text() - the integration point
              pattern_detector.py  registry rule_id -> detector
              urgency_detector.py, confirm_shaming_detector.py   Phase 1 rule detectors
              evidence_engine.py   evidence object + attach scanner evidence
              confidence.py, models.py
  nlp/        preprocessing.py  inference.py   (classifier.py comes with DeBERTa)
tests/backend/   demo_m2.py
```
ML side: `ml/nlp/{make_dataset,train,evaluate,inference}.py` + `ml/nlp/README.md` (Colab steps).
Not yet created: `backend/app/nlp/classifier.py` and the confidence fusion.

## Run (Windows PowerShell, from the project root)
```powershell
python demo_m2.py
python -m unittest discover -s tests/backend -v
python -m unittest discover -s tests/ml -v
```
No packages needed for Phase 1.

## Integration
**Member 3 (backend):**
```python
from app.detection.rule_engine import analyze_text      # run from backend/
findings = analyze_text(page_text, page="checkout")      # list of dicts
```
Each finding: `rule_id, name, severity, confidence, status, evidence{text,selector,page,screenshot},
recommendation, explanation, pattern`. M3 adds `regulation`, `risk_score`, `risk_level`, top-level fields.

**Member 1 (scanner):** pass visible text, one element per line. After matching a finding to an element:
```python
from app.detection.evidence_engine import attach_scanner_evidence
f = attach_scanner_evidence(f, selector="#decline", page="checkout", screenshot="checkout.png")
```
Status becomes `VERIFIED` only when text + selector + screenshot are all present.

## Team decisions (confirmed)
1. M2 emits `CANDIDATE`; a finding becomes `VERIFIED` once text + selector + screenshot exist.
2. `explanation` and `pattern` are part of every finding.
3. `risk_engine.py` belongs to M3 (M2 does not build it).
4. M1 will send real scanner text after integration; until then M2 tests on sample text.

## Known limitations
English only; fixed phrase lists; flags urgency *language*, not whether scarcity is fake; confidence is a
rule-based heuristic (not a probability); opt-out wording like "I don't want insurance" may be flagged.
