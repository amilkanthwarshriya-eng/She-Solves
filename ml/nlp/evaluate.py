"""Evaluate classifiers on the labeled dataset.

    python ml/nlp/evaluate.py                       # rule baseline (Phase 1), test split
    python ml/nlp/evaluate.py --split all           # rule baseline on every example
    python ml/nlp/evaluate.py --model ml/models/nlp/deberta     # DeBERTa (after training)
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

LABELS = ["FALSE_URGENCY", "CONFIRM_SHAMING", "NONE"]


def load(split: str):
    with open(os.path.join(HERE, "data", f"{split}.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    per_class = {}
    for lab in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                          "support": sum(1 for t in y_true if t == lab)}
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(LABELS)
    confusion = {t: {p: sum(1 for a, b in zip(y_true, y_pred) if a == t and b == p) for p in LABELS} for t in LABELS}
    return {"accuracy": round(accuracy, 3), "macro_f1": round(macro_f1, 3),
            "per_class": per_class, "confusion": confusion}


def predict_rules(texts: List[str]) -> List[str]:
    from app.nlp.inference import classify_text
    return [classify_text(t)["pattern"] for t in texts]


def predict_model(texts: List[str], model_dir: str) -> List[str]:
    sys.path.insert(0, HERE)
    from inference import DebertaClassifier
    clf = DebertaClassifier(model_dir)
    return [r["pattern"] for r in clf.predict(texts)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--model", default=None, help="folder of a fine-tuned DeBERTa model")
    args = ap.parse_args()

    texts, y_true = load(args.split)
    name = "deberta" if args.model else "rules"
    y_pred = predict_model(texts, args.model) if args.model else predict_rules(texts)
    m = metrics(y_true, y_pred)

    print(f"\n== {name} on '{args.split}' split ({len(texts)} samples) ==")
    print(f"accuracy {m['accuracy']:.3f}   macro-F1 {m['macro_f1']:.3f}\n")
    print(f"{'class':<16}{'precision':>10}{'recall':>9}{'f1':>7}{'support':>9}")
    for lab, c in m["per_class"].items():
        print(f"{lab:<16}{c['precision']:>10.3f}{c['recall']:>9.3f}{c['f1']:>7.3f}{c['support']:>9}")
    print("\nconfusion (rows = true, columns = predicted):")
    print(" " * 16 + "".join(f"{l[:10]:>12}" for l in LABELS))
    for t in LABELS:
        print(f"{t:<16}" + "".join(f"{m['confusion'][t][p]:>12}" for p in LABELS))
    errors = [(t, a, b) for t, a, b in zip(texts, y_true, y_pred) if a != b]
    print(f"\nmistakes: {len(errors)}")
    for t, a, b in errors:
        print(f"  true={a:<15} pred={b:<15} {t}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{name}_{args.split}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({**m, "samples": len(texts), "mistakes": [list(e) for e in errors]}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
