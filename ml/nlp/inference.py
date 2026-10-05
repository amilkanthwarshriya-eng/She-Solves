"""Load a fine-tuned DeBERTa model and classify text.  (Needs: torch, transformers)

    from inference import DebertaClassifier
    clf = DebertaClassifier("ml/models/nlp/deberta")
    clf.predict(["Only 2 left!"])  # [{'pattern': 'FALSE_URGENCY', 'confidence': 0.97, 'probabilities': {...}}]
"""
import json
import os
from typing import Dict, List


class DebertaClassifier:
    def __init__(self, model_dir: str, max_len: int = 64, device: str = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model folder not found: {model_dir}")
        self.torch = torch
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().to(self.device).eval()
        with open(os.path.join(model_dir, "labels.json"), encoding="utf-8") as f:
            self.labels: List[str] = json.load(f)

    def predict(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        results: List[Dict] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            for p in probs:
                best = max(range(len(p)), key=lambda k: p[k])
                results.append({"pattern": self.labels[best], "confidence": round(p[best], 4),
                                "probabilities": {l: round(v, 4) for l, v in zip(self.labels, p)}})
        return results