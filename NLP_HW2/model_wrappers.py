"""model_wrappers.py — picklable wrapper classes shared by finalize_inference.py and app.py.

Both scripts import from here so joblib can resolve the classes when loading
the inference bundle.
"""
from __future__ import annotations
import numpy as np


class _SetFitWrapper:
    """Picklable wrapper around a SetFit model."""
    def __init__(self, ckpt_path: str):
        from setfit import SetFitModel
        self._model = SetFitModel.from_pretrained(ckpt_path)
        self._ckpt_path = ckpt_path

    def __call__(self, texts):
        p = self._model.predict_proba(list(texts))
        try:
            arr = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
        except Exception:
            arr = np.asarray(p)
        return arr[:, 1] if arr.ndim == 2 and arr.shape[1] == 2 else arr


class _FinBERTWrapper:
    """Picklable wrapper around a FinBERT model."""
    def __init__(self, ckpt_path: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._ckpt_path = ckpt_path
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok   = AutoTokenizer.from_pretrained(ckpt_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(self._device)
        self._model.eval()

    def __call__(self, texts):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = self._tok(list(texts[i:i + 32]), return_tensors="pt",
                                truncation=True, padding=True, max_length=96).to(self._device)
                logits = self._model(**enc).logits
                out.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist())
        return np.asarray(out)
