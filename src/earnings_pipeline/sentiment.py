from typing import Dict

import pandas as pd
from tqdm import tqdm

from .config import FINBERT_MODEL


def role_bucket(role_line: str) -> str:
    s = (role_line or "").lower()
    if "analyst" in s:
        return "analyst"
    if "chief executive" in s or " ceo" in s or "chair" in s or "president" in s:
        return "ceo"
    if "cfo" in s or "chief financial" in s or "treasurer" in s:
        return "cfo"
    if "investor relations" in s or "ir" in s:
        return "ir"
    if "operator" in s:
        return "operator"
    if "executive" in s:
        return "exec_other"
    return "other"


def get_finbert_pipeline():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=FINBERT_MODEL, truncation=True)


def finbert_to_score(label: str, score: float) -> float:
    lab = (label or "").lower()
    if "positive" in lab:
        return float(score)
    if "negative" in lab:
        return -float(score)
    return 0.0


def _batched(iterable, n=16):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_speaker_sentiment_records(transcripts, finbert_pipe=None, max_chars=1800):
    if finbert_pipe is None:
        finbert_pipe = get_finbert_pipeline()

    rows = []
    for t in tqdm(transcripts, desc="finbert_speaker_sent"):
        chunks = []

        for b in t.prepared:
            role = b.get("role", "")
            text = (b.get("text") or "")[:max_chars]
            if text.strip():
                chunks.append(
                    {
                        "ticker": t.ticker,
                        "quarter": t.quarter,
                        "call_date": t.call_date,
                        "segment": "prepared",
                        "role_bucket": role_bucket(role),
                        "text": text,
                    }
                )

        for qa in t.qa:
            q_text = (qa.get("question") or "")[:max_chars]
            a_text = (qa.get("answer") or "")[:max_chars]
            q_role = qa.get("q_role") or ""
            a_role = qa.get("a_role") or ""
            if q_text.strip():
                chunks.append(
                    {
                        "ticker": t.ticker,
                        "quarter": t.quarter,
                        "call_date": t.call_date,
                        "segment": "qa_question",
                        "role_bucket": role_bucket(q_role),
                        "text": q_text,
                    }
                )
            if a_text.strip():
                chunks.append(
                    {
                        "ticker": t.ticker,
                        "quarter": t.quarter,
                        "call_date": t.call_date,
                        "segment": "qa_answer",
                        "role_bucket": role_bucket(a_role),
                        "text": a_text,
                    }
                )

        if not chunks:
            rows.append(
                {
                    "ticker": t.ticker,
                    "quarter": t.quarter,
                    "call_date": t.call_date,
                    "sent_overall_finbert": 0.0,
                    "sent_ceo": 0.0,
                    "sent_cfo": 0.0,
                    "sent_analyst": 0.0,
                    "sent_prepared": 0.0,
                    "sent_qa_answer": 0.0,
                    "prepared_qa_slippage": 0.0,
                }
            )
            continue

        preds = []
        for batch in _batched(chunks, n=12):
            out = finbert_pipe([x["text"] for x in batch])
            for m, y in zip(batch, out):
                preds.append({**m, "sent": finbert_to_score(y["label"], y["score"])})

        p = pd.DataFrame(preds)

        def _mean(mask):
            s = p.loc[mask, "sent"]
            return float(s.mean()) if len(s) else 0.0

        sent_prepared = _mean(p["segment"] == "prepared")
        sent_qa_answer = _mean(p["segment"] == "qa_answer")

        rows.append(
            {
                "ticker": t.ticker,
                "quarter": t.quarter,
                "call_date": t.call_date,
                "sent_overall_finbert": _mean(p.index == p.index),
                "sent_ceo": _mean(p["role_bucket"] == "ceo"),
                "sent_cfo": _mean(p["role_bucket"] == "cfo"),
                "sent_analyst": _mean(p["role_bucket"] == "analyst"),
                "sent_prepared": sent_prepared,
                "sent_qa_answer": sent_qa_answer,
                "prepared_qa_slippage": sent_prepared - sent_qa_answer,
            }
        )

    return pd.DataFrame(rows)
