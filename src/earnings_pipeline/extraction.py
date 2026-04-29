import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from requests.exceptions import ReadTimeout

from .config import (
    OLLAMA_HOST,
    OLLAMA_NUM_CTX,
    PRIMARY_MODEL,
    SECONDARY_MODEL,
    BACKEND,
    ANTHROPIC_API_KEY,
    ANTHROPIC_PRIMARY_MODEL,
    ANTHROPIC_SECONDARY_MODEL,
)


def salvage_json(raw: str) -> dict:
    s = raw or ""
    s = re.sub(r"```(?:json)?", "", s)
    s = re.sub(r"<think>.*?(</think>|$)", "", s, flags=re.DOTALL)
    lo, hi = s.find("{"), s.rfind("}")
    if lo >= 0 and hi > lo:
        s = s[lo : hi + 1]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = s.replace("True", "true").replace("False", "false").replace("None", "null")
    try:
        return json.loads(s)
    except Exception:
        return {}


def transcript_text_for_prompt(t, max_chars=90000):
    parts = [f"[PREPARED - {b['role']}]\n{b['text']}" for b in t.prepared]
    for qa in t.qa:
        parts.append(f"[Q - {qa['q_role']}] {qa['question']}\n[A - {qa['a_role']}] {qa['answer'] or ''}")
    return "\n\n".join(parts)[:max_chars]


EXTRACT_PROMPT_V2 = """You are a finance extraction engine.
Return STRICT JSON only (no markdown), with exactly these keys:
{{
    \"overall_sentiment\": <float in [-1,1]>,
    \"sentiment_bucket\": <\"very_bearish\"|\"bearish\"|\"neutral\"|\"bullish\"|\"very_bullish\">,
    \"guidance_direction\": <\"raised\"|\"reaffirmed\"|\"lowered\"|\"mixed\"|\"none\">,
    \"guidance_confidence\": <float in [0,1]>,
    \"wins\": [
        {{\"label\": <short string>, \"severity\": <1-5>, \"evidence\": <short direct quote>}}
    ],
    \"risks\": [
        {{\"label\": <short string>, \"severity\": <1-5>, \"evidence\": <short direct quote>}}
    ],
    \"themes\": [<short tags>]
}}
Rules:
- Keep max 5 wins and max 5 risks.
- Evidence must be directly grounded in transcript text.
- If uncertain, lower confidence instead of inventing.

TRANSCRIPT:
{transcript}
"""


def _llm_call_anthropic(prompt: str, model: str, api_key: str, max_retries: int = 3) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY in config.py or as an env var."
        )
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    # Discover models available to this key so we can avoid stale model IDs.
    discovered_models = []
    try:
        r_models = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=20,
        )
        if r_models.ok:
            body = r_models.json()
            discovered_models = [m.get("id", "") for m in body.get("data", []) if isinstance(m, dict)]
    except Exception:
        discovered_models = []

    # Prefer requested model, then Haiku, then Sonnet. Never use Opus.
    discovered_non_opus = [m for m in discovered_models if m and "opus" not in m.lower()]
    discovered_haiku = [m for m in discovered_non_opus if "haiku" in m.lower()]
    discovered_sonnet = [m for m in discovered_non_opus if "sonnet" in m.lower()]

    preferred_fallbacks = [
        model,
        os.environ.get("ANTHROPIC_PRIMARY_MODEL", "").strip(),
        os.environ.get("ANTHROPIC_SECONDARY_MODEL", "").strip(),
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-6",
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
    ] + discovered_haiku + discovered_sonnet
    candidates = []
    seen = set()
    for m in preferred_fallbacks:
        if not m:
            continue
        if "opus" in m.lower():
            continue
        if m in seen:
            continue
        seen.add(m)
        candidates.append(m)

    last_status = None
    last_body = ""
    last_err = None
    for candidate_model in candidates:
        payload = {
            "model": candidate_model,
            "max_tokens": 2200,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }
        for _ in range(max_retries):
            try:
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                if r.ok:
                    payload = r.json()
                    content = payload.get("content", [])
                    text_parts = [x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text"]
                    return "\n".join([x for x in text_parts if x])

                last_status = r.status_code
                last_body = (r.text or "")[:500]
                # Model-name or endpoint mismatch: try next safe candidate model.
                if r.status_code in (400, 404):
                    break
                # Transient/server/rate errors: retry same candidate.
                if r.status_code in (408, 409, 429, 500, 502, 503, 504):
                    continue
                # Other statuses are unlikely to recover by retrying or fallback.
                break
            except Exception as e:
                last_err = e

    if last_err is not None and last_status is None:
        raise last_err
    raise RuntimeError(
        f"Anthropic request failed status={last_status} body={last_body}"
    )


def llm_call_model(
    prompt: str,
    model: str,
    num_ctx: int = OLLAMA_NUM_CTX,
    timeout: int = 1800,
    max_retries: int = 3,
) -> str:
    runtime_backend = os.environ.get("EARNINGS_LLM_BACKEND", BACKEND).strip().lower()

    # Route to Anthropic if backend is set
    if runtime_backend == "anthropic":
        primary_rt = os.environ.get("ANTHROPIC_PRIMARY_MODEL", ANTHROPIC_PRIMARY_MODEL)
        secondary_rt = os.environ.get("ANTHROPIC_SECONDARY_MODEL", ANTHROPIC_SECONDARY_MODEL)
        anthropic_model = primary_rt if model == PRIMARY_MODEL else secondary_rt
        runtime_key = os.environ.get("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
        return _llm_call_anthropic(prompt, anthropic_model, runtime_key, max_retries)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "think": False,
            "num_ctx": num_ctx,
            "temperature": 0.1,
            "num_predict": 2200,
        },
    }

    last_err = None
    for _ in range(max_retries):
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json().get("response", "")
        except ReadTimeout as e:
            last_err = e
            continue

    raise last_err if last_err else RuntimeError("llm_call_model failed without explicit exception")


def _safe_extract_labels(items):
    out = []
    for x in (items or []):
        if isinstance(x, dict):
            lab = str(x.get("label", "")).strip().lower()
        else:
            lab = str(x).strip().lower()
        if lab:
            out.append(lab)
    return set(out)


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extraction_agreement(primary: dict, secondary: dict) -> dict:
    wins_j = _jaccard(_safe_extract_labels(primary.get("wins")), _safe_extract_labels(secondary.get("wins")))
    risks_j = _jaccard(_safe_extract_labels(primary.get("risks")), _safe_extract_labels(secondary.get("risks")))
    themes_j = _jaccard(_safe_extract_labels(primary.get("themes")), _safe_extract_labels(secondary.get("themes")))
    guide_eq = float(str(primary.get("guidance_direction", "")).lower() == str(secondary.get("guidance_direction", "")).lower())
    sentiment_gap = abs(float(primary.get("overall_sentiment", 0.0) or 0.0) - float(secondary.get("overall_sentiment", 0.0) or 0.0))
    agreement = 0.30 * wins_j + 0.30 * risks_j + 0.25 * themes_j + 0.15 * guide_eq
    confidence = float(max(0.0, min(1.0, agreement * (1.0 - min(1.0, sentiment_gap)))))
    return {
        "wins_jaccard": wins_j,
        "risks_jaccard": risks_j,
        "themes_jaccard": themes_j,
        "guidance_match": guide_eq,
        "sentiment_gap": sentiment_gap,
        "agreement_score": agreement,
        "agreement_confidence": confidence,
    }


def normalize_event_obj(obj: dict) -> dict:
    obj = obj or {}
    obj.setdefault("overall_sentiment", 0.0)
    obj.setdefault("sentiment_bucket", "neutral")
    obj.setdefault("guidance_direction", "none")
    obj.setdefault("guidance_confidence", 0.0)
    obj.setdefault("wins", [])
    obj.setdefault("risks", [])
    obj.setdefault("themes", [])

    def _clean_items(items):
        cleaned = []
        for x in (items or [])[:5]:
            if isinstance(x, dict):
                cleaned.append({
                    "label": str(x.get("label", "")).strip(),
                    "severity": int(x.get("severity", 3) or 3),
                    "evidence": str(x.get("evidence", "")).strip(),
                })
            else:
                cleaned.append({"label": str(x).strip(), "severity": 3, "evidence": ""})
        return cleaned

    obj["wins"] = _clean_items(obj.get("wins"))
    obj["risks"] = _clean_items(obj.get("risks"))
    obj["themes"] = [str(t).strip().lower() for t in (obj.get("themes") or []) if str(t).strip()]
    return obj


def extract_events_dual_llm(t, extractions_dir: Path, primary_model=PRIMARY_MODEL, secondary_model=SECONDARY_MODEL, force=False):
    dual_extract_dir = extractions_dir / "dual_llm"
    dual_extract_dir.mkdir(parents=True, exist_ok=True)

    key = f"{t.ticker}_{t.quarter}"
    cache = dual_extract_dir / f"{key}.json"
    if cache.exists() and not force:
        try:
            return json.loads(cache.read_text())
        except Exception:
            # Corrupt cache should not block progress.
            pass

    prompt = EXTRACT_PROMPT_V2.format(transcript=transcript_text_for_prompt(t))
    raw_primary = llm_call_model(prompt, primary_model)
    # If both models are the same, avoid paying for a duplicate call.
    if primary_model == secondary_model:
        raw_secondary = raw_primary
    else:
        raw_secondary = llm_call_model(prompt, secondary_model)

    p_obj = normalize_event_obj(salvage_json(raw_primary))
    s_obj = normalize_event_obj(salvage_json(raw_secondary))
    agree = extraction_agreement(p_obj, s_obj)

    out = {
        "_ticker": t.ticker,
        "_quarter": t.quarter,
        "_call_date": t.call_date,
        "backend": os.environ.get("EARNINGS_LLM_BACKEND", BACKEND),
        "primary_model": primary_model,
        "secondary_model": secondary_model,
        "single_call_mode": bool(primary_model == secondary_model),
        "primary": p_obj,
        "secondary": s_obj,
        "agreement": agree,
        "chosen": p_obj,
    }

    (dual_extract_dir / f"{key}.{primary_model.replace(':','-')}.raw.txt").write_text(raw_primary)
    (dual_extract_dir / f"{key}.{secondary_model.replace(':','-')}.raw.txt").write_text(raw_secondary)
    cache.write_text(json.dumps(out, indent=2))
    return out


def run_dual_extraction_batch(
    transcripts: List,
    extractions_dir: Path,
    primary_model: str = PRIMARY_MODEL,
    secondary_model: str = SECONDARY_MODEL,
    force: bool = False,
    max_new: Optional[int] = None,
    stop_on_error: bool = False,
) -> Dict:
    dual_extract_dir = extractions_dir / "dual_llm"
    dual_extract_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    attempted = 0
    failed = []

    for t in transcripts:
        key = f"{t.ticker}_{t.quarter}"
        cache = dual_extract_dir / f"{key}.json"
        if cache.exists() and not force:
            continue
        if max_new is not None and attempted >= max_new:
            break

        attempted += 1
        try:
            extract_events_dual_llm(
                t=t,
                extractions_dir=extractions_dir,
                primary_model=primary_model,
                secondary_model=secondary_model,
                force=force,
            )
            processed += 1
        except Exception as e:
            failed.append({"ticker": t.ticker, "quarter": t.quarter, "error": str(e)})
            if stop_on_error:
                raise

    cached_total = len(list(dual_extract_dir.glob("*.json")))
    return {
        "attempted": attempted,
        "processed": processed,
        "failed": failed,
        "cached_total": cached_total,
        "failed_count": len(failed),
    }


def load_dual_extractions_df(extractions_dir: Path) -> pd.DataFrame:
    dual_extract_dir = extractions_dir / "dual_llm"
    rows = []
    for p in sorted(dual_extract_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        chosen = obj.get("chosen", {})
        agree = obj.get("agreement", {})
        wins = chosen.get("wins", [])
        risks = chosen.get("risks", [])
        rows.append(
            {
                "ticker": obj.get("_ticker"),
                "quarter": obj.get("_quarter"),
                "call_date": obj.get("_call_date"),
                "sentiment": float(chosen.get("overall_sentiment", 0.0) or 0.0),
                "sentiment_bucket": chosen.get("sentiment_bucket", "neutral"),
                "guidance": chosen.get("guidance_direction", "none"),
                "guidance_confidence": float(chosen.get("guidance_confidence", 0.0) or 0.0),
                "n_wins": len(wins),
                "n_risks": len(risks),
                "wins": wins,
                "risks": risks,
                "themes": chosen.get("themes", []),
                "agreement_score": float(agree.get("agreement_score", 0.0) or 0.0),
                "agreement_confidence": float(agree.get("agreement_confidence", 0.0) or 0.0),
                "sent_gap_models": float(agree.get("sentiment_gap", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)
