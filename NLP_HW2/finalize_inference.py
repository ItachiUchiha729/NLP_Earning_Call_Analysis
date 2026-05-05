"""finalize_inference.py — build a self-contained inference bundle for the GUI.

The training notebook saves `models/bp_best.joblib` with the winner name,
threshold, and test metrics.  This script reads the cached training data,
refits every member model on the full train+val pool, and writes a single
`models/bp_inference.joblib` that the Streamlit GUI loads at startup.

Run once after the notebook finishes:
    python finalize_inference.py
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------
# Paths  (ROOT = directory that contains this script = NLP HW2 project root)
# --------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent
CACHE_DIR  = ROOT / "cache"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)

BUNDLE_IN  = MODELS_DIR / "bp_best.joblib"
BUNDLE_OUT = MODELS_DIR / "bp_inference.joblib"

# --------------------------------------------------------------------------
# Regex feature pack — must match the notebook EXACTLY
# --------------------------------------------------------------------------
FEATURE_PATTERNS = [
    ("starts_with_operator", r"^\s*(operator|moderator)[\s:.,]"),
    ("mute_lines",           r"\b(lines? (have been|are) placed on mute|on mute|in listen[- ]only mode)\b"),
    ("queue_phrase",         r"\b(press\s*\*?\s*[1-9]|press star|in the queue|withdraw your question|ask a question)\b"),
    ("recording_phrase",     r"\b(call is being recorded|today's call is being recorded)\b"),
    ("welcome_phrase",       r"\b(welcome to (the|today's|our)|good (morning|afternoon|evening),?\s*(everyone|all|ladies))"),
    ("conclude_phrase",      r"\b(this concludes|thank you for (joining|participating)|that concludes our call)"),
    ("turn_call_over",       r"\b(turn (the call|it) (over|back) to|hand (it|the call) (over|back) to)"),
    ("forward_looking",      r"\bforward[- ]looking statements?\b"),
    ("safe_harbor",          r"\b(safe harbor|private securities litigation reform act|may differ materially|risks and uncertainties)\b"),
    ("non_gaap",             r"\b(non[- ]?gaap|gaap (to|and non)|reconciliation (of|to) gaap)\b"),
    ("sec_filings",          r"\b(10[- ]?[KQ]|sec (filings?|filing)|annual report|proxy statement)\b"),
    ("thanks_for_question",  r"\b(thanks?|thank you)( so much)?( ,)? for (taking|having|the question|joining)"),
    ("generic_greeting",     r"^\s*(hi|hey|hello|good (morning|afternoon|evening))[, ]"),
    ("name_intro",           r"\bthis is \w+ (from|at|with|on for)\s+[A-Z]\w+"),
    ("analyst_firm",         r"\b(goldman( sachs)?|jp\s*morgan|morgan stanley|wells fargo|bank of america|citi(group)?|barclays|deutsche|ubs|credit suisse|jefferies|cowen|raymond james|piper sandler|wedbush|kbw|stifel|baird|evercore|guggenheim|bernstein|oppenheimer|truist|td (cowen|securities))\b"),
    ("has_dollar",           r"\$\s?\d"),
    ("has_percent",          r"\d+(\.\d+)?\s?%|\bpercent\b"),
    ("has_bps",              r"\bbasis points?\b|\bbps\b"),
    ("has_million_billion",  r"\b(million|billion|trillion|mn|bn)\b"),
    ("has_year",             r"\b(20[12]\d|fiscal\s+\d{4}|fy\s?\d{4})\b"),
    ("has_quarter",          r"\b(q[1-4]\b|first quarter|second quarter|third quarter|fourth quarter|fy\d+q\d)\b"),
    ("guidance_word",        r"\b(guidance|guide|outlook|expect|anticipate|forecast|raise|raising|raised|reaffirm|reiterate)\b"),
    ("segment_word",         r"\b(segment|division|business unit|product line|geography|vertical|category)\b"),
    ("margin_word",          r"\b(margin|operating income|gross margin|EBITDA|free cash flow|FCF|EPS|earnings per share)\b"),
]
_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in FEATURE_PATTERNS]


def regex_features(sentence: str) -> dict:
    """Identical to the notebook's feature extractor."""
    s = sentence
    feats = {name: int(bool(rx.search(s))) for name, rx in _COMPILED}
    n_chars = len(s)
    words = s.split()
    n_words = len(words)
    n_digits = sum(c.isdigit() for c in s)
    n_alpha  = sum(c.isalpha() for c in s)
    n_upper  = sum(c.isupper() for c in s)
    feats.update({
        "len_chars":          n_chars,
        "len_words":          n_words,
        "digit_ratio":        n_digits / max(1, n_chars),
        "uppercase_ratio":    n_upper / max(1, n_alpha),
        "ends_with_question": int(s.rstrip().endswith("?")),
        "starts_with_number": int(bool(re.match(r"^\s*\d", s))),
        "first_person_count": sum(1 for w in words if w.lower() in {"we", "our", "us", "i"}),
        "modal_count":        sum(1 for w in words if w.lower() in
                                   {"will", "would", "expect", "expects", "expected",
                                    "plan", "plans", "intend", "may", "might", "could", "should"}),
        "proper_noun_run":    int(bool(re.search(r"\b([A-Z][a-z]+\s+){2,}", s))),
    })
    return feats


REGEX_FEATURE_NAMES = list(regex_features("placeholder").keys())

# Hard-cue keys
HARD_SUB_KEYS = ["has_dollar", "has_percent", "has_bps", "has_million_billion",
                 "has_year", "has_quarter", "guidance_word", "margin_word"]
HARD_BOI_KEYS = ["starts_with_operator", "mute_lines", "queue_phrase",
                 "recording_phrase", "safe_harbor", "forward_looking",
                 "name_intro", "analyst_firm"]

SEED = 42


# --------------------------------------------------------------------------
# Member model factories — must match the notebook EXACTLY
# --------------------------------------------------------------------------

def make_logreg_emb():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf",    LogisticRegression(max_iter=2000, C=1.0,
                                       class_weight="balanced", random_state=SEED)),
    ])


def make_svm_tfidf():
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=2, sublinear_tf=True)),
        ("clf",   CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced",
                                                   random_state=SEED),
                                          method="sigmoid", cv=3)),
    ])


def make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.07, max_depth=6,
        min_samples_leaf=20, random_state=SEED, class_weight="balanced",
    )


def make_distill():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf",    Ridge(alpha=1.0, random_state=SEED)),
    ])


def make_xgb():
    """XGBoost on (embeddings ⊕ regex flags) — the notebook's winning model."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def make_lgb():
    """LightGBM on (embeddings ⊕ regex flags)."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def to_proba_distill(z):
    return 1.0 / (1.0 + np.exp(-(z * 4.0 - 2.0)))


def two_stage_score(rx_df: pd.DataFrame, emb: np.ndarray, base_logreg) -> np.ndarray:
    base_p = base_logreg.predict_proba(emb)[:, 1]
    sub_hits = rx_df[HARD_SUB_KEYS].sum(axis=1).values
    boi_hits = rx_df[HARD_BOI_KEYS].sum(axis=1).values
    out = base_p.copy()
    out = np.where(sub_hits >= 1, np.maximum(out, 0.92), out)
    out = np.where((sub_hits == 0) & (boi_hits >= 2),
                   np.minimum(out, 0.10), out)
    return out


# --------------------------------------------------------------------------
# Optional: SetFit and FinBERT  (classes live in model_wrappers.py so
# joblib can unpickle them from any script that imports this module)
# --------------------------------------------------------------------------
from model_wrappers import _SetFitWrapper, _FinBERTWrapper  # noqa: E402

SETFIT_CKPT  = MODELS_DIR / "setfit_final"
FINBERT_CKPT = MODELS_DIR / "finbert_final"


def try_load_setfit():
    if not SETFIT_CKPT.exists():
        return None
    try:
        from setfit import SetFitModel  # noqa: F401
    except ImportError:
        print("  [skip] setfit checkpoint exists but `setfit` not installed.")
        return None
    print(f"  loading SetFit from {SETFIT_CKPT}...")
    return _SetFitWrapper(str(SETFIT_CKPT))


def try_load_finbert():
    if not FINBERT_CKPT.exists():
        return None
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: F401
    except ImportError:
        print("  [skip] finbert checkpoint exists but transformers/torch not installed.")
        return None
    print(f"  loading FinBERT from {FINBERT_CKPT}...")
    return _FinBERTWrapper(str(FINBERT_CKPT))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print(f"Loading bundle from: {BUNDLE_IN}")
    if not BUNDLE_IN.exists():
        raise FileNotFoundError(
            f"{BUNDLE_IN} not found.\n"
            "Run the notebook end-to-end first so it saves models/bp_best.joblib."
        )
    bundle = joblib.load(BUNDLE_IN)
    winner = bundle["winner_name"]
    threshold = bundle["threshold"]
    embed_model_name = bundle["embed_model"]
    print(f"  winner={winner}  threshold={threshold:.3f}  embedder={embed_model_name}")

    # Load cached data
    train_df = pd.read_parquet(CACHE_DIR / "split_train.parquet")
    val_df   = pd.read_parquet(CACHE_DIR / "split_val.parquet")
    pool_df  = pd.concat([train_df, val_df]).reset_index(drop=True)
    y_pool   = pool_df["y"].values

    # Embeddings
    embed_safe_name = embed_model_name.replace("/", "__")
    emb_npz = CACHE_DIR / f"embeddings_{embed_safe_name}.npz"
    cache_npz = np.load(emb_npz, allow_pickle=True)
    sid_to_idx  = {sid: i for i, sid in enumerate(cache_npz["sentence_ids"])}
    embeddings  = cache_npz["embeddings"]

    def get_emb(df: pd.DataFrame) -> np.ndarray:
        return np.stack([embeddings[sid_to_idx[sid]] for sid in df["sentence_id"]])

    X_pool_emb  = get_emb(pool_df)
    X_pool_rx   = pd.DataFrame([regex_features(s) for s in pool_df["sentence"]],
                               columns=REGEX_FEATURE_NAMES)
    X_pool_full = np.hstack([X_pool_emb, X_pool_rx.values])
    texts_pool  = pool_df["sentence"].tolist()

    # Soft targets for distill
    votes = pd.read_parquet(CACHE_DIR / "judge_votes.parquet")
    votes_pool = votes[votes["sentence_id"].isin(pool_df["sentence_id"])].copy()
    votes_pool["p_sub"] = np.where(
        votes_pool["label"] == "substantive",
        votes_pool["confidence"], 1.0 - votes_pool["confidence"],
    )
    mean_p   = votes_pool.groupby("sentence_id")["p_sub"].mean()
    agree    = votes_pool.groupby("sentence_id")["label"].apply(
        lambda s: int(s.value_counts().iloc[0] == len(s)))
    soft_target = pool_df["sentence_id"].map(mean_p).fillna(pool_df["y"]).values
    sample_w    = (0.6 + 0.4 * pool_df["sentence_id"].map(agree).fillna(0.5)).values

    # ------------------------------------------------------------------
    # Refit every member model on the full train+val pool
    # ------------------------------------------------------------------
    print("\nRefitting member models on the full train+val pool...")
    fitted = {}

    t0 = time.perf_counter()
    fitted["logreg_embed"] = make_logreg_emb().fit(X_pool_emb, y_pool)
    print(f"  logreg_embed       {time.perf_counter() - t0:5.1f}s")

    t0 = time.perf_counter()
    fitted["svm_charngram"] = make_svm_tfidf().fit(texts_pool, y_pool)
    print(f"  svm_charngram      {time.perf_counter() - t0:5.1f}s")

    t0 = time.perf_counter()
    fitted["hgb_combined"] = make_hgb().fit(X_pool_full, y_pool)
    print(f"  hgb_combined       {time.perf_counter() - t0:5.1f}s")

    t0 = time.perf_counter()
    distill = make_distill().fit(X_pool_emb, soft_target,
                                 clf__sample_weight=sample_w)
    fitted["distill_softlabel"] = distill
    print(f"  distill_softlabel  {time.perf_counter() - t0:5.1f}s")

    # two_stage uses the logreg as its base
    fitted["two_stage_base"] = fitted["logreg_embed"]

    # XGBoost — the WINNER
    try:
        t0 = time.perf_counter()
        xgb_model = make_xgb()
        xgb_model.fit(X_pool_full, y_pool)
        fitted["xgb_combined"] = xgb_model
        print(f"  xgb_combined       {time.perf_counter() - t0:5.1f}s")

        # xgb_guarded shares the same underlying XGB model;
        # guardrails are applied at inference time in app.py
        fitted["xgb_guarded"] = xgb_model
        print(f"  xgb_guarded        (reuses xgb_combined, guardrails at inference)")
    except ImportError as e:
        print(f"  [WARN] {e}  — xgb_combined / xgb_guarded skipped")

    # LightGBM
    try:
        t0 = time.perf_counter()
        lgb_model = make_lgb()
        lgb_model.fit(X_pool_full, y_pool)
        fitted["lgb_combined"] = lgb_model
        print(f"  lgb_combined       {time.perf_counter() - t0:5.1f}s")
    except ImportError as e:
        print(f"  [WARN] {e}  — lgb_combined skipped")

    # Optional transformer families
    setfit_obj  = try_load_setfit()
    finbert_obj = try_load_finbert()
    if setfit_obj  is not None: fitted["setfit_predict"]  = setfit_obj
    if finbert_obj is not None: fitted["finbert_predict"] = finbert_obj

    # Sanity check
    needs = set(bundle.get("ensemble_members", []))
    needs.add(winner)
    missing = [
        k for k in needs
        if k not in fitted
        and k not in ("setfit",)  # setfit referenced via setfit_predict
        and k not in ("stacked_meta", "mean_ensemble", "weighted_recall_ensemble_v2",
                      "optimized_ensemble", "recall_safe_blend")  # meta-entries
    ]
    if winner == "xgb_combined" and "xgb_combined" not in fitted:
        print("\n!! Winner is xgb_combined but XGBoost is not installed (see above).")

    # ------------------------------------------------------------------
    # Write inference bundle
    # ------------------------------------------------------------------
    out = {
        "winner_name":        winner,
        "threshold":          float(threshold),
        "embed_model":        embed_model_name,
        "feature_patterns":   FEATURE_PATTERNS,
        "regex_feature_names": REGEX_FEATURE_NAMES,
        "hard_sub_keys":      HARD_SUB_KEYS,
        "hard_boi_keys":      HARD_BOI_KEYS,
        "test_metrics":       bundle.get("test_metrics", {}),
        "ensemble_members":   bundle.get("ensemble_members", []),
        "members":            fitted,
        "meta_estimator":     bundle.get("meta_estimator"),
        "leaderboard":        bundle.get("leaderboard"),
        "trained_at":         bundle.get("trained_at"),
        "has_setfit":         setfit_obj  is not None,
        "has_finbert":        finbert_obj is not None,
    }
    joblib.dump(out, BUNDLE_OUT)
    print(f"\nWrote inference bundle: {BUNDLE_OUT}")
    print(f"  size   : {BUNDLE_OUT.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  winner : {winner}  (threshold={threshold:.3f})")
    if out["ensemble_members"]:
        print(f"  members: {out['ensemble_members']}")


if __name__ == "__main__":
    main()
