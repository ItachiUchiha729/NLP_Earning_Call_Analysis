"""app.py — Streamlit GUI for inline boilerplate vs. substantive tagging.

Run:
    streamlit run app.py

Loads `models/bp_inference.joblib` produced by `finalize_inference.py`.
Accepts a transcript via file-upload or paste, sentence-tokenises it,
predicts each sentence with the trained ensemble, and renders the document
inline with boilerplate sentences highlighted in red.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from model_wrappers import _SetFitWrapper, _FinBERTWrapper  # noqa: F401 — needed for joblib unpickling

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BPClassifier — Earnings-Call Tagger",
    page_icon="📞",
    layout="wide",
)

ROOT        = Path(__file__).resolve().parent
BUNDLE_PATH = ROOT / "models" / "bp_inference.joblib"

# ---------------------------------------------------------------------------
# NLTK punkt (first-run download)
# ---------------------------------------------------------------------------
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    with st.spinner("Downloading NLTK punkt tokeniser (first run only)…"):
        nltk.download("punkt_tab", quiet=True)
        nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize


# ---------------------------------------------------------------------------
# Lazy-load models (cached so no reload on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        st.error(
            f"Inference bundle not found at **{BUNDLE_PATH}**.\n\n"
            "Run `python finalize_inference.py` first to build it."
        )
        st.stop()
    return joblib.load(BUNDLE_PATH)


@st.cache_resource
def load_embedder(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


bundle   = load_bundle()
embedder = load_embedder(bundle["embed_model"])
THRESHOLD = bundle["threshold"]
WINNER    = bundle["winner_name"]

# ---------------------------------------------------------------------------
# Feature extraction (must match finalize_inference.py / notebook exactly)
# ---------------------------------------------------------------------------
_COMPILED = [
    (name, re.compile(pat, re.IGNORECASE))
    for name, pat in bundle["feature_patterns"]
]
REGEX_FEATURE_NAMES = bundle["regex_feature_names"]
HARD_SUB_KEYS       = bundle["hard_sub_keys"]
HARD_BOI_KEYS       = bundle["hard_boi_keys"]


def regex_features(sentence: str) -> dict:
    s     = sentence
    feats = {name: int(bool(rx.search(s))) for name, rx in _COMPILED}
    n_chars  = len(s)
    words    = s.split()
    n_words  = len(words)
    n_digits = sum(c.isdigit() for c in s)
    n_alpha  = sum(c.isalpha() for c in s)
    n_upper  = sum(c.isupper() for c in s)
    feats.update({
        "len_chars":          n_chars,
        "len_words":          n_words,
        "digit_ratio":        n_digits / max(1, n_chars),
        "uppercase_ratio":    n_upper  / max(1, n_alpha),
        "ends_with_question": int(s.rstrip().endswith("?")),
        "starts_with_number": int(bool(re.match(r"^\s*\d", s))),
        "first_person_count": sum(1 for w in words if w.lower() in {"we", "our", "us", "i"}),
        "modal_count":        sum(1 for w in words if w.lower() in
                                   {"will", "would", "expect", "expects", "expected",
                                    "plan", "plans", "intend", "may", "might", "could", "should"}),
        "proper_noun_run":    int(bool(re.search(r"\b([A-Z][a-z]+\s+){2,}", s))),
    })
    return feats


def two_stage_score(rx_df: pd.DataFrame, emb: np.ndarray, base_logreg) -> np.ndarray:
    base_p   = base_logreg.predict_proba(emb)[:, 1]
    sub_hits = rx_df[HARD_SUB_KEYS].sum(axis=1).values
    boi_hits = rx_df[HARD_BOI_KEYS].sum(axis=1).values
    out = base_p.copy()
    out = np.where(sub_hits >= 1,                      np.maximum(out, 0.92), out)
    out = np.where((sub_hits == 0) & (boi_hits >= 2),  np.minimum(out, 0.10), out)
    return out


def to_proba_distill(z):
    return 1.0 / (1.0 + np.exp(-(z * 4.0 - 2.0)))


# ---------------------------------------------------------------------------
# Per-member inference
# ---------------------------------------------------------------------------
def _predict_member(name: str, sentences: list[str],
                    emb: np.ndarray, rx: pd.DataFrame) -> np.ndarray:
    members = bundle["members"]

    if name == "logreg_embed":
        return members["logreg_embed"].predict_proba(emb)[:, 1]

    if name == "svm_charngram":
        return members["svm_charngram"].predict_proba(sentences)[:, 1]

    if name in ("hgb_combined", "xgb_combined", "lgb_combined"):
        full = np.hstack([emb, rx.values])
        return members[name].predict_proba(full)[:, 1]

    if name == "xgb_guarded":
        # XGB probabilities + hard guardrails on obvious boilerplate / sub cues
        full     = np.hstack([emb, rx.values])
        base_p   = members["xgb_guarded"].predict_proba(full)[:, 1]
        sub_hits = rx[HARD_SUB_KEYS].sum(axis=1).values
        boi_hits = rx[HARD_BOI_KEYS].sum(axis=1).values
        out = base_p.copy()
        out = np.where(sub_hits >= 1,                     np.maximum(out, 0.92), out)
        out = np.where((sub_hits == 0) & (boi_hits >= 2), np.minimum(out, 0.10), out)
        return out

    if name == "two_stage":
        return two_stage_score(rx, emb, members["two_stage_base"])

    if name == "distill_softlabel":
        return to_proba_distill(members["distill_softlabel"].predict(emb))

    if name == "setfit":
        if "setfit_predict" not in members:
            raise RuntimeError(
                "SetFit referenced but no checkpoint loaded — "
                "check models/setfit_final/ exists."
            )
        return members["setfit_predict"](sentences)

    if name == "finbert":
        if "finbert_predict" not in members:
            raise RuntimeError("FinBERT referenced but no checkpoint loaded.")
        return members["finbert_predict"](sentences)

    raise ValueError(f"Unknown ensemble member: {name!r}")


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------
def predict_probs(sentences: list[str]) -> np.ndarray:
    """Return P(substantive) for each sentence using the bundled winner."""
    if not sentences:
        return np.zeros(0)

    emb = embedder.encode(
        sentences, batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    rx = pd.DataFrame(
        [regex_features(s) for s in sentences],
        columns=REGEX_FEATURE_NAMES,
    )

    members = bundle["members"]

    # --- Ensemble winners ---
    if WINNER in ("stacked_meta",):
        member_probs = np.column_stack([
            _predict_member(m, sentences, emb, rx)
            for m in bundle["ensemble_members"]
        ])
        return bundle["meta_estimator"].predict_proba(member_probs)[:, 1]

    if WINNER in ("mean_ensemble", "weighted_recall_ensemble_v2",
                  "optimized_ensemble", "recall_safe_blend"):
        member_names = bundle.get("ensemble_members", [])
        if not member_names:
            # Fallback: use all refitted base models
            member_names = [k for k in ("xgb_combined", "lgb_combined",
                                        "hgb_combined", "logreg_embed",
                                        "svm_charngram", "distill_softlabel")
                            if k in members]
        return np.mean(
            [_predict_member(m, sentences, emb, rx) for m in member_names],
            axis=0,
        )

    # --- Single-model winners (xgb_combined, lgb_combined, etc.) ---
    return _predict_member(WINNER, sentences, emb, rx)


# ---------------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------------
MIN_SENT_CHARS = 40


def extract_sentences(transcript: str) -> list[tuple[str, bool]]:
    """Return (sentence, is_taggable) pairs preserving original document order."""
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for para in (p.strip() for p in re.split(r"\n\s*\n", transcript) if p.strip()):
        for sent in sent_tokenize(para):
            sent = re.sub(r"\s+", " ", sent).strip()
            if not sent:
                continue
            taggable = len(sent) >= MIN_SENT_CHARS and sent not in seen
            if taggable:
                seen.add(sent)
            out.append((sent, taggable))
    return out


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .doc-pane {
        background: #fafafa;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 18px 22px;
        max-height: 72vh;
        overflow-y: auto;
        font-size: 15px;
        line-height: 1.75;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }
      .boilerplate {
        background: #ffe1e1;
        padding: 1px 3px;
        border-radius: 3px;
        border-bottom: 2px solid #ff8888;
      }
      .substantive { color: #1a1a1a; }
      .untagged    { color: #aaa; font-style: italic; }
      .stat-card {
        background: white;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
      }
      .stat-num   { font-size: 30px; font-weight: 700; margin: 0; }
      .stat-label { font-size: 11px; color: #666; text-transform: uppercase;
                    letter-spacing: 0.6px; margin: 0; }
      .legend-box { display: inline-block; width: 14px; height: 14px;
                    border-radius: 3px; vertical-align: middle; margin-right: 5px; }
      .metric-bar { background: #f0f2f6; border-radius: 6px; padding: 8px 12px;
                    margin-bottom: 6px; font-size: 12px; color: #444; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📞 BPClassifier — Earnings-Call Tagger")

test_m = bundle.get("test_metrics", {})
macro_f1  = test_m.get("macro_f1",  0.0)
sub_rec   = test_m.get("sub_recall", 0.0)
sub_f1    = test_m.get("sub_f1",     0.0)
boi_f1    = test_m.get("boi_f1",     0.0)
test_acc  = test_m.get("test_acc",   0.0)

st.markdown(
    f"<div class='metric-bar'>"
    f"Model: <b>{WINNER}</b> &nbsp;|&nbsp; "
    f"Threshold: <b>{THRESHOLD:.3f}</b> &nbsp;|&nbsp; "
    f"Test macro-F1: <b>{macro_f1:.4f}</b> &nbsp;|&nbsp; "
    f"Substantive recall: <b>{sub_rec:.4f}</b> &nbsp;|&nbsp; "
    f"Boilerplate F1: <b>{boi_f1:.4f}</b> &nbsp;|&nbsp; "
    f"Accuracy: <b>{test_acc:.4f}</b>"
    f"</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.subheader("Load a transcript")
col_up, col_paste = st.columns([1, 2])

with col_up:
    uploaded = st.file_uploader("Upload a .txt transcript", type=["txt"])

with col_paste:
    pasted = st.text_area(
        "Or paste transcript text here",
        height=160,
        placeholder="Paste an earnings-call transcript…",
    )

transcript_text = ""
if uploaded is not None:
    transcript_text = uploaded.read().decode("utf-8", errors="ignore")
elif pasted.strip():
    transcript_text = pasted

if not transcript_text:
    st.info("Upload or paste a transcript to begin tagging.")
    st.stop()

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
sent_pairs = extract_sentences(transcript_text)
taggable   = [s for s, ok in sent_pairs if ok]

if not taggable:
    st.warning("No sentences ≥ 40 characters found in the input.")
    st.stop()

with st.spinner(f"Tagging {len(taggable):,} sentences…"):
    probs          = predict_probs(taggable)
    is_substantive = probs >= THRESHOLD
    decisions      = {
        sent: (bool(sub), float(p))
        for sent, sub, p in zip(taggable, is_substantive, probs)
    }

# ---------------------------------------------------------------------------
# Stats panel
# ---------------------------------------------------------------------------
n_total = len(taggable)
n_sub   = int(is_substantive.sum())
n_boi   = n_total - n_sub
pct_sub = n_sub / n_total * 100
pct_boi = n_boi / n_total * 100

st.subheader("Results")
left, right = st.columns([3, 1], gap="large")

with right:
    st.markdown("#### Statistics")
    st.markdown(
        f"""
        <div class="stat-card">
          <p class="stat-label">Total sentences tagged</p>
          <p class="stat-num">{n_total:,}</p>
        </div>
        <div class="stat-card" style="border-left: 4px solid #ff8888;">
          <p class="stat-label">Boilerplate</p>
          <p class="stat-num">{n_boi:,}</p>
          <p style="margin:0;color:#666;">{pct_boi:.1f}%</p>
        </div>
        <div class="stat-card" style="border-left: 4px solid #2c8a2c;">
          <p class="stat-label">Substantive</p>
          <p class="stat-num">{n_sub:,}</p>
          <p style="margin:0;color:#666;">{pct_sub:.1f}%</p>
        </div>
        <div style="margin-top:14px;font-size:12px;color:#666;">
          <span class="legend-box" style="background:#ffe1e1;border-bottom:2px solid #ff8888;"></span>boilerplate (highlighted)<br>
          <span style="display:inline-block;width:14px;"></span>substantive (plain text)
        </div>
        """,
        unsafe_allow_html=True,
    )

with left:
    st.markdown("#### Tagged transcript")
    parts: list[str] = []
    for sent, ok in sent_pairs:
        safe = (sent.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
        if not ok:
            parts.append(f'<span class="untagged">{safe}</span> ')
            continue
        is_sub, p = decisions[sent]
        if is_sub:
            parts.append(f'<span class="substantive" title="P(sub)={p:.3f}">{safe}</span> ')
        else:
            parts.append(f'<span class="boilerplate"  title="P(sub)={p:.3f}">{safe}</span> ')

    st.markdown(
        f'<div class="doc-pane">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Per-sentence detail + download
# ---------------------------------------------------------------------------
with st.expander("Per-sentence detail (downloadable CSV)"):
    detail_df = pd.DataFrame([
        {
            "sentence":      s,
            "p_substantive": round(p, 4),
            "prediction":    "substantive" if sub else "boilerplate",
        }
        for s, (sub, p) in decisions.items()
    ])
    st.dataframe(detail_df, use_container_width=True, height=300)
    st.download_button(
        "⬇ Download CSV",
        detail_df.to_csv(index=False),
        file_name="tagged_sentences.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Probability distribution chart
# ---------------------------------------------------------------------------
with st.expander("Probability distribution"):
    import altair as alt

    prob_df = pd.DataFrame({
        "P(substantive)": probs,
        "label": ["substantive" if s else "boilerplate" for s in is_substantive],
    })
    chart = (
        alt.Chart(prob_df)
        .mark_bar(opacity=0.75, binSpacing=1)
        .encode(
            alt.X("P(substantive):Q", bin=alt.Bin(maxbins=40), title="P(substantive)"),
            alt.Y("count():Q", title="Count"),
            alt.Color("label:N",
                      scale=alt.Scale(
                          domain=["boilerplate", "substantive"],
                          range=["#ff8888",       "#2c8a2c"],
                      )),
        )
        .properties(width=600, height=220, title="Sentence probability distribution")
    )
    threshold_line = (
        alt.Chart(pd.DataFrame({"x": [THRESHOLD]}))
        .mark_rule(color="#333", strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="x:Q")
    )
    st.altair_chart(chart + threshold_line, use_container_width=True)
    st.caption(f"Vertical dashed line = decision threshold ({THRESHOLD:.3f}). "
               "Sentences to the right are classified as substantive.")
