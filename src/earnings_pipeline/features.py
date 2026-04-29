import numpy as np
import pandas as pd


def _set_from_items(items):
    out = set()
    for x in (items or []):
        if isinstance(x, dict):
            lab = str(x.get("label", "")).strip().lower()
        else:
            lab = str(x).strip().lower()
        if lab:
            out.add(lab)
    return out


def _avg_severity(items):
    vals = []
    for x in (items or []):
        if isinstance(x, dict):
            vals.append(float(x.get("severity", 3) or 3))
    return float(np.mean(vals)) if vals else 0.0


def _price_context_features(ticker: str, call_date: str, prices: dict):
    df = prices.get(ticker)
    if df is None or len(df) == 0:
        return {"mom_21d": np.nan, "dist_52w_high": np.nan}

    d0 = pd.Timestamp(call_date)
    # Anti-lookahead (strict): anchor features to the last known close strictly before call date.
    # This avoids any dependence on same-day close when call time within the day is unknown.
    hist = df[df.Date < d0]
    if hist.empty:
        return {"mom_21d": np.nan, "dist_52w_high": np.nan}

    i = int(hist.index[-1])
    if i - 21 < 0:
        mom_21d = np.nan
    else:
        mom_21d = float(df.Close.iloc[i] / df.Close.iloc[i - 21] - 1)

    lo = max(0, i - 252)
    rolling_high = float(df.Close.iloc[lo : i + 1].max()) if i >= lo else np.nan
    dist_52w_high = float(df.Close.iloc[i] / rolling_high - 1) if rolling_high and not np.isnan(rolling_high) else np.nan
    return {"mom_21d": mom_21d, "dist_52w_high": dist_52w_high}


def build_enhanced_feature_table(events_df: pd.DataFrame, speaker_sent_df: pd.DataFrame, returns_df: pd.DataFrame, prices: dict) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    base = events_df.copy()
    base["call_date"] = pd.to_datetime(base["call_date"])

    spk = speaker_sent_df.copy()
    spk["call_date"] = pd.to_datetime(spk["call_date"])
    df = base.merge(spk, on=["ticker", "quarter", "call_date"], how="left")

    df["wins_set"] = df["wins"].apply(_set_from_items)
    df["risks_set"] = df["risks"].apply(_set_from_items)
    df["avg_win_severity"] = df["wins"].apply(_avg_severity)
    df["avg_risk_severity"] = df["risks"].apply(_avg_severity)

    df = df.sort_values(["ticker", "call_date"]).reset_index(drop=True)

    df["sentiment_delta"] = df.groupby("ticker")["sentiment"].diff()
    df["n_wins_delta"] = df.groupby("ticker")["n_wins"].diff()
    df["n_risks_delta"] = df.groupby("ticker")["n_risks"].diff()

    prev_risks = df.groupby("ticker")["risks_set"].shift(1)
    prev_themes = df.groupby("ticker")["themes"].shift(1)

    def _risk_persistence(curr, prev):
        curr = curr or set()
        prev = prev or set()
        return len(curr & prev) / max(1, len(curr))

    def _new_theme_count(curr, prev):
        curr = {str(x).strip().lower() for x in (curr or [])}
        prev = {str(x).strip().lower() for x in (prev or [])}
        return len(curr - prev)

    df["risk_persistence"] = [
        _risk_persistence(c, p if isinstance(p, set) else set()) for c, p in zip(df["risks_set"], prev_risks)
    ]
    df["new_theme_count"] = [
        _new_theme_count(c, p if isinstance(p, list) else []) for c, p in zip(df["themes"], prev_themes)
    ]

    df["role_asym_ceo_cfo"] = df["sent_ceo"].fillna(0.0) - df["sent_cfo"].fillna(0.0)
    df["analyst_pressure_proxy"] = (-df["sent_analyst"].fillna(0.0)) * (1.0 + df["n_risks"].fillna(0.0))
    df["guidance_credibility_gap"] = df["guidance_confidence"].fillna(0.0) - df["analyst_pressure_proxy"].clip(lower=0)

    px_feats = [_price_context_features(tk, str(cd.date()), prices) for tk, cd in zip(df["ticker"], df["call_date"])]
    df = pd.concat([df, pd.DataFrame(px_feats)], axis=1)

    tgt = returns_df.copy()
    tgt["call_date"] = pd.to_datetime(tgt["call_date"])
    df = df.merge(tgt, on=["ticker", "quarter", "call_date"], how="left", suffixes=("", "_ret"))

    keep = [
        "ticker",
        "quarter",
        "call_date",
        "sentiment",
        "sent_overall_finbert",
        "sent_ceo",
        "sent_cfo",
        "sent_analyst",
        "sent_prepared",
        "sent_qa_answer",
        "prepared_qa_slippage",
        "agreement_score",
        "agreement_confidence",
        "sent_gap_models",
        "guidance",
        "guidance_confidence",
        "n_wins",
        "n_risks",
        "avg_win_severity",
        "avg_risk_severity",
        "sentiment_delta",
        "n_wins_delta",
        "n_risks_delta",
        "risk_persistence",
        "new_theme_count",
        "role_asym_ceo_cfo",
        "analyst_pressure_proxy",
        "guidance_credibility_gap",
        "mom_21d",
        "dist_52w_high",
        "fwd_excess_1d",
        "fwd_excess_5d",
        "fwd_excess_21d",
        "fwd_excess_63d",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep].copy()
