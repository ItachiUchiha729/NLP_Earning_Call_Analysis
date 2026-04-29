# SUBMISSION_CACHE — Pre-computed LLM Extractions

Place this folder at the root of the NLP_HW1 directory before running the notebook.
This allows you to skip the Anthropic API calls (~$5) and run straight to modeling.

## Contents

### extractions/
131 JSON files (one per earnings call): `{TICKER}_{QUARTER}.json`

Each file contains dual-LLM extraction results (qwen3:14b primary, gemma3:4b secondary):
  - `primary` / `secondary`: sentiment, guidance, wins, risks
  - `agreement`: score and confidence
  - `chosen`: final merged values

The notebook cell "Phase 1H: module-driven pipeline run" auto-detects these files and
skips the API call if the file already exists.

### features_v2_all_horizons.parquet
Pre-built feature table (131 rows × 34 columns) including:
  - Sentiment features (LLM + FinBERT, speaker-level)
  - QoQ deltas
  - Forward returns at 1d, 5d, 21d, 63d horizons

This file allows you to skip directly to Section 22+ (modeling/classification).
Load it with: `pd.read_parquet("SUBMISSION_CACHE/features_v2_all_horizons.parquet")`

### extractions/speaker_sent_finbert.parquet
FinBERT speaker-level sentiment (CEO / CFO / analyst per call).

## Quick start to skip LLM extraction
```python
import pandas as pd, shutil
from pathlib import Path

# Copy cache into notebook's expected location
shutil.copytree("SUBMISSION_CACHE/extractions", "cache/extractions", dirs_exist_ok=True)
shutil.copy("SUBMISSION_CACHE/features_v2_all_horizons.parquet", "cache/")

# Then run cells from Section 22 onward
features_v2_all = pd.read_parquet("cache/features_v2_all_horizons.parquet")
```
