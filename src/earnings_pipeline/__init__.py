"""Earnings call NLP pipeline package for Assignment 1."""

from .parser import Transcript, parse_transcript, parse_transcript_v2
from .sentiment import get_finbert_pipeline, extract_speaker_sentiment_records
from .extraction import extract_events_dual_llm, load_dual_extractions_df
from .features import build_enhanced_feature_table
