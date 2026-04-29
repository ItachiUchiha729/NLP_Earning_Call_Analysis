from pathlib import Path


def default_paths(root: Path) -> dict:
    cache = root / "cache"
    return {
        "ROOT": root,
        "TRANSCRIPTS": root / "transcripts",
        "CACHE": cache,
        "EXTRACTIONS": cache / "extractions",
        "PRICES": cache / "prices",
    }


OLLAMA_HOST = "http://localhost:11434"
OLLAMA_NUM_CTX = 32768
PRIMARY_MODEL = "gemma3:4b"
SECONDARY_MODEL = "gemma3:4b"
FINBERT_MODEL = "ProsusAI/finbert"

# --- Anthropic API backend ---
# Set BACKEND = "anthropic" to use Claude instead of local Ollama.
# Paste your key from https://console.anthropic.com
BACKEND = "ollama"          # "ollama" | "anthropic"
ANTHROPIC_API_KEY = ""      # sk-ant-... (only needed if BACKEND = "anthropic")
ANTHROPIC_PRIMARY_MODEL = "claude-3-5-haiku-latest"
ANTHROPIC_SECONDARY_MODEL = "claude-3-5-haiku-latest"
