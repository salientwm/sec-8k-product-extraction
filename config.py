"""
Central configuration for the 8-K product extraction pipeline.

Edit the values here (or set the matching environment variables) rather than
touching the scripts themselves.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # one .json per downloaded filing
OUTPUT_DIR = DATA_DIR / "output"    # CSV results land here
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SEC EDGAR
# ---------------------------------------------------------------------------
# EDGAR rejects requests that do not identify the client. The SEC asks for a
# name and a contact email, e.g. "Jane Doe jane@example.com".
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Patrick Sharpe patrick.sharpe88@gmail.com")

# Max requests per second EDGAR tolerates (their published limit is 10).
SEC_RATE_LIMIT = 8

# Full-text search phrases used to find 8-Ks that announce products.
# Each phrase is run as its own query; results are de-duplicated.
SEARCH_PHRASES = [
    '"announces the launch of"',
    '"announced the launch of"',
    '"introduces new"',
    '"new product"',
    '"product launch"',
    '"now available"',
    '"unveils"',
]

# Restrict to 8-K exhibits that are usually press releases (EX-99.x).
# Set to None to search the whole filing.
SEARCH_FORMS = "8-K"

# Date window for the search (YYYY-MM-DD). None = EDGAR default (last ~5 yrs).
SEARCH_START_DATE = "2025-01-01"
SEARCH_END_DATE = None

# Give up on a search phrase after this many result pages (100 hits each).
MAX_PAGES_PER_PHRASE = 15

# Stop once this many distinct filings have been collected.
MAX_FILINGS = int(os.getenv("MAX_FILINGS", "120"))

# Only keep filings from S&P 500 companies if a ticker list is provided.
# Leave empty to accept any filer.
SP500_TICKER_FILE = ROOT / "data" / "sp500_tickers.txt"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# "ollama"   -> local model via the Ollama HTTP API (free)
# "anthropic" or "openai" -> hosted API, needs an API key in the environment
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Characters of filing text sent to the model. Press releases are front-loaded,
# so the first few thousand characters usually carry the announcement.
MAX_CONTEXT_CHARS = 9000

# Hard limit required by the assignment.
MAX_DESCRIPTION_CHARS = 180

# ---------------------------------------------------------------------------
# NER (optional, spaCy). Used to give the LLM candidate entities as hints.
# ---------------------------------------------------------------------------
USE_SPACY_NER = True
SPACY_MODEL = "en_core_web_sm"
