# SEC 8-K New Product Extraction with LLMs

Pulls Form 8-K filings from SEC EDGAR, runs Named Entity Recognition plus a local
large language model over each one, and writes the new product announcements to a CSV:

```
company_name, stock_name, filing_time, new_product, product_description
```

Built for the AI for Finance course in the UCF MS in Financial Technology.

## How it works

| Step | File | What it does |
|---|---|---|
| 1 | `fetch_filings.py` | Queries EDGAR full-text search for 8-K press release exhibits containing launch language ("announces the launch of", "introduces new", ...), maps CIK to ticker with the SEC's `company_tickers.json`, downloads each document, strips the HTML, and saves one JSON per filing to `data/raw/`. |
| 2 | `extract_products.py` | For each filing: trims the text to the window around the first launch keyword, runs spaCy NER to collect ORG and PRODUCT candidates, then prompts the LLM (Ollama by default) for strict JSON. Descriptions are clipped to 180 characters. Results go to `data/output/`. |
| 3 | `notebooks/8k_product_extraction.ipynb` | Walks through both steps interactively and shows the results table. |

Outputs:

* `data/output/new_products.csv`  the five-column deliverable
* `data/output/extraction_full.csv`  same rows plus accession number and source URL, and one blank row per filing where no product was found (audit trail)
* `data/output/llm_responses.jsonl`  raw model output per filing, for debugging

## Setup

```bash
git clone <this repo>
cd sec-8k-product-extraction
python -m venv .venv
.venv\Scripts\activate            # Windows    (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.1
```

Edit `config.py`:

* `SEC_USER_AGENT`  your name and email. EDGAR refuses anonymous clients.
* `OLLAMA_MODEL`  whichever model you pulled (`llama3.1`, `deepseek-r1:8b`, ...)
* `MAX_FILINGS`  how many filings to collect (default 120)

Optional: drop a `data/sp500_tickers.txt` (one ticker per line) to restrict to S&P 500 filers.

## Run

```bash
python fetch_filings.py            # ~2 minutes for 120 filings
python extract_products.py         # depends on your GPU; ~10-40 s per filing on CPU
```

Or open the notebook. Both scripts are resumable: rerunning skips filings already done.

To use a hosted model instead of Ollama:

```bash
set ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY
python extract_products.py --backend anthropic
```

## Design notes

* **Why full-text search instead of crawling every 8-K.** Most 8-Ks are earnings, financing, or governance items. Searching for launch phrases up front means the LLM spends its time on filings that are likely to contain a product, which matters when each query costs 10 to 40 seconds on a local model.
* **Why NER plus LLM.** spaCy's statistical tagger is fast and good at spotting proper nouns but has no idea which product is *new*. The LLM is good at the reasoning but can hallucinate names. Feeding spaCy's candidates into the prompt as hints gets the best of both: the LLM anchors on strings that appear in the text.
* **Why `format: json` and temperature 0.** Structured output plus deterministic decoding keeps the parser simple and the results repeatable.
* **Truncation.** Descriptions are clipped at a word boundary to stay under 180 characters, the limit set in the assignment.
