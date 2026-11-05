"""
Step 2: run each downloaded 8-K through an LLM and pull out new-product details.

Pipeline per filing:
  1. Trim the document to the part most likely to hold the announcement.
  2. (Optional) spaCy NER pass to collect ORG / PRODUCT candidates. These are
     handed to the LLM as hints so it anchors on real entity strings.
  3. Prompt the LLM for strict JSON: a list of products, each with a name and
     a description under 180 characters.
  4. Validate, truncate, and append rows to the CSV.

Usage:
    python extract_products.py                      # all files in data/raw
    python extract_products.py --limit 10           # first 10 only
    python extract_products.py --backend anthropic  # override config
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import requests

import config

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a financial analyst extracting structured data from SEC Form 8-K filings.
Your job is Named Entity Recognition for two entity types:
  - ORGANIZATION: the company filing the 8-K
  - PRODUCT: any NEW product, service, device, drug, platform, or offering the filing announces as launched, released, introduced, or unveiled.

Rules:
- Only report products that are genuinely new launches or releases described in the text. Ignore earnings figures, executive changes, acquisitions, and existing products mentioned in passing.
- Securities are NOT products. Never report notes, bonds, debentures, preferred stock, depositary shares, common stock offerings, credit facilities, warrants, or any other financing instrument.
- Partnerships, contracts, and distribution agreements are NOT products unless a specific named product is being launched.
- Use the product's exact name as written in the filing.
- Each description must be a single plain sentence, factual, and UNDER 180 characters.
- If no new product is announced, return an empty list.
- Respond with JSON only. No markdown, no commentary."""

USER_TEMPLATE = """Company (from filing header): {company}
Ticker: {ticker}
Filing date: {filing_date}
{ner_block}
FILING TEXT:
\"\"\"
{text}
\"\"\"

Return JSON in exactly this shape:
{{"company_name": "<official company name as written in the text>",
  "products": [
    {{"new_product": "<product name>", "product_description": "<one sentence, under 180 chars>"}}
  ]}}"""


# ---------------------------------------------------------------------------
# Text trimming
# ---------------------------------------------------------------------------
TRIGGER_RE = re.compile(
    r"\b(launch(?:es|ed|ing)?|introduc(?:es|ed|ing)|unveil(?:s|ed)?|releas(?:es|ed|ing)|"
    r"announc(?:es|ed)\s+(?:the\s+)?(?:launch|availability|release|introduction)|now available)\b",
    re.I,
)


def focus_text(text, max_chars):
    """Keep the window of text around the first launch keyword, else the start."""
    if len(text) <= max_chars:
        return text
    m = TRIGGER_RE.search(text)
    if not m:
        return text[:max_chars]
    start = max(0, m.start() - max_chars // 3)
    return text[start:start + max_chars]


# ---------------------------------------------------------------------------
# spaCy NER (optional hint generator)
# ---------------------------------------------------------------------------
_nlp = None


def ner_candidates(text):
    """Return (orgs, products) found by spaCy, or empty lists if unavailable."""
    global _nlp
    if not config.USE_SPACY_NER:
        return [], []
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(config.SPACY_MODEL)
        except Exception as e:  # spaCy or model not installed
            print(f"spaCy NER unavailable ({e}); continuing with LLM only.")
            config.USE_SPACY_NER = False
            return [], []
    doc = _nlp(text[:20000])
    orgs, prods = [], []
    for ent in doc.ents:
        s = ent.text.strip()
        if len(s) < 3:
            continue
        if ent.label_ == "ORG" and s not in orgs:
            orgs.append(s)
        elif ent.label_ == "PRODUCT" and s not in prods:
            prods.append(s)
    return orgs[:10], prods[:10]


def build_ner_block(orgs, prods):
    if not orgs and not prods:
        return ""
    lines = ["NER candidates found by a statistical tagger (hints only, verify against the text):"]
    if orgs:
        lines.append("  ORGANIZATION: " + "; ".join(orgs))
    if prods:
        lines.append("  PRODUCT: " + "; ".join(prods))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
def call_ollama(system, user):
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_anthropic(system, user):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=800,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def call_openai(system, user):
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content


BACKENDS = {"ollama": call_ollama, "anthropic": call_anthropic, "openai": call_openai}


def parse_json(raw):
    """Pull the first JSON object out of a model response, tolerating stray text."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
SECURITY_RE = re.compile(
    r"\b(notes?|bonds?|debentures?|preferred (stock|shares?)|depositary shares?|common stock|"
    r"senior (secured|unsecured)|credit (facility|agreement)|warrants?|term loan|offering)\b",
    re.I,
)

PREFERRED_TICKER_RE = re.compile(r"^([A-Z]+)-P[A-Z]?$")


def is_security(name):
    """True when the 'product' is really a financing instrument."""
    return bool(SECURITY_RE.search(name or ""))


def normalize_ticker(t):
    """AXS-PE (a preferred share listing) -> AXS."""
    m = PREFERRED_TICKER_RE.match(t or "")
    return m.group(1) if m else t


def clip(s, n):
    s = " ".join(str(s or "").split())
    if len(s) <= n:
        return s
    cut = s[: n - 1].rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "…"


def extract_one(rec, backend):
    text = focus_text(rec["text"], config.MAX_CONTEXT_CHARS)
    orgs, prods = ner_candidates(text)
    prompt = USER_TEMPLATE.format(
        company=rec["company_name"],
        ticker=rec["ticker"],
        filing_date=rec["filing_date"],
        ner_block=build_ner_block(orgs, prods),
        text=text,
    )
    raw = BACKENDS[backend](SYSTEM_PROMPT, prompt)
    data = parse_json(raw)
    if not data:
        return [], raw
    company = clip(data.get("company_name") or rec["company_name"], 120)
    rows = []
    for p in data.get("products") or []:
        name = clip(p.get("new_product"), 100)
        desc = clip(p.get("product_description"), config.MAX_DESCRIPTION_CHARS)
        if not name or is_security(name):
            continue
        rows.append({
            "company_name": company,
            "stock_name": normalize_ticker(rec["ticker"]),
            "filing_time": rec["filing_date"],
            "new_product": name,
            "product_description": desc,
            "accession": rec["accession"],
            "source_url": rec["url"],
        })
    return rows, raw


FIELDS = ["company_name", "stock_name", "filing_time", "new_product", "product_description"]
FIELDS_FULL = FIELDS + ["accession", "source_url"]


def main(limit, backend, out_name):
    if backend not in BACKENDS:
        sys.exit(f"Unknown backend {backend!r}; choose from {list(BACKENDS)}")
    files = sorted(config.RAW_DIR.glob("*.json"))
    if limit:
        files = files[:limit]
    if not files:
        sys.exit(f"No filings in {config.RAW_DIR}. Run fetch_filings.py first.")

    out_path = config.OUTPUT_DIR / out_name
    log_path = config.OUTPUT_DIR / "llm_responses.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            done = {r["accession"] for r in csv.DictReader(f)}

    print(f"Backend: {backend}  model: {getattr(config, backend.upper() + '_MODEL')}")
    print(f"{len(files)} filings, {len(done)} already processed -> {out_path}")

    new_file = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f, open(log_path, "a", encoding="utf-8") as log:
        writer = csv.DictWriter(f, fieldnames=FIELDS_FULL, quoting=csv.QUOTE_ALL)
        if new_file:
            writer.writeheader()
        for i, path in enumerate(files, 1):
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec["accession"] in done:
                continue
            t0 = time.time()
            try:
                rows, raw = extract_one(rec, backend)
            except Exception as e:
                print(f"[{i:>4}] {rec['ticker']:<6} ERROR {e}")
                continue
            log.write(json.dumps({"accession": rec["accession"], "raw": raw}) + "\n")
            if not rows:
                # keep a record that we looked and found nothing
                rows = [{"company_name": rec["company_name"], "stock_name": normalize_ticker(rec["ticker"]),
                         "filing_time": rec["filing_date"], "new_product": "", "product_description": "",
                         "accession": rec["accession"], "source_url": rec["url"]}]
            for r in rows:
                writer.writerow(r)
            f.flush()
            names = ", ".join(r["new_product"] for r in rows if r["new_product"]) or "(none)"
            print(f"[{i:>4}] {rec['ticker']:<6} {time.time()-t0:5.1f}s  {names}")

    # Also emit the clean 5-column version the assignment asks for.
    clean_path = config.OUTPUT_DIR / "new_products.csv"
    with open(out_path, newline="", encoding="utf-8") as src, open(clean_path, "w", newline="", encoding="utf-8") as dst:
        w = csv.DictWriter(dst, fieldnames=FIELDS, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        w.writeheader()
        n = 0
        for r in csv.DictReader(src):
            if r["new_product"]:
                w.writerow(r)
                n += 1
    print(f"Wrote {n} product rows to {clean_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N filings")
    ap.add_argument("--backend", default=config.LLM_BACKEND, choices=list(BACKENDS))
    ap.add_argument("--out", default="extraction_full.csv", help="CSV name inside data/output")
    args = ap.parse_args()
    main(args.limit, args.backend, args.out)
