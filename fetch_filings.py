"""
Step 1: find and download SEC Form 8-K filings that look like product announcements.

Data source: SEC EDGAR full-text search (efts.sec.gov) plus the EDGAR archive.
Everything is public and free; the SEC only asks that you identify yourself
in the User-Agent header and stay under 10 requests/second.

Output: one JSON file per filing in data/raw/, each holding the metadata
(company, CIK, ticker, filing date) and the cleaned text of the document.

Usage:
    python fetch_filings.py                # uses settings in config.py
    python fetch_filings.py --max 50       # stop after 50 filings
"""
import argparse
import json
import re
import sys
import time
from html import unescape

import requests

import config

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": config.SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
})

_last_request = 0.0


def _get(url, **kwargs):
    """GET with EDGAR's rate limit respected."""
    global _last_request
    wait = (1.0 / config.SEC_RATE_LIMIT) - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    resp = SESSION.get(url, timeout=30, **kwargs)
    _last_request = time.time()
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Ticker lookup
# ---------------------------------------------------------------------------
def load_cik_to_ticker():
    """Map CIK (int) -> ticker using the SEC's official company_tickers.json."""
    data = _get("https://www.sec.gov/files/company_tickers.json").json()
    out = {}
    for row in data.values():
        cik, ticker = int(row["cik_str"]), row["ticker"]
        # A CIK can list several symbols (common + preferred, e.g. AXS and AXS-PE).
        # Prefer the plain common-stock symbol.
        if cik not in out or ("-" in out[cik] and "-" not in ticker):
            out[cik] = ticker
    return out


def load_sp500_tickers():
    path = config.SP500_TICKER_FILE
    if not path.exists():
        return None
    tickers = {t.strip().upper() for t in path.read_text().splitlines() if t.strip()}
    return tickers or None


# ---------------------------------------------------------------------------
# Full-text search
# ---------------------------------------------------------------------------
def search_filings(phrase, start=None, end=None, page=1):
    """Query EDGAR full-text search for one phrase. Returns the raw hit list."""
    params = {"q": phrase, "forms": config.SEARCH_FORMS, "page": page}
    if start or end:
        # EDGAR ignores the window unless both ends are supplied.
        params["dateRange"] = "custom"
        params["startdt"] = start or "2001-01-01"
        params["enddt"] = end or time.strftime("%Y-%m-%d")
    resp = _get("https://efts.sec.gov/LATEST/search-index", params=params)
    return resp.json().get("hits", {}).get("hits", [])


def hit_to_record(hit):
    src = hit["_source"]
    adsh = src["adsh"]
    cik = int(src["ciks"][0])
    # _id looks like "0001234567-25-000123:ex99-1.htm"
    doc_name = hit["_id"].split(":", 1)[1]
    return {
        "accession": adsh,
        "cik": cik,
        "company_name": src["display_names"][0].split("  (CIK")[0].strip(),
        "filing_date": src["file_date"],
        "form": src["form"],
        "doc_name": doc_name,
        "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{doc_name}",
    }


# ---------------------------------------------------------------------------
# Document download + cleanup
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t\r\f\v]+")
NL_RE = re.compile(r"\n\s*\n+")


def html_to_text(html):
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>|</h\d>", "\n", html)
    text = TAG_RE.sub(" ", html)
    text = unescape(text).replace("\xa0", " ")
    text = WS_RE.sub(" ", text)
    text = NL_RE.sub("\n", text)
    return text.strip()


def download_document(url):
    resp = _get(url)
    return html_to_text(resp.text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(max_filings):
    print(f"User-Agent: {config.SEC_USER_AGENT}")
    if "example.com" in config.SEC_USER_AGENT:
        sys.exit("Set SEC_USER_AGENT in config.py (or the environment) to your name and email first.")

    cik_to_ticker = load_cik_to_ticker()
    sp500 = load_sp500_tickers()
    if sp500:
        print(f"Filtering to {len(sp500)} S&P 500 tickers")

    seen = {p.stem for p in config.RAW_DIR.glob("*.json")}
    collected = len(seen)
    print(f"{collected} filings already in {config.RAW_DIR}")

    for phrase in config.SEARCH_PHRASES:
        if collected >= max_filings:
            break
        page = 1
        skipped = 0
        while collected < max_filings and page <= config.MAX_PAGES_PER_PHRASE:
            print(f"  searching {phrase} page {page} ({skipped} hits skipped so far)", flush=True)
            try:
                hits = search_filings(phrase, config.SEARCH_START_DATE, config.SEARCH_END_DATE, page)
            except requests.HTTPError as e:
                print(f"  search failed for {phrase!r} page {page}: {e}")
                break
            if not hits:
                break
            for hit in hits:
                if collected >= max_filings:
                    break
                rec = hit_to_record(hit)
                key = rec["accession"]
                if key in seen:
                    continue
                ticker = cik_to_ticker.get(rec["cik"])
                if not ticker or (sp500 and ticker.upper() not in sp500):
                    skipped += 1  # no ticker (delisted/private) or outside S&P 500
                    continue
                rec["ticker"] = ticker
                try:
                    rec["text"] = download_document(rec["url"])
                except requests.HTTPError as e:
                    print(f"  download failed {rec['url']}: {e}")
                    continue
                if len(rec["text"]) < 500:
                    continue  # cover page only, nothing to read
                (config.RAW_DIR / f"{key}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
                seen.add(key)
                collected += 1
                print(f"[{collected:>4}] {rec['filing_date']} {ticker:<6} {rec['company_name']}", flush=True)
            page += 1

    print(f"Done. {collected} filings in {config.RAW_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=config.MAX_FILINGS, help="stop after this many filings")
    args = ap.parse_args()
    main(args.max)
