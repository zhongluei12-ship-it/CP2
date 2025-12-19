"""
dataset_builder.py

Offline tool to:
 - read raw URL lists (tranco_list.csv, verified_online.csv)
 - fetch HTML (HTTPS first)
 - extract features using feature_extraction.create_vector(soup, url)
 - append to structured_data_legitimate.csv and structured_data_phishing.csv
"""

import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning, LocationParseError

import feature_extraction as fe

disable_warnings(InsecureRequestWarning)

# ---------------------------
# CONFIG
# ---------------------------
BATCH_SIZE = 100
REQUEST_TIMEOUT = 8
DELAY_BETWEEN_REQUESTS = 0.25

# Slice (0-based like python slicing). Set None for full range.
LEGIT_START = 15100
LEGIT_END = 20000
PHISH_START = 15000
PHISH_END = 20000

LEGIT_INPUT = "tranco_list.csv"
LEGIT_OUTPUT = "structured_data_legitimate.csv"

PHISH_INPUT = "verified_online.csv"
PHISH_OUTPUT = "structured_data_phishing.csv"

SKIPPED_LOG = "skipped_urls.txt"

# Browser-like headers reduce 403 blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# Columns for output CSV
COLUMNS = fe.COLUMNS + ["URL", "label"]


# ---------------------------
# LOGGING
# ---------------------------
def log_skipped(url: str, reason: str):
    try:
        with open(SKIPPED_LOG, "a", encoding="utf-8") as f:
            f.write(f"{str(url)}\t{reason}\n")
    except Exception:
        pass


# ---------------------------
# URL NORMALIZATION (HTTPS FIRST)
# ---------------------------
def is_valid_hostname(host: str) -> bool:
    if not host:
        return False
    host = host.strip()
    if host.startswith(".") or host.endswith("."):
        return False
    if ".." in host:
        return False
    parts = host.split(".")
    for part in parts:
        if len(part) == 0 or len(part) > 63:
            return False
    return True


def normalize_url(raw: str) -> str | None:
    """
    Normalize a raw string into a valid URL, preferring HTTPS.
    Returns None if invalid.
    """
    if not isinstance(raw, str):
        return None

    u = raw.strip()
    if not u:
        return None

    if u.startswith(("javascript:", "mailto:", "data:")):
        return None

    # If already has scheme, validate hostname
    if u.startswith(("http://", "https://")):
        try:
            p = urlparse(u)
            if not is_valid_hostname(p.hostname or ""):
                return None
            return u
        except Exception:
            return None

    # Try HTTPS first
    https_url = "https://" + u
    try:
        p = urlparse(https_url)
        if not is_valid_hostname(p.hostname or ""):
            return None
        return https_url
    except Exception:
        return None


def load_urls_from_csv(path: str) -> list[str]:
    """
    Load URL-like column from CSV:
      - if a column named URL/url exists, use that
      - otherwise, fallback to second column then first.
    """
    df = pd.read_csv(path)
    cols_lower = [c.lower() for c in df.columns]

    if "url" in cols_lower:
        col_name = df.columns[cols_lower.index("url")]
        raws = df[col_name].astype(str).tolist()
    else:
        raws = df.iloc[:, 1].astype(str).tolist() if df.shape[1] > 1 else df.iloc[:, 0].astype(str).tolist()

    urls = []
    for r in raws:
        n = normalize_url(r)
        if n:
            urls.append(n)
        else:
            log_skipped(r, "normalize_failed")
    return urls


# ---------------------------
# FETCH HTML
# ---------------------------
def fetch_html(url: str) -> tuple[int | None, bytes | None, str, str | None]:
    """
    Fetch HTML with redirects. If HTTPS fails, optionally try HTTP fallback.
    Returns: (status_code, content_bytes, final_url, error_msg)
    """
    last_err = None

    # 1) Try the given URL first (usually https://)
    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            verify=False
        )
        return r.status_code, r.content, r.url, None
    except (LocationParseError, requests.exceptions.InvalidURL) as e:
        return None, None, url, f"bad_url: {e}"
    except requests.RequestException as e:
        last_err = str(e)

    # 2) If it was https://something and failed, try http:// fallback
    try:
        p = urlparse(url)
        if (p.scheme or "").lower() == "https":
            http_url = "http://" + (p.netloc + (p.path or ""))
            r = requests.get(
                http_url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=False
            )
            return r.status_code, r.content, r.url, None
    except Exception as e:
        last_err = f"{last_err} | http_fallback_failed: {e}"

    return None, None, url, last_err


def scrape_to_soup(url: str) -> tuple[BeautifulSoup | None, str | None]:
    """
    Return (soup, final_url). soup is None if fetch failed or not HTML.
    """
    status, content, final_url, err = fetch_html(url)

    if err:
        log_skipped(url, f"fetch_error: {err}")
        return None, None

    if status != 200 or not content:
        log_skipped(url, f"http_status: {status}")
        return None, None

    # Basic content sanity check
    if len(content) < 200:
        log_skipped(url, f"content_too_small: {len(content)}")
        return None, None

    try:
        soup = BeautifulSoup(content, "html.parser")
        return soup, final_url
    except Exception as e:
        log_skipped(url, f"bs4_parse_error: {e}")
        return None, None


# ---------------------------
# CSV APPEND
# ---------------------------
def append_rows_to_csv(rows: list[list], output_path: str):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    if not os.path.exists(output_path):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, mode="a", header=False, index=False)


# ---------------------------
# MAIN BUILD FUNCTION
# ---------------------------
def build_structured_dataset(input_csv: str, output_csv: str, label_value: int,
                             start: int = 0, end: int | None = None):
    urls = load_urls_from_csv(input_csv)

    # slicing
    urls = urls[start:end] if end is not None else urls[start:]

    print(f"[INFO] Loaded {len(urls)} URLs from {input_csv} (after slicing)")

    expected_feat_len = len(fe.COLUMNS)

    for batch_start in range(0, len(urls), BATCH_SIZE):
        batch = urls[batch_start:batch_start + BATCH_SIZE]
        rows = []
        print(f"\n--- Batch {batch_start}..{batch_start + len(batch) - 1} ({len(batch)} URLs) ---")

        for i, url in enumerate(batch, start=batch_start + 1):
            soup, final_url = scrape_to_soup(url)

            if not soup or not final_url:
                time.sleep(DELAY_BETWEEN_REQUESTS)
                continue

            try:
                # IMPORTANT: pass final_url so your URL features (https, host length, etc.) are correct
                vec = fe.create_vector(soup, final_url)

                # safety: pad/truncate if mismatch
                if len(vec) != expected_feat_len:
                    log_skipped(final_url, f"vector_len_mismatch: {len(vec)} vs {expected_feat_len}")
                    if len(vec) < expected_feat_len:
                        vec = vec + [0] * (expected_feat_len - len(vec))
                    else:
                        vec = vec[:expected_feat_len]

                # Add metadata columns at the end
                vec.append(final_url)
                vec.append(label_value)

                rows.append(vec)
                print(f"[OK] {i}: {final_url}")

            except Exception as ex:
                log_skipped(final_url, f"feature_extraction_failed: {ex}")
                print(f"[ERROR] feature extraction failed for {final_url}: {ex}")

            time.sleep(DELAY_BETWEEN_REQUESTS)

        append_rows_to_csv(rows, output_csv)
        print(f"[INFO] Appended {len(rows)} rows to {output_csv}")


# ---------------------------
# CLI ENTRYPOINT
# ---------------------------
if __name__ == "__main__":
    # (Optional) delete old outputs to rebuild clean dataset
    # If you want fresh datasets each run, uncomment these:
    # if os.path.exists(LEGIT_OUTPUT): os.remove(LEGIT_OUTPUT)
    # if os.path.exists(PHISH_OUTPUT): os.remove(PHISH_OUTPUT)

    # Legitimate sites (label 0)
    build_structured_dataset(
        LEGIT_INPUT,
        LEGIT_OUTPUT,
        label_value=0,
        start=LEGIT_START,
        end=LEGIT_END
    )

    # Phishing sites (label 1)
    build_structured_dataset(
        PHISH_INPUT,
        PHISH_OUTPUT,
        label_value=1,
        start=PHISH_START,
        end=PHISH_END
    )

    print("[DONE] Dataset building finished.")
