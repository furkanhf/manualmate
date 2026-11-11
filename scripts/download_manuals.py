"""Download product-manual PDFs from a list of links and check each file.

Input:  data/download_list.csv
        columns: manual_id,brand,model,category,language,filename,source_url,note
        (the first 7 columns are the same as data/manifest.csv)
Output: data/raw/<filename>        the PDFs (data/raw is gitignored)
        data/download_report.csv   one row per manual: status + checks

A note that starts with DROPPED or SCANNED skips the row (nothing is downloaded).

Rows with an empty source_url are "BY HAND" rows: download the PDF in your
browser, save it as data/raw/<filename>, then run this script again. The
script then only checks the file.

Run from the project folder (or press the Run button in VS Code):
    python scripts/download_manuals.py                 # all rows
    python scripts/download_manuals.py ac_005 ac_006   # some rows
    python scripts/download_manuals.py --category ac   # one category
    python scripts/download_manuals.py --force         # download again

This script does NOT change data/manifest.csv. Claude reads the report first,
then adds the good rows to the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import pymupdf
import requests

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIST_PATH = PROJECT_ROOT / "data" / "download_list.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "data" / "download_report.csv"

# --- Settings ----------------------------------------------------------------
MIN_PAGES = 10          # fewer pages -> SHORT (usually a quick guide)
LOW_TEXT_CHARS = 50     # same rule as pdf_parser.py: page with < 50 chars
MAX_LOW_TEXT = 0.5      # more than 50% low-text pages -> SCANNED (needs OCR)
MIN_EN_RATIO = 0.80     # share of letters that are a-z/A-Z; lower -> NOT_EN
TRIES = 3               # download tries per link
TIMEOUT = 60            # seconds per try

# Sites we never use (AI-written fakes or copies; see docs/PROGRESS.md)
BLOCKED = ("manuals.plus", "download_pdfs", "manualslib", "manualzz",
           "manualmachine", "scribd", "notice-facile", "manuals.online")

HEADERS = {
    # Some brand sites refuse requests that do not look like a browser.
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/130.0 Safari/537.36"),
    "Accept": "application/pdf,*/*;q=0.8",
}


def download(url: str, dest: Path) -> str:
    """Download url to dest. Return "" if OK, else a short error text."""
    last_error = ""
    for attempt in range(1, TRIES + 1):
        try:
            with requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                              stream=True, allow_redirects=True) as r:
                if r.status_code != 200:
                    last_error = f"HTTP {r.status_code}"
                else:
                    part = dest.with_suffix(".part")
                    with open(part, "wb") as f:
                        for block in r.iter_content(1 << 16):
                            f.write(block)
                    part.replace(dest)
                    return ""
        except requests.RequestException as e:
            last_error = type(e).__name__
        if attempt < TRIES:
            time.sleep(3 * attempt)
    return last_error


def check_pdf(path: Path) -> dict:
    """Open the PDF and measure it. Return a dict of checks."""
    info = {"pages": 0, "chars": 0, "low_text_pages": 0, "en_ratio": "",
            "size_kb": round(path.stat().st_size / 1024), "error": ""}
    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            info["error"] = "not a PDF (maybe an HTML page)"
            return info
    try:
        doc = pymupdf.open(path)
    except Exception as e:                      # broken file
        info["error"] = f"cannot open: {type(e).__name__}"
        return info
    if doc.needs_pass:
        info["error"] = "password protected"
        return info
    letters = ascii_letters = 0
    for page in doc:
        text = page.get_text()
        n = len(text.strip())
        info["chars"] += n
        if n < LOW_TEXT_CHARS:
            info["low_text_pages"] += 1
        for ch in text:
            if ch.isalpha():
                letters += 1
                if ch.isascii():
                    ascii_letters += 1
    info["pages"] = doc.page_count
    info["en_ratio"] = round(ascii_letters / letters, 2) if letters else 0.0
    doc.close()
    return info


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verdict(info: dict) -> str:
    if info["error"]:
        return "BAD_FILE"
    if info["pages"] and info["low_text_pages"] / info["pages"] > MAX_LOW_TEXT:
        return "SCANNED"
    if info["en_ratio"] != "" and info["en_ratio"] < MIN_EN_RATIO:
        return "NOT_EN"
    if info["pages"] < MIN_PAGES:
        return "SHORT"
    return "OK"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="*", help="manual_ids to process (default all)")
    ap.add_argument("--category", help="only this category, e.g. ac")
    ap.add_argument("--force", action="store_true", help="download again")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(LIST_PATH, encoding="utf-8")))
    if args.ids:
        rows = [r for r in rows if r["manual_id"] in args.ids]
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]
    if not rows:
        print("No rows selected. Check the ids / --category.")
        return

    # Existing manuals: catch id clashes and duplicate PDFs.
    manifest = list(csv.DictReader(open(MANIFEST_PATH, encoding="utf-8")))
    manifest_ids = {m["manual_id"] for m in manifest}
    known_hashes = {}
    for m in manifest:
        p = RAW_DIR / m["filename"]
        if p.exists():
            known_hashes[sha256(p)] = m["manual_id"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    print(f"{len(rows)} manuals -> {RAW_DIR}\n")
    for r in rows:
        mid, url, dest = r["manual_id"], r["source_url"].strip(), RAW_DIR / r["filename"]
        out = {"manual_id": mid, "filename": r["filename"], "status": "",
               "pages": "", "chars": "", "low_text_pages": "", "en_ratio": "",
               "size_kb": "", "detail": "", "source_url": url}

        if r["note"].upper().startswith(("DROPPED", "SCANNED")):
            out["status"], out["detail"] = "SKIPPED", r["note"]
        elif mid in manifest_ids and dest.exists():
            out["status"], out["detail"] = "IN_MANIFEST", "already added"
        elif mid in manifest_ids:
            out["status"], out["detail"] = "ID_IN_MANIFEST", "pick a new manual_id"
        elif url and any(b in url.lower() for b in BLOCKED):
            out["status"], out["detail"] = "BLOCKED_SITE", "use an official brand site"
        elif not url and not dest.exists():
            out["status"] = "BY_HAND"
            out["detail"] = f"save the PDF as data/raw/{r['filename']} ({r['note']})"
        else:
            if url and (args.force or not dest.exists()):
                print(f"  {mid}: downloading ...", end=" ", flush=True)
                err = download(url, dest)
                print(err or "done")
                if err:
                    out["status"], out["detail"] = "DOWNLOAD_FAILED", err
            if not out["status"]:
                info = check_pdf(dest)
                out.update({k: info[k] for k in
                            ("pages", "chars", "low_text_pages", "en_ratio", "size_kb")})
                out["status"] = verdict(info)
                out["detail"] = info["error"]
                if out["status"] == "BAD_FILE" and url:
                    # Move it away, so the next run downloads it again.
                    dest.replace(dest.with_suffix(".bad"))
                    out["detail"] += f" -> moved to {dest.stem}.bad"
                if out["status"] != "BAD_FILE":
                    dup = known_hashes.get(sha256(dest))
                    if dup:
                        out["status"], out["detail"] = "DUPLICATE", f"same file as {dup}"
        report.append(out)
        print(f"{mid:8} {out['status']:15} pages={out['pages']!s:>4} "
              f"low_text={out['low_text_pages']!s:>3} en={out['en_ratio']!s:>4} "
              f"{out['detail']}")

    # Merge with an older report, so runs on a few ids keep the other rows.
    old = {}
    if REPORT_PATH.exists():
        old = {o["manual_id"]: o for o in csv.DictReader(open(REPORT_PATH, encoding="utf-8"))}
    for o in report:
        old[o["manual_id"]] = o
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(sorted(old.values(), key=lambda o: o["manual_id"]))

    counts = {}
    for o in report:
        counts[o["status"]] = counts.get(o["status"], 0) + 1
    print("\nSUMMARY: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
