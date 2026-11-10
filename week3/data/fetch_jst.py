"""Download the Jordà-Schularick-Taylor Macrohistory database (release 6) into this folder.

Run once from the Week3_Lab folder:   uv run data/fetch_jst.py

Source: www.macrohistory.net/database (Jordà, Schularick and Taylor 2017, NBER Macroeconomics Annual).
If the direct link below has moved, download "JSTdatasetR6.xlsx" from that page by hand and save it here.
"""
from pathlib import Path

import requests

URL = "https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.xlsx"
OUT = Path(__file__).resolve().parent / "JSTdatasetR6.xlsx"

r = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0 (course data download)"})
r.raise_for_status()
if not r.content.startswith(b"PK"):               # an .xlsx file is a zip archive
    raise SystemExit("The download is not an Excel file: get JSTdatasetR6.xlsx by hand from "
                     "www.macrohistory.net/database and save it in data/.")
OUT.write_bytes(r.content)
print(f"Saved {OUT} ({len(r.content) / 1e6:.1f} MB)")
