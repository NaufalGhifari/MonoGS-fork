#!/usr/bin/env python3
"""
Log a MonoGS run's PSNR/SSIM/LPIPS results to a Google Sheet.

Reads:
    <run_dir>/psnr/before_opt/final_result.json
    <run_dir>/psnr/after_opt/final_result.json
    <run_dir>/plot/stats_final.json   (key "rmse" -> ate_rmse column)

Appends one row with: timestamp, run_name (CLI arg), dir_name, ate_rmse, and every key
from both JSONs (columns named before_opt/<key> and after_opt/<key>).
New keys automatically get new columns, so the sheet adapts to your JSON.

Setup (once):
    pip install gspread
    1. Google Cloud console -> create a service account, enable the Google
       Sheets API + Google Drive API, download its JSON key.
    2. Share your target spreadsheet with the service account's email
       (Editor access).

Usage:
    python log_run_to_sheets.py "path/to/2026-10-03-16-56-30 GT 5000" \
        --run-name "GT 5000" \
        --spreadsheet "MonoGS results" \
        --creds ~/.config/gspread/service_account.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import gspread

FIXED_COLS = ["timestamp", "run_name", "dir_name", "ate_rmse"]


def load_results(run_dir: Path, stage: str) -> dict:
    path = run_dir / "psnr" / stage / "final_result.json"
    if not path.is_file():
        sys.exit(f"Missing results file: {path}")
    with open(path) as f:
        data = json.load(f)
    return flatten(data, prefix=f"{stage}/")


def load_ate_rmse(run_dir: Path):
    path = run_dir / "plot" / "stats_final.json"
    if not path.is_file():
        # sys.exit(f"Missing stats file: {path}")
        print(f"Warning: Missing stats file: {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    if "rmse" not in data:
        sys.exit(f"No 'rmse' key in {path} (keys: {list(data.keys())})")
    return data["rmse"]


def flatten(d, prefix="") -> dict:
    """Flatten nested dicts so every value becomes one column."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, prefix=f"{key}/"))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def open_worksheet(client, args):
    if args.spreadsheet_url:
        sh = client.open_by_url(args.spreadsheet_url)
    else:
        sh = client.open(args.spreadsheet)
    try:
        return sh.worksheet(args.worksheet)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=args.worksheet, rows=1000, cols=26)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="Run directory (contains psnr/before_opt, psnr/after_opt)")
    p.add_argument("--run-name", required=True, help="Name of the run to store in the sheet")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--spreadsheet", help="Spreadsheet title")
    g.add_argument("--spreadsheet-url", help="Spreadsheet URL")
    p.add_argument("--worksheet", default="Sheet1", help="Worksheet/tab name (created if missing)")
    p.add_argument("--creds", type=Path, default=None,
                   help="Service account JSON key (default: gspread's default location)")
    args = p.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    row_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "dir_name": run_dir.name,
        "ate_rmse": load_ate_rmse(run_dir),
    }
    row_data.update(load_results(run_dir, "before_opt"))
    row_data.update(load_results(run_dir, "after_opt"))

    if args.creds:
        client = gspread.service_account(filename=str(args.creds.expanduser()))
    else:
        client = gspread.service_account()
    ws = open_worksheet(client, args)

    # Sync header: keep existing columns, append any new ones.
    header = ws.row_values(1)
    if not header:
        header = list(FIXED_COLS)
    for key in row_data:
        if key not in header:
            header.append(key)
    ws.update(range_name="A1", values=[header])

    row = [row_data.get(col, "") for col in header]
    ws.append_row(row, value_input_option="USER_ENTERED", table_range="A1")

    print(f"Logged '{args.run_name}' ({run_dir.name}) to worksheet '{ws.title}'.")


if __name__ == "__main__":
    main()