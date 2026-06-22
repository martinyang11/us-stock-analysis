#!/usr/bin/env python3
"""
Backfill SA daily score CSVs and SANN historical_samples.csv.

This script is for historical reconstruction. It uses the latest available
SA fundamental dimensions as a baseline, then recomputes each target date's
24 technical dimensions from a one-year trailing yfinance window.

Example:
  python scripts/backfill_sa_sann_history.py --start 2026-05-22 --end 2026-06-22
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.StockAnalysis.scripts.gtrade_data import compute_technical_scores


LOG = logging.getLogger("backfill")

HEADER = (
    ["date", "variety_id", "variety_name", "month"]
    + [f"dim{i}" for i in range(1, 15)]
    + [f"tech{i}" for i in range(1, 25)]
)
SAMPLE_HEADER = HEADER + ["y", "raw_change"]

TICKER_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "XAU": "GC=F",
    "XAG": "SI=F",
    "WTI": "CL=F",
    "XPT": "PL=F",
    "XPD": "PA=F",
    "HG": "HG=F",
    "NATGAS": "NG=F",
    "BRENT": "BZ=F",
    "SPX500": "^GSPC",
    "NAS100": "^NDX",
    "USA30": "^DJI",
}


def parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d").date())


def compact_date(day: pd.Timestamp) -> str:
    return day.strftime("%Y%m%d")


def display_date(day: pd.Timestamp) -> str:
    return day.strftime("%Y-%m-%d")


def to_yf_ticker(name: str) -> str:
    return TICKER_MAP.get(name.upper(), name.upper())


def normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    rename = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=rename)
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_varieties(data_dir: Path) -> List[dict]:
    meta_path = data_dir / "tradfi_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        varieties = meta.get("varieties", [])
        if varieties:
            return sorted(varieties, key=lambda x: int(x["variety_id"]))

    latest = find_latest_scores_file(data_dir / "daily_scores")
    if latest is None:
        raise FileNotFoundError("No tradfi_meta.json or scores_*.csv found.")
    with latest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {
                "variety_id": int(r["variety_id"]),
                "name": r["variety_name"],
            }
            for r in reader
        ]


def find_latest_scores_file(scores_dir: Path) -> Optional[Path]:
    files = sorted(scores_dir.glob("scores_*.csv"))
    # Ignore the old dashed file when compact files exist, but still allow it as fallback.
    compact = [p for p in files if p.stem.replace("scores_", "").isdigit()]
    return (compact or files)[-1] if files else None


def load_fundamental_template(scores_dir: Path) -> Dict[int, dict]:
    latest = find_latest_scores_file(scores_dir)
    if latest is None:
        LOG.warning("No scores_*.csv found in %s; using neutral 0.5 SA fundamentals.", scores_dir)
        return {}

    LOG.info("Using SA fundamental baseline: %s", latest.name)
    template: Dict[int, dict] = {}
    with latest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = int(row["variety_id"])
            template[vid] = row
    return template


def _literal_assignments(py_path: Path) -> dict:
    """Read top-level literal assignments from run_sa_scoring.py without executing it."""
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except Exception:
            continue
    return values


def load_static_sa_config() -> dict:
    """Load the hard-coded SA scoring constants used by run_sa_scoring.py."""
    values = _literal_assignments(PROJECT_ROOT / "scripts" / "run_sa_scoring.py")
    company_scores = dict(values.get("COMPANY_SCORES", {}))

    for name in values.get("INDEX_NAMES", set()):
        company_scores.setdefault(name, {
            "dim5": 0.65, "dim6": 0.60, "dim7": 0.50, "dim8": 0.55,
            "dim10": 0.55, "dim13": 0.55, "dim14": 0.50,
        })
    for name in values.get("COMMODITY_NAMES", set()):
        company_scores.setdefault(name, {
            "dim5": 0.60, "dim6": 0.50, "dim7": 0.50, "dim8": 0.50,
            "dim10": 0.55, "dim13": 0.50, "dim14": 0.52,
        })
    for name in values.get("ETF_NAMES", set()):
        company_scores.setdefault(name, {
            "dim5": 0.58, "dim6": 0.52, "dim7": 0.50, "dim8": 0.52,
            "dim10": 0.55, "dim13": 0.50, "dim14": 0.50,
        })

    return {
        "D1_BASE": float(values.get("D1_BASE", 0.5)),
        "D2_BASE": float(values.get("D2_BASE", 0.5)),
        "D3_BASE": float(values.get("D3_BASE", 0.5)),
        "D11_BASE": float(values.get("D11_BASE", 0.5)),
        "CATEGORY_ADJUST": values.get("CATEGORY_ADJUST", {"other": (1.0, 1.0, 1.0, 1.0)}),
        "VARIETY_CATEGORY": values.get("VARIETY_CATEGORY", {}),
        "D4_INDUSTRY": values.get("D4_INDUSTRY", {}),
        "COMPANY_SCORES": company_scores,
    }


def download_history(varieties: Iterable[dict], start: pd.Timestamp, end: pd.Timestamp) -> Dict[int, pd.DataFrame]:
    end_exclusive = end + pd.Timedelta(days=1)
    history: Dict[int, pd.DataFrame] = {}
    for i, v in enumerate(varieties, 1):
        vid = int(v["variety_id"])
        name = v["name"]
        ticker = to_yf_ticker(name)
        LOG.info("[%02d] download %s (%s)", i, name, ticker)
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end_exclusive.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            history[vid] = normalize_yf_df(df)
            if history[vid].empty:
                LOG.warning("  %s: no usable yfinance rows", name)
        except Exception as exc:
            LOG.warning("  %s: download failed: %s", name, exc)
            history[vid] = pd.DataFrame()
        time.sleep(0.1)
    return history


def get_us_market_days(history: Dict[int, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    # Prefer SPY if present; otherwise use any index with a normal US calendar.
    for df in history.values():
        if not df.empty and start in df.index:
            break

    spy_days: Optional[pd.DatetimeIndex] = None
    for df in history.values():
        if not df.empty:
            days = df.loc[(df.index >= start) & (df.index <= end)].index
            # A stock/index series in this window has roughly 20-25 rows; crypto has every day.
            if 10 <= len(days) <= 25:
                spy_days = days
                break
    if spy_days is None:
        raise RuntimeError("Could not infer US market days from downloaded data.")
    return [pd.Timestamp(d).normalize() for d in spy_days]


def slice_one_year(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    window_start = day - pd.Timedelta(days=365)
    return df.loc[(df.index >= window_start) & (df.index <= day)].copy()


def score_d12(tech24: List[float]) -> float:
    if not tech24 or len(tech24) < 24:
        return 0.5
    ma_score = (tech24[0] + tech24[1] + tech24[2]) / 3.0
    alignment = tech24[3]
    rsi = tech24[7]
    momentum = (tech24[9] + tech24[10] + tech24[11] + tech24[12]) / 4.0
    boll_pos = tech24[4]
    return round(float(np.clip(
        ma_score * 0.25 + alignment * 0.15 + rsi * 0.20 + momentum * 0.25 + boll_pos * 0.15,
        0.0,
        1.0,
    )), 4)


def compute_d9_from_history(variety: dict, window: pd.DataFrame) -> float:
    base = 0.50
    spread = variety.get("spread_pct")
    try:
        spread = float(spread)
    except Exception:
        spread = None

    # Existing gTrade metadata has appeared both as percent (0.3) and fraction
    # (0.003). Normalize larger values to a fraction for the original thresholds.
    if spread is not None:
        spread_fraction = spread / 100.0 if spread > 1 else spread
        if spread_fraction < 0.0005:
            base += 0.10
        elif spread_fraction < 0.001:
            base += 0.05
        elif spread_fraction > 0.005:
            base -= 0.10

    if not window.empty and "volume" in window.columns:
        vol = float(window.iloc[-1]["volume"])
        if vol > 0:
            log_vol = np.log10(vol)
            if log_vol > 6:
                base += 0.08
            elif log_vol > 5:
                base += 0.04
            elif log_vol < 3:
                base -= 0.08

    return round(float(np.clip(base, 0.0, 1.0)), 4)


def static_fundamental_dims(variety: dict, tech: List[float], window: pd.DataFrame, config: dict) -> Dict[str, str]:
    name = variety["name"]
    cat = config["VARIETY_CATEGORY"].get(name, "other")
    adj = config["CATEGORY_ADJUST"].get(cat, config["CATEGORY_ADJUST"].get("other", (1.0, 1.0, 1.0, 1.0)))
    company = config["COMPANY_SCORES"].get(name, {})

    dims = {
        "dim1": round(float(np.clip(config["D1_BASE"] * adj[0], 0.0, 1.0)), 4),
        "dim2": round(float(np.clip(config["D2_BASE"] * adj[1], 0.0, 1.0)), 4),
        "dim3": round(float(np.clip(config["D3_BASE"] * adj[2], 0.0, 1.0)), 4),
        "dim4": round(float(config["D4_INDUSTRY"].get(cat, 0.50)), 4),
        "dim5": round(float(company.get("dim5", 0.50)), 4),
        "dim6": round(float(company.get("dim6", 0.50)), 4),
        "dim7": round(float(company.get("dim7", 0.50)), 4),
        "dim8": round(float(company.get("dim8", 0.50)), 4),
        "dim9": compute_d9_from_history(variety, window),
        "dim10": round(float(company.get("dim10", 0.50)), 4),
        "dim11": round(float(np.clip(config["D11_BASE"] * adj[3], 0.0, 1.0)), 4),
        "dim12": score_d12(tech),
        "dim13": round(float(company.get("dim13", 0.50)), 4),
        "dim14": round(float(company.get("dim14", 0.50)), 4),
    }
    return {k: f"{v:.4f}" for k, v in dims.items()}


def build_score_row(
    day: pd.Timestamp,
    variety: dict,
    base: dict,
    df: pd.DataFrame,
    static_config: dict,
) -> dict:
    vid = int(variety["variety_id"])
    name = variety["name"]
    window = slice_one_year(df, day)
    if len(window) >= 20:
        tech = compute_technical_scores(window)
    else:
        tech = [0.5] * 24

    row = {
        "date": display_date(day),
        "variety_id": str(vid),
        "variety_name": name,
        "month": str(day.month),
    }
    if base:
        for i in range(1, 15):
            row[f"dim{i}"] = base.get(f"dim{i}", "0.5000")
        row["dim9"] = f"{compute_d9_from_history(variety, window):.4f}"
        row["dim12"] = f"{score_d12(tech):.4f}"
    else:
        row.update(static_fundamental_dims(variety, tech, window, static_config))
    for i, score in enumerate(tech, 1):
        row[f"tech{i}"] = f"{float(score):.4f}"
    return row


def next_close_change(df: pd.DataFrame, day: pd.Timestamp, next_day: Optional[pd.Timestamp]) -> tuple[float, float]:
    if df.empty or next_day is None:
        return 0.5, 0.0
    if day not in df.index or next_day not in df.index:
        # For assets that do not have the exact market-calendar date, use nearest rows on/after dates.
        today_rows = df.loc[df.index >= day]
        next_rows = df.loc[df.index >= next_day]
        if today_rows.empty or next_rows.empty:
            return 0.5, 0.0
        today_close = float(today_rows.iloc[0]["close"])
        next_close = float(next_rows.iloc[0]["close"])
    else:
        today_close = float(df.loc[day, "close"])
        next_close = float(df.loc[next_day, "close"])
    if today_close <= 0:
        return 0.5, 0.0
    raw_change = (next_close - today_close) / today_close
    y = 1.0 / (1.0 + np.exp(-raw_change * 20.0))
    return float(y), float(raw_change)


def write_scores(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def write_samples(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill SA score CSVs and SANN historical samples.")
    parser.add_argument("--start", default="2026-05-22", help="First score date, YYYY-MM-DD.")
    parser.add_argument("--end", default="2026-06-22", help="Last score date, YYYY-MM-DD.")
    parser.add_argument("--history-start", default="2025-02-22", help="Download start date, YYYY-MM-DD.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "skills" / "SANN" / "data"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scores_YYYYMMDD.csv files.")
    parser.add_argument("--keep-existing-samples", action="store_true", help="Merge with existing historical_samples.csv.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    data_dir = Path(args.data_dir).resolve()
    scores_dir = data_dir / "daily_scores"
    start = parse_date(args.start)
    end = parse_date(args.end)
    history_start = parse_date(args.history_start)

    varieties = load_varieties(data_dir)
    template = load_fundamental_template(scores_dir)
    static_config = load_static_sa_config()

    history = download_history(varieties, history_start, end + pd.Timedelta(days=7))
    market_days = get_us_market_days(history, start, end)
    LOG.info("US market days: %s", ", ".join(compact_date(d) for d in market_days))

    all_sample_rows: List[dict] = []
    for idx, day in enumerate(market_days):
        next_day = market_days[idx + 1] if idx + 1 < len(market_days) else None
        score_rows: List[dict] = []

        for variety in varieties:
            vid = int(variety["variety_id"])
            base = template.get(vid, {})
            row = build_score_row(day, variety, base, history.get(vid, pd.DataFrame()), static_config)
            score_rows.append(row)

            y, raw_change = next_close_change(history.get(vid, pd.DataFrame()), day, next_day)
            sample = dict(row)
            sample["y"] = f"{y:.6f}"
            sample["raw_change"] = f"{raw_change:.6f}"
            all_sample_rows.append(sample)

        score_path = scores_dir / f"scores_{compact_date(day)}.csv"
        if score_path.exists() and not args.overwrite:
            LOG.info("skip existing %s (use --overwrite to replace)", score_path.name)
        else:
            write_scores(score_path, score_rows)
            LOG.info("wrote %s (%d rows)", score_path.name, len(score_rows))

    samples_path = data_dir / "historical_samples.csv"
    if args.keep_existing_samples and samples_path.exists():
        existing: List[dict] = []
        with samples_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
        backfill_dates = {display_date(d) for d in market_days}
        existing = [r for r in existing if r.get("date") not in backfill_dates]
        all_sample_rows = existing + all_sample_rows

    write_samples(samples_path, all_sample_rows)
    valid = sum(1 for r in all_sample_rows if abs(float(r["raw_change"])) > 1e-8)
    LOG.info("wrote historical_samples.csv (%d rows, %d train-valid rows)", len(all_sample_rows), valid)
    LOG.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
