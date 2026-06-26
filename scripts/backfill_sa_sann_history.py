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
import io
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skills.StockAnalysis.scripts.gtrade_data import compute_technical_scores
from skills.common.tradfi_universe import TRADFI_SYMBOLS


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
    "SPCX": "^GSPC",
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


def cache_name(ticker: str) -> str:
    return ticker.replace("^", "_IDX_").replace("=", "_F_").replace("/", "_")


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


def normalize_ohlcv_df(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    required = ["open", "high", "low", "close"]
    if date_col not in df.columns or any(c not in df.columns for c in required):
        return pd.DataFrame()
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, "close"])
    if df.empty:
        return pd.DataFrame()
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["close"]).sort_index()


def stooq_candidates(name: str) -> List[str]:
    symbol = name.upper()
    if symbol in {"SPCX", "SPX500"}:
        return ["^spx", "spy.us"]
    return [f"{symbol.lower()}.us"]


def download_stooq_history(
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retries: int = 3,
) -> pd.DataFrame:
    """Download daily OHLCV from Stooq CSV endpoint, independent of Yahoo."""
    for stooq_symbol in stooq_candidates(name):
        for base_url in ("https://stooq.com/q/d/l/", "http://stooq.com/q/d/l/"):
            for attempt in range(1, retries + 1):
                try:
                    resp = requests.get(
                        base_url,
                        params={
                            "s": stooq_symbol,
                            "i": "d",
                            "d1": start.strftime("%Y%m%d"),
                            "d2": end.strftime("%Y%m%d"),
                        },
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    text = resp.text.strip()
                    if not text or text.lower().startswith("no data"):
                        break
                    hist = normalize_ohlcv_df(pd.read_csv(io.StringIO(text)))
                    if not hist.empty:
                        LOG.info("  stooq hit %s via %s rows=%d", name, stooq_symbol, len(hist))
                        return hist
                    LOG.warning(
                        "  %s: stooq %s returned unparsed text: %s",
                        name,
                        stooq_symbol,
                        text[:160].replace("\n", " | "),
                    )
                    break
                except Exception as exc:
                    LOG.warning(
                        "  %s: stooq %s attempt %d/%d failed: %s",
                        name,
                        stooq_symbol,
                        attempt,
                        retries,
                        exc,
                    )
                    time.sleep(min(2 * attempt, 8))
    return pd.DataFrame()


def load_cached_history(cache_dir: Path, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = cache_dir / f"{cache_name(ticker)}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            return pd.DataFrame()
        df = df.set_index("date").sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if df.empty:
            return pd.DataFrame()
        if df.index.min() <= start and df.index.max() >= end:
            return df
    except Exception as exc:
        LOG.warning("  cache read failed %s: %s", ticker, exc)
    return pd.DataFrame()


def save_cached_history(cache_dir: Path, ticker: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "date"
    out.reset_index().to_csv(cache_dir / f"{cache_name(ticker)}.csv", index=False)


def download_yahoo_chart(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Lightweight Yahoo chart fallback. Returns normalized OHLCV."""
    period1 = int(pd.Timestamp(start).timestamp())
    period2 = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(
            url,
            params={"period1": period1, "period2": period2, "interval": "1d", "events": "history"},
            headers=headers,
            timeout=20,
        )
        if resp.status_code == 429:
            raise RuntimeError("Yahoo chart 429 rate limited")
        resp.raise_for_status()
        result = (resp.json().get("chart", {}).get("result") or [None])[0]
        if not result:
            return pd.DataFrame()
        ts = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        if not ts or not quote:
            return pd.DataFrame()
        df = pd.DataFrame({
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
        }, index=pd.to_datetime(ts, unit="s").normalize())
        return normalize_yf_df(df)
    except Exception as exc:
        LOG.warning("  %s: yahoo chart fallback failed: %s", ticker, exc)
        return pd.DataFrame()


def extract_batch_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):
        return normalize_yf_df(df)

    levels = df.columns.names
    # yfinance can return either (ticker, field) or (field, ticker).
    for level in (0, 1):
        values = [str(v) for v in df.columns.get_level_values(level)]
        if ticker in values:
            try:
                part = df.xs(ticker, axis=1, level=level)
                return normalize_yf_df(part)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


def batch_download_history(
    missing: List[tuple[int, str, str]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> Dict[int, pd.DataFrame]:
    if not missing:
        return {}
    tickers = sorted({ticker for _, _, ticker in missing})
    LOG.info("batch download %d tickers via yfinance", len(tickers))
    try:
        df = yf.download(
            " ".join(tickers),
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
            group_by="ticker",
        )
    except Exception as exc:
        LOG.warning("batch download failed: %s", exc)
        return {}

    by_ticker = {ticker: extract_batch_ticker(df, ticker) for ticker in tickers}
    result = {}
    for vid, name, ticker in missing:
        hist = by_ticker.get(ticker, pd.DataFrame())
        if not hist.empty:
            save_cached_history(cache_dir, ticker, hist)
            result[vid] = hist
            LOG.info("  cached %s (%s) rows=%d", name, ticker, len(hist))
    return result


def load_varieties(data_dir: Path) -> List[dict]:
    if TRADFI_SYMBOLS:
        return [
            {"variety_id": i, "name": name}
            for i, name in enumerate(TRADFI_SYMBOLS)
        ]

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
    current_id_by_name = {name: i for i, name in enumerate(TRADFI_SYMBOLS)}
    with latest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("variety_name", row.get("crypto_name", "")).upper()
            if current_id_by_name:
                if name not in current_id_by_name:
                    continue
                vid = current_id_by_name[name]
            else:
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


def download_history(
    varieties: Iterable[dict],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    refresh_cache: bool = False,
    retries: int = 3,
    sleep_seconds: float = 8.0,
    sources: Optional[List[str]] = None,
    stooq_sleep: float = 1.0,
    stooq_retries: int = 3,
) -> Dict[int, pd.DataFrame]:
    end_exclusive = end + pd.Timedelta(days=1)
    sources = [s.strip().lower() for s in (sources or ["yahoo"]) if s.strip()]
    history: Dict[int, pd.DataFrame] = {}
    missing: List[tuple[int, str, str]] = []

    for v in varieties:
        vid = int(v["variety_id"])
        name = v["name"]
        ticker = to_yf_ticker(name)
        cached = pd.DataFrame() if refresh_cache else load_cached_history(cache_dir, ticker, start, end)
        if not cached.empty:
            history[vid] = cached
            LOG.info("cache hit %s (%s) rows=%d", name, ticker, len(cached))
        else:
            missing.append((vid, name, ticker))

    for source in sources:
        if not missing:
            break
        if source == "yahoo":
            before = list(missing)
            history.update(batch_download_history(missing, start, end, cache_dir))
            missing = [
                (vid, name, ticker)
                for vid, name, ticker in before
                if vid not in history or history[vid].empty
            ]
        elif source == "stooq":
            stooq_missing: List[tuple[int, str, str]] = []
            for i, (vid, name, ticker) in enumerate(missing, 1):
                hist = download_stooq_history(name, start, end, retries=stooq_retries)
                if not hist.empty:
                    save_cached_history(cache_dir, ticker, hist)
                    history[vid] = hist
                else:
                    stooq_missing.append((vid, name, ticker))
                if stooq_sleep > 0 and i < len(missing):
                    time.sleep(stooq_sleep)
            missing = stooq_missing
        else:
            LOG.warning("unknown history source ignored: %s", source)

    for i, (vid, name, ticker) in enumerate(missing, 1):
        if vid in history and not history[vid].empty:
            continue
        if "yahoo" not in sources:
            LOG.warning("  %s: no history from selected sources; using fallbacks/placeholders", name)
            history[vid] = pd.DataFrame()
            continue
        LOG.info("[%02d/%02d] download missing %s (%s)", i, len(missing), name, ticker)
        hist = pd.DataFrame()
        for attempt in range(1, retries + 1):
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
                hist = normalize_yf_df(df)
                if hist.empty:
                    hist = download_yahoo_chart(ticker, start, end)
                if not hist.empty:
                    break
                LOG.warning("  %s: no usable rows on attempt %d/%d", name, attempt, retries)
            except Exception as exc:
                LOG.warning("  %s: download failed attempt %d/%d: %s", name, attempt, retries, exc)
            delay = sleep_seconds * attempt + random.uniform(0.0, 2.0)
            LOG.info("  sleep %.1fs before retry", delay)
            time.sleep(delay)

        if not hist.empty:
            save_cached_history(cache_dir, ticker, hist)
        else:
            LOG.warning("  %s: giving neutral technical/y placeholders", name)
        history[vid] = hist

        delay = sleep_seconds + random.uniform(0.0, 2.0)
        LOG.info("  throttle sleep %.1fs", delay)
        time.sleep(delay)
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
        LOG.warning("Could not infer US market days from downloaded data; falling back to business days.")
        spy_days = pd.bdate_range(start=start, end=end)
    return [pd.Timestamp(d).normalize() for d in spy_days]


def slice_one_year(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
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


def fallback_tech_from_sample(row: Optional[dict]) -> Optional[List[float]]:
    if not row:
        return None
    try:
        tech = [float(row[f"tech{i}"]) for i in range(1, 25)]
    except Exception:
        return None
    return tech if len(tech) == 24 else None


def fallback_target_from_sample(row: Optional[dict]) -> Optional[tuple[float, float]]:
    if not row:
        return None
    try:
        return float(row.get("y", "0.5")), float(row.get("raw_change", "0.0"))
    except Exception:
        return None


def build_score_row(
    day: pd.Timestamp,
    variety: dict,
    base: dict,
    df: pd.DataFrame,
    static_config: dict,
    existing_sample: Optional[dict] = None,
) -> dict:
    vid = int(variety["variety_id"])
    name = variety["name"]
    window = slice_one_year(df, day)
    if len(window) >= 20:
        tech = compute_technical_scores(window)
    else:
        tech = fallback_tech_from_sample(existing_sample) or [0.5] * 24

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


def cleanup_stale_score_files(scores_dir: Path, start: pd.Timestamp, end: pd.Timestamp, keep_days: List[pd.Timestamp]) -> None:
    keep = {compact_date(d) for d in keep_days}
    for day in pd.bdate_range(start=start, end=end):
        stamp = compact_date(pd.Timestamp(day))
        if stamp in keep:
            continue
        path = scores_dir / f"scores_{stamp}.csv"
        if path.exists():
            path.unlink()
            LOG.info("removed stale non-market score file %s", path.name)


def load_existing_sample_lookup(samples_path: Path) -> Dict[tuple[str, str], dict]:
    if not samples_path.exists():
        return {}
    lookup: Dict[tuple[str, str], dict] = {}
    try:
        with samples_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row.get("date")
                name = row.get("variety_name")
                if date and name:
                    lookup[(date, name.upper())] = row
    except Exception as exc:
        LOG.warning("Could not read existing historical_samples.csv as fallback: %s", exc)
    return lookup


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill SA score CSVs and SANN historical samples.")
    parser.add_argument("--start", default="2026-05-22", help="First score date, YYYY-MM-DD.")
    parser.add_argument("--end", default="2026-06-22", help="Last score date, YYYY-MM-DD.")
    parser.add_argument("--history-start", default="", help="Download start date, YYYY-MM-DD. Default: START minus --history-years.")
    parser.add_argument("--history-years", type=float, default=2.0, help="Years of K-line history to download before START when --history-start is omitted.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "skills" / "SANN" / "data"))
    parser.add_argument("--cache-dir", default="", help="K-line cache dir. Default: DATA_DIR/kline_cache.")
    parser.add_argument("--sources", default="yahoo", help="Comma-separated history sources in order: yahoo,stooq. Default: yahoo.")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore existing kline cache and download again.")
    parser.add_argument("--stooq-sleep", type=float, default=1.0, help="Seconds to sleep between Stooq CSV requests.")
    parser.add_argument("--stooq-retries", type=int, default=3, help="Retries for each Stooq symbol/base URL.")
    parser.add_argument("--download-sleep", type=float, default=8.0, help="Seconds to sleep between fallback single-ticker downloads.")
    parser.add_argument("--retries", type=int, default=3, help="Retries for each missing ticker.")
    parser.add_argument("--use-score-template", action="store_true", help="Reuse latest scores_*.csv fundamentals by symbol name. Default rebuilds from static SA config.")
    parser.add_argument("--ignore-score-template", action="store_true", help="Deprecated no-op; static SA config is now the default.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scores_YYYYMMDD.csv files.")
    parser.add_argument("--keep-existing-samples", action="store_true", help="Merge with existing historical_samples.csv.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    data_dir = Path(args.data_dir).resolve()
    scores_dir = data_dir / "daily_scores"
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else data_dir / "kline_cache"
    start = parse_date(args.start)
    end = parse_date(args.end)
    if args.history_start:
        history_start = parse_date(args.history_start)
    else:
        history_start = start - pd.Timedelta(days=int(args.history_years * 365))
        LOG.info(
            "history-start omitted; downloading %.2f years of K-lines from %s",
            args.history_years,
            display_date(history_start),
        )

    varieties = load_varieties(data_dir)
    static_config = load_static_sa_config()
    template = load_fundamental_template(scores_dir) if args.use_score_template else {}
    if args.use_score_template:
        LOG.info("Using latest score baseline for fundamental dimensions.")
    else:
        LOG.info("Using static SA scoring config for fundamental dimensions; not reusing latest score template.")

    history = download_history(
        varieties,
        history_start,
        end + pd.Timedelta(days=7),
        cache_dir=cache_dir,
        refresh_cache=args.refresh_cache,
        retries=args.retries,
        sleep_seconds=args.download_sleep,
        sources=args.sources.split(","),
        stooq_sleep=args.stooq_sleep,
        stooq_retries=args.stooq_retries,
    )
    market_days = get_us_market_days(history, start, end)
    LOG.info("US market days: %s", ", ".join(compact_date(d) for d in market_days))
    if args.overwrite:
        cleanup_stale_score_files(scores_dir, start, end, market_days)

    samples_path = data_dir / "historical_samples.csv"
    existing_sample_lookup = load_existing_sample_lookup(samples_path)
    all_sample_rows: List[dict] = []
    for idx, day in enumerate(market_days):
        next_day = market_days[idx + 1] if idx + 1 < len(market_days) else None
        score_rows: List[dict] = []

        for variety in varieties:
            vid = int(variety["variety_id"])
            base = template.get(vid, {})
            existing_sample = existing_sample_lookup.get((display_date(day), variety["name"].upper()))
            hist = history.get(vid, pd.DataFrame())
            row = build_score_row(day, variety, base, hist, static_config, existing_sample)
            score_rows.append(row)

            y, raw_change = next_close_change(hist, day, next_day)
            fallback_target = fallback_target_from_sample(existing_sample)
            if abs(raw_change) <= 1e-12 and fallback_target is not None:
                y, raw_change = fallback_target
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
