"""NSE archive data: official daily OHLC (price fallback), delivery quantities (sector heatmap), FII-DII flows.

Each trading day is cached as a small parquet so daily updates are cheap.
"""
import datetime as dt
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from ..config import STORAGE

log = logging.getLogger(__name__)
NSE_DIR = STORAGE / "nse"
DELIV_DIR = NSE_DIR / "deliv"
DELIV_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DAYS = 120  # the heatmap needs ~50 sessions of delivery history
FII_PATH = NSE_DIR / "fii_dii.csv"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}


def _get(url: str, tries: int = 3) -> bytes | None:
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:
            if i == tries - 1:
                log.warning("NSE fetch failed %s: %s", url, e)
            time.sleep(1.5 * (i + 1))
    return None


def fetch_delivery(day: dt.date) -> pd.DataFrame | None:
    raw = _get(f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv")
    if not raw:
        return None
    df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[df.SERIES.astype(str).str.strip() == "EQ"]
    out = pd.DataFrame({
        "symbol": df.SYMBOL.astype(str).str.strip(),
        "deliv_per": pd.to_numeric(df.DELIV_PER, errors="coerce"),
        "deliv_qty": pd.to_numeric(df.DELIV_QTY, errors="coerce"),
    })
    out["date"] = pd.Timestamp(day)
    return out


BHAV_DIR = NSE_DIR / "bhav"
BHAV_DIR.mkdir(parents=True, exist_ok=True)


def bhav_ohlc(day: dt.date) -> pd.DataFrame | None:
    """Official NSE daily OHLCV (EQ series) for one day, indexed by symbol. Cached."""
    path = BHAV_DIR / f"{day:%Y%m%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = _get(f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv")
    if not raw:
        return None
    df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[df.SERIES.astype(str).str.strip() == "EQ"]
    out = pd.DataFrame({
        "open": pd.to_numeric(df.OPEN_PRICE, errors="coerce").values,
        "high": pd.to_numeric(df.HIGH_PRICE, errors="coerce").values,
        "low": pd.to_numeric(df.LOW_PRICE, errors="coerce").values,
        "close": pd.to_numeric(df.CLOSE_PRICE, errors="coerce").values,
        "volume": pd.to_numeric(df.TTL_TRD_QNTY, errors="coerce").values,
    }, index=df.SYMBOL.astype(str).str.strip().values)
    out.to_parquet(path)
    return out


def _cache_day(day: dt.date) -> None:
    path = DELIV_DIR / f"{day:%Y%m%d}.parquet"
    if not path.exists():
        df = fetch_delivery(day)
        if df is not None and len(df):
            df.to_parquet(path)


def trading_days(years: float) -> list[dt.date]:
    from .prices import load

    idx = load("RELIANCE").index
    start = pd.Timestamp.now() - pd.DateOffset(days=int(365 * years))
    return [d.date() for d in idx if d >= start]


def backfill(years: float = 0.35, workers: int = 3) -> None:
    """Fetch recent delivery history (default ~4 months, enough for the heatmap's 50-session averages)."""
    days = [d for d in trading_days(years) if not (DELIV_DIR / f"{d:%Y%m%d}.parquet").exists()]
    log.info("NSE backfill: %d days to fetch", len(days))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(_cache_day, days), 1):
            if i % 50 == 0:
                log.info("NSE backfill %d/%d", i, len(days))
    _cache.clear()


def update_recent(days: int = 7) -> None:
    for d in trading_days(days / 365):
        _cache_day(d)
    _cache.clear()


_cache: dict = {}


def nse_panel() -> pd.DataFrame:
    """Recent cached days merged: index (symbol, date) → deliv_per, deliv_qty."""
    if "panel" not in _cache:
        files = sorted(DELIV_DIR.glob("*.parquet"))[-PANEL_DAYS:]
        dl = [pd.read_parquet(p) for p in files]
        _cache["panel"] = pd.concat(dl).set_index(["symbol", "date"]).sort_index() if dl else pd.DataFrame()
    return _cache["panel"]


def for_symbol(symbol: str) -> pd.DataFrame | None:
    if "by_sym" not in _cache:
        p = nse_panel()
        _cache["by_sym"] = {} if p.empty else {
            s: g.droplevel("symbol") for s, g in p.groupby(level="symbol", sort=False)}
    return _cache["by_sym"].get(symbol)


_fii_cache: dict = {"ts": 0.0, "data": {}}


def fii_dii(max_age: int = 600) -> dict:
    """Today's FII/DII net flows (Rs crore), cached for `max_age` seconds."""
    if time.time() - _fii_cache["ts"] < max_age and _fii_cache["data"]:
        return _fii_cache["data"]
    data = _fetch_fii_dii()
    if data:
        _fii_cache.update(ts=time.time(), data=data)
    return data or _fii_cache["data"]


def _fetch_fii_dii() -> dict:
    """Fetch today's FII/DII cash market net flows; also appended to a local history file."""
    s = requests.Session()
    s.headers.update(UA)
    try:
        s.get("https://www.nseindia.com/", timeout=20)
        rows = s.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=20).json()
    except Exception as e:
        log.warning("FII/DII fetch failed: %s", e)
        rows = []
    out = {}
    for r in rows:
        key = "fii" if "FII" in r["category"] else "dii"
        out[key] = float(r["netValue"])
        out["date"] = r["date"]
    if out.get("date"):
        hist = pd.read_csv(FII_PATH) if FII_PATH.exists() else pd.DataFrame(columns=["date", "fii", "dii"])
        if out["date"] not in hist.date.astype(str).values:
            hist = pd.concat([hist, pd.DataFrame([{k: out.get(k) for k in ("date", "fii", "dii")}])])
            hist.to_csv(FII_PATH, index=False)
        recent = pd.read_csv(FII_PATH).tail(5)
        out["fii_5d"] = round(float(recent.fii.sum()), 1)
        out["dii_5d"] = round(float(recent.dii.sum()), 1)
        out["days_tracked"] = len(recent)
    return out


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    backfill(float(sys.argv[1]) if len(sys.argv) > 1 else 0.35)
    p = nse_panel()
    print("rows", len(p), "days", p.index.get_level_values(1).nunique())
