"""NSE archive data: delivery % (equity bhavcopy), F&O open interest / PCR, FII-DII flows.

Each trading day is cached as a small parquet so backfill can resume and daily updates are cheap.
"""
import datetime as dt
import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from ..config import STORAGE

log = logging.getLogger(__name__)
NSE_DIR = STORAGE / "nse"
DELIV_DIR, FO_DIR = NSE_DIR / "deliv", NSE_DIR / "fo"
for d in (DELIV_DIR, FO_DIR):
    d.mkdir(parents=True, exist_ok=True)
FII_PATH = NSE_DIR / "fii_dii.csv"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
UDIFF_START = dt.date(2024, 7, 8)  # NSE switched F&O bhavcopy to the UDiFF format


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


def fetch_fo(day: dt.date) -> pd.DataFrame | None:
    """Per stock: total futures OI, call OI and put OI across expiries."""
    if day >= UDIFF_START:
        url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    else:
        mon = day.strftime("%b").upper()
        url = (f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{day:%Y}/{mon}/"
               f"fo{day:%d}{mon}{day:%Y}bhav.csv.zip")
    raw = _get(url)
    if not raw:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    if "TckrSymb" in df.columns:  # UDiFF
        df = pd.DataFrame({"inst": df.FinInstrmTp, "symbol": df.TckrSymb, "opt": df.OptnTp, "oi": df.OpnIntrst})
        fut, opt = "STF", "STO"
    else:
        df = pd.DataFrame({"inst": df.INSTRUMENT, "symbol": df.SYMBOL, "opt": df.OPTION_TYP, "oi": df.OPEN_INT})
        fut, opt = "FUTSTK", "OPTSTK"
    df["symbol"] = df.symbol.astype(str).str.strip()
    df["oi"] = pd.to_numeric(df.oi, errors="coerce").fillna(0)
    f = df[df.inst == fut].groupby("symbol").oi.sum().rename("fut_oi")
    o = df[df.inst == opt]
    c = o[o.opt == "CE"].groupby("symbol").oi.sum().rename("call_oi")
    p = o[o.opt == "PE"].groupby("symbol").oi.sum().rename("put_oi")
    out = pd.concat([f, c, p], axis=1).reset_index().rename(columns={"index": "symbol"})
    out["date"] = pd.Timestamp(day)
    return out


def _cache_day(day: dt.date) -> None:
    for folder, fn in ((DELIV_DIR, fetch_delivery), (FO_DIR, fetch_fo)):
        path = folder / f"{day:%Y%m%d}.parquet"
        if path.exists():
            continue
        df = fn(day)
        if df is not None and len(df):
            df.to_parquet(path)


def trading_days(years: float) -> list[dt.date]:
    from .prices import load

    idx = load("RELIANCE").index
    start = pd.Timestamp.now() - pd.DateOffset(days=int(365 * years))
    return [d.date() for d in idx if d >= start]


def backfill(years: float = 4, workers: int = 3) -> None:
    days = [d for d in trading_days(years)
            if not (DELIV_DIR / f"{d:%Y%m%d}.parquet").exists() or not (FO_DIR / f"{d:%Y%m%d}.parquet").exists()]
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
    """All cached days merged: index (symbol, date) → deliv_per, deliv_qty, fut_oi, call_oi, put_oi."""
    if "panel" not in _cache:
        dl = [pd.read_parquet(p) for p in sorted(DELIV_DIR.glob("*.parquet"))]
        fo = [pd.read_parquet(p) for p in sorted(FO_DIR.glob("*.parquet"))]
        if not dl:
            _cache["panel"] = pd.DataFrame()
        else:
            d = pd.concat(dl)
            if fo:
                d = d.merge(pd.concat(fo), on=["symbol", "date"], how="left")
            _cache["panel"] = d.set_index(["symbol", "date"]).sort_index()
    return _cache["panel"]


def for_symbol(symbol: str) -> pd.DataFrame | None:
    if "by_sym" not in _cache:
        p = nse_panel()
        _cache["by_sym"] = {} if p.empty else {
            s: g.droplevel("symbol") for s, g in p.groupby(level="symbol", sort=False)}
    return _cache["by_sym"].get(symbol)


def fii_dii() -> dict:
    """Today's FII/DII cash market net flows (Rs crore); also appended to a local history file."""
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
    backfill(float(sys.argv[1]) if len(sys.argv) > 1 else 4)
    p = nse_panel()
    print("rows", len(p), "days", p.index.get_level_values(1).nunique())
