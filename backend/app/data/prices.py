"""Daily OHLCV from yfinance with a parquet cache per symbol."""
import datetime as dt
import logging
import threading
import warnings
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from ..config import BENCHMARK, HISTORY_PERIOD, PRICE_DIR

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning)


def yf_ticker(symbol: str) -> str:
    return symbol if symbol.startswith("^") or symbol.endswith(".NS") else f"{symbol}.NS"


def _path(symbol: str):
    return PRICE_DIR / f"{symbol.replace('^', 'IDX_').replace('&', '_')}.parquet"


IST = ZoneInfo("Asia/Kolkata")


def ist_now() -> dt.datetime:
    """Current Indian time as a naive datetime, whatever the server's own timezone (Docker runs on UTC)."""
    return dt.datetime.now(IST).replace(tzinfo=None)


def expected_last_bar(now: dt.datetime | None = None) -> dt.date:
    """Latest session whose daily bar should exist: today after 16:00 IST, else the previous weekday."""
    now = now or ist_now()
    d = now.date() if now.hour >= 16 else now.date() - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def _fresh(path) -> bool:
    """Fresh if it already has the expected latest bar, or was refreshed in the last 20 minutes
    (covers NSE holidays, when no new bar will appear). The window is short so that when Yahoo publishes
    the day's bars late, the daily job's 30-minute retries really download again."""
    if not path.exists():
        return False
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    if dt.datetime.now() - mtime < dt.timedelta(minutes=20):
        return True
    try:
        last = pd.read_parquet(path, columns=["close"]).index[-1].date()
    except Exception:
        return False
    return last >= expected_last_bar()


_missing: dict[str, float] = {}  # symbols Yahoo has no data for (delisted / renamed), skip for a day


def _clean(df: pd.DataFrame, daily: bool = True) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:  # Ticker.history returns IST-aware timestamps; keep the local date
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    df.index = idx.normalize() if daily else idx
    df = df.dropna(subset=["close"])
    if daily:
        # During market hours Yahoo returns today's unfinished bar. Caching it would make the evening scan
        # read a mid-day snapshot as the close (the file then looks fresh), so keep finished sessions only.
        df = df[df.index <= pd.Timestamp(expected_last_bar())]
    return df


_refresh_lock = threading.Lock()


def load_many(symbols: list[str], force: bool = False) -> dict[str, pd.DataFrame]:
    """Cached daily bars; stale symbols are refreshed under a lock so two threads never write one file."""
    with _refresh_lock:
        return _load_many(symbols, force)


def _load_many(symbols: list[str], force: bool = False) -> dict[str, pd.DataFrame]:
    import time as _t

    out, stale = {}, []
    for s in symbols:
        p = _path(s)
        if not force and _fresh(p):
            out[s] = pd.read_parquet(p)
        elif not force and _t.time() - _missing.get(s, 0) < 86400:
            continue
        else:
            stale.append(s)

    for i in range(0, len(stale), 50):
        chunk = stale[i : i + 50]
        raw = yf.download(
            [yf_ticker(s) for s in chunk], period=HISTORY_PERIOD, interval="1d",
            group_by="ticker", auto_adjust=True, progress=False, threads=True,
        )
        for s in chunk:
            try:
                t = yf_ticker(s)
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0) else raw
                df = _clean(df.copy())
                if len(df) < 250:
                    if not _path(s).exists():
                        _missing[s] = _t.time()
                    continue
                df.to_parquet(_path(s))
                out[s] = df
            except Exception as e:  # delisted / missing
                log.warning("price load failed %s: %s", s, e)
                if _path(s).exists():
                    out[s] = pd.read_parquet(_path(s))

    # Yahoo batch downloads sometimes miss the latest bar for part of the batch: retry those in small groups
    exp = expected_last_bar()
    lagging = [s for s in stale if s in out and out[s].index[-1].date() < exp]

    def _one(s):
        try:
            return s, _clean(yf.Ticker(yf_ticker(s)).history(period="1mo", auto_adjust=True))
        except Exception:
            return s, None

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as ex:
        for s, recent in ex.map(_one, lagging):
            if recent is not None and len(recent) and recent.index[-1] > out[s].index[-1]:
                merged = pd.concat([out[s], recent[recent.index > out[s].index[-1]]])
                merged.to_parquet(_path(s))
                out[s] = merged
    # Still missing (Yahoo has NaN for the day): fill from the official NSE bhavcopy
    still = [s for s in lagging if out[s].index[-1].date() < exp and not s.startswith("^")]
    if still:
        from .nse import bhav_ohlc

        last = min(out[s].index[-1].date() for s in still)
        days = [d.date() for d in pd.bdate_range(last + dt.timedelta(days=1), exp)]
        bhavs = {d: bhav_ohlc(d) for d in days}
        for s in still:
            add = [pd.DataFrame([b.loc[s]], index=[pd.Timestamp(d)]) for d, b in bhavs.items()
                   if b is not None and s in b.index and pd.Timestamp(d) > out[s].index[-1]]
            if add:
                merged = pd.concat([out[s], *add])
                merged.index.name = "Date"
                merged.to_parquet(_path(s))
                out[s] = merged
        log.info("filled %d symbols from NSE bhavcopy", len(still))
    return out


def load(symbol: str) -> pd.DataFrame | None:
    return load_many([symbol]).get(symbol)


def benchmark() -> pd.DataFrame:
    """Nifty 50 daily; missing latest sessions are filled from NSE's official index closing file."""
    df = load(BENCHMARK)
    exp = expected_last_bar()
    if df is None or df.index[-1].date() >= exp:
        return df
    from .nse import _get

    add = []
    for d in pd.bdate_range(df.index[-1] + dt.timedelta(days=1), exp):
        raw = _get(f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{d:%d%m%Y}.csv")
        if not raw:
            continue
        import io

        t = pd.read_csv(io.BytesIO(raw))
        row = t[t["Index Name"].str.strip() == "Nifty 50"]
        if len(row):
            r = row.iloc[0]
            add.append(pd.DataFrame({"open": [float(r["Open Index Value"])], "high": [float(r["High Index Value"])],
                                     "low": [float(r["Low Index Value"])], "close": [float(r["Closing Index Value"])],
                                     "volume": [float(r["Volume"])]}, index=[pd.Timestamp(d)]))
    if add:
        df = pd.concat([df, *add])
        df.index.name = "Date"
        df.to_parquet(_path(BENCHMARK))
    return df
