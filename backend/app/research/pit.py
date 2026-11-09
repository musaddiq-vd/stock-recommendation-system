"""Point-in-time universe from NSE daily bhavcopies, including stocks that were later delisted.

Survivorship bias: backtesting only today's Nifty 500 ignores every company that fell out or was delisted,
which flatters results. The NSE bhavcopy for each day lists every stock that actually traded, so a universe
rebuilt from it day by day has no hindsight.

    python -m app.research.pit download      # ~5,500 daily files (resumable)
    python -m app.research.pit build         # adjusted per-symbol prices + monthly top-500 universe

Prices are back-adjusted for splits / bonuses detected from large overnight gaps (see build).
The universe each month is the 500 most traded stocks over the previous 6 months, a liquidity proxy for Nifty 500.
"""
import datetime as dt
import io
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..config import STORAGE
from ..data.nse import _get

log = logging.getLogger(__name__)
DIR = STORAGE / "research" / "pit"
DAYS_DIR = DIR / "days"
DAYS_DIR.mkdir(parents=True, exist_ok=True)
NEW_FORMAT_FROM = dt.date(2024, 7, 8)  # NSE retired the old cm bhavcopy around the UDiFF switch
SERIES = ("EQ", "BE", "BZ")  # trade-to-trade series keep stocks that were moved, not delisted
# price ratios produced by common splits / bonuses (new price / old price), e.g. 1:1 bonus = 1/2, 1:10 split = 1/10
CA_RATIOS = (3 / 4, 2 / 3, 3 / 5, 1 / 2, 2 / 5, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 8, 1 / 10, 1 / 15, 1 / 20, 2, 5, 10)


def _old(day: dt.date) -> pd.DataFrame | None:
    mon = day.strftime("%b").upper()
    raw = _get(f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{day:%Y}/{mon}/"
               f"cm{day:%d}{mon}{day:%Y}bhav.csv.zip")
    if not raw:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    df.columns = [c.strip() for c in df.columns]
    return pd.DataFrame({"symbol": df.SYMBOL.astype(str).str.strip(), "series": df.SERIES.astype(str).str.strip(),
                         "open": df.OPEN, "high": df.HIGH, "low": df.LOW, "close": df.CLOSE,
                         "prevclose": df.PREVCLOSE, "value": df.TOTTRDVAL})


def _new(day: dt.date) -> pd.DataFrame | None:
    raw = _get(f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv")
    if not raw:
        return None
    df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    return pd.DataFrame({"symbol": df.SYMBOL.astype(str).str.strip(), "series": df.SERIES.astype(str).str.strip(),
                         "open": num("OPEN_PRICE"), "high": num("HIGH_PRICE"), "low": num("LOW_PRICE"),
                         "close": num("CLOSE_PRICE"), "prevclose": num("PREV_CLOSE"),
                         "value": num("TURNOVER_LACS") * 1e5})


def fetch_day(day: dt.date) -> None:
    path = DAYS_DIR / f"{day:%Y%m%d}.parquet"
    if path.exists():
        return
    df = _new(day) if day >= NEW_FORMAT_FROM else _old(day)
    if df is None or df.empty:
        return
    df = df[df.series.isin(SERIES)].copy()
    # one row per symbol: prefer the EQ series when a stock trades in several
    df["rank"] = df.series.map({"EQ": 0, "BE": 1, "BZ": 2})
    df = df.sort_values("rank").drop_duplicates("symbol").drop(columns=["rank"])
    for c in ("open", "high", "low", "close", "prevclose", "value"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df.to_parquet(path)


def trading_days(start: str) -> list[dt.date]:
    """Weekdays from `start`; holidays simply return no file."""
    return [d.date() for d in pd.bdate_range(start, pd.Timestamp.now().normalize())]


def download(start: str = "2002-01-01", workers: int = 4) -> None:
    days = [d for d in trading_days(start) if not (DAYS_DIR / f"{d:%Y%m%d}.parquet").exists()]
    log.info("bhavcopy days to fetch: %d", len(days))
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(fetch_day, days), 1):
            if i % 250 == 0:
                log.info("fetched %d/%d", i, len(days))


def build(min_days: int = 260, top: int = 500) -> None:
    """Adjusted per-symbol OHLC panel and a monthly point-in-time universe."""
    frames = []
    for p in sorted(DAYS_DIR.glob("*.parquet")):
        d = pd.read_parquet(p)
        d["date"] = pd.Timestamp(p.stem)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw.close > 0) & (raw.high >= raw.low)].sort_values(["symbol", "date"])
    log.info("raw rows %d, symbols %d", len(raw), raw.symbol.nunique())

    # Back-adjust for splits / bonuses. Bhavcopy PREVCLOSE is NOT restated on ex-dates (verified against Yahoo),
    # so corporate actions are detected from the overnight gap: NSE price bands cap normal daily moves at 20%,
    # so a >22% gap that matches a standard split/bonus ratio is treated as a corporate action, not a crash.
    prev_close = raw.groupby("symbol").close.shift()
    gap = (raw.open / prev_close).where(prev_close.notna())
    f = pd.Series(1.0, index=raw.index)
    for ratio in CA_RATIOS:
        hit = (gap / ratio - 1).abs() < 0.06
        f = f.where(~hit | (f != 1.0), ratio)
    f = f.where((gap < 0.78) | (gap > 1.8), 1.0)  # only large gaps can be corporate actions
    raw["ca_factor"] = f.values
    log.info("corporate actions detected: %d", int((f != 1.0).sum()))
    # multiplier for each row = product of all later factors of the same symbol
    raw["adj"] = (f.groupby(raw.symbol).transform(lambda s: s[::-1].cumprod()[::-1].shift(-1).fillna(1.0))).values
    for c in ("open", "high", "low", "close"):
        raw[c] = (raw[c] * raw.adj).astype("float32")
    raw = raw.drop(columns=["adj", "prevclose", "ca_factor"])
    raw.to_parquet(DIR / "prices.parquet")

    # Monthly universe: top-N by median daily traded value over the previous ~126 sessions
    raw["month"] = raw.date.dt.to_period("M")
    months = sorted(raw.month.unique())
    hist = raw.groupby("symbol").date.transform("rank")
    raw["age"] = hist
    rows = []
    for m in months:
        start = m.to_timestamp() - pd.DateOffset(months=6)
        window = raw[(raw.date >= start) & (raw.date < m.to_timestamp())]
        if window.empty:
            continue
        liq = window[window.age >= min_days].groupby("symbol").value.median().nlargest(top)
        rows += [{"month": str(m), "symbol": s, "rank": i + 1} for i, s in enumerate(liq.index)]
    uni = pd.DataFrame(rows)
    uni.to_parquet(DIR / "universe.parquet")
    log.info("universe months %d, distinct symbols %d", uni.month.nunique(), uni.symbol.nunique())


def load_prices(symbols: set[str] | None = None) -> dict[str, pd.DataFrame]:
    p = pd.read_parquet(DIR / "prices.parquet")
    if symbols is not None:
        p = p[p.symbol.isin(symbols)]
    return {s: g.set_index("date")[["open", "high", "low", "close", "value"]].rename(columns={"value": "traded"})
            for s, g in p.groupby("symbol")}


def load_universe() -> pd.DataFrame:
    return pd.read_parquet(DIR / "universe.parquet")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"download": download, "build": build}[sys.argv[1] if len(sys.argv) > 1 else "download"]()
    np.seterr(all="ignore")
