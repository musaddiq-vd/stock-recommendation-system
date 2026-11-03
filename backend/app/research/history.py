"""Long-history dataset (1996 onwards) for research backtests.

Kept separate from the live 6-year cache. Before Nifty 50 exists on Yahoo (Sept 2007), an
equal-weight index of the universe stands in as the benchmark.
"""
import logging
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from ..config import STORAGE
from ..data.prices import _clean, yf_ticker
from ..universe import get_universe

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
DIR = STORAGE / "research" / "prices_max"
DIR.mkdir(parents=True, exist_ok=True)


def _path(sym: str):
    return DIR / f"{sym.replace('^', 'IDX_').replace('&', '_')}.parquet"


def download(symbols: list[str] | None = None, force: bool = False) -> None:
    symbols = symbols or get_universe().symbol.tolist() + ["^NSEI"]
    todo = [s for s in symbols if force or not _path(s).exists()]
    for i in range(0, len(todo), 40):
        chunk = todo[i : i + 40]
        raw = yf.download([yf_ticker(s) for s in chunk], period="max", interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
        for s in chunk:
            t = yf_ticker(s)
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0) else raw
                df = _clean(df.copy())
                # drop zero-volume placeholder rows and obviously broken prints
                df = df[(df.close > 0) & (df.high >= df.low)]
                if len(df) >= 300:
                    df.to_parquet(_path(s))
            except Exception as e:
                log.warning("max history failed %s: %s", s, e)
        log.info("downloaded %d/%d", min(i + 40, len(todo)), len(todo))


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    for p in DIR.glob("*.parquet"):
        if p.stem.startswith("IDX_"):
            continue
        out[p.stem] = pd.read_parquet(p)
    return out


def benchmark(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Nifty 50 from 2007; before that an equal-weight index of available stocks, chained to Nifty."""
    rets = pd.DataFrame({s: df.close.pct_change() for s, df in prices.items()})
    rets = rets.clip(-0.25, 0.25)  # guard against bad prints
    ew = (1 + rets.median(axis=1).fillna(0)).cumprod()
    nifty_p = _path("^NSEI")
    if nifty_p.exists():
        n = pd.read_parquet(nifty_p).close
        start = n.index[0]
        scale = n.iloc[0] / ew.asof(start)
        close = pd.concat([ew[ew.index < start] * scale, n])
    else:
        close = ew
    close = close[~close.index.duplicated()].sort_index()
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": np.nan})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download()
    p = load_all()
    starts = pd.Series({s: df.index[0].year for s, df in p.items()})
    print("stocks", len(p), "by start decade:", starts.floordiv(10).mul(10).value_counts().sort_index().to_dict())
