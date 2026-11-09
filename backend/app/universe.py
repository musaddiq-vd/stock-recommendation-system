"""Nifty 50 / 100 / 500 constituents with sector tags, downloaded from NSE and cached."""
import io
import time

import pandas as pd
import requests

from .config import UNIVERSE_DIR

INDEX_URLS = {
    "NIFTY50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "text/csv,*/*",
}
MAX_AGE = 7 * 24 * 3600


def _fetch(index: str) -> pd.DataFrame:
    r = requests.get(INDEX_URLS[index], headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns={"Company Name": "name", "Industry": "sector", "Symbol": "symbol"})[
        ["symbol", "name", "sector"]
    ]


def get_index(index: str) -> pd.DataFrame:
    index = index.upper()
    if index not in INDEX_URLS:
        raise ValueError(f"Unknown index {index}")
    path = UNIVERSE_DIR / f"{index}.csv"
    if not path.exists() or time.time() - path.stat().st_mtime > MAX_AGE:
        try:
            _fetch(index).to_csv(path, index=False)
        except Exception:
            if not path.exists():
                raise
    return pd.read_csv(path)


def get_universe() -> pd.DataFrame:
    """All Nifty 500 stocks with tags: which indices they belong to + sector."""
    n500 = get_index("NIFTY500")
    n100 = set(get_index("NIFTY100").symbol)
    n50 = set(get_index("NIFTY50").symbol)

    def tags(s):
        t = ["NIFTY500"]
        if s in n100:
            t.append("NIFTY100")
        if s in n50:
            t.append("NIFTY50")
        return t

    n500["indices"] = n500.symbol.map(tags)
    return n500


def symbols_for(index: str, sector: str | None = None) -> list[str]:
    df = get_index(index)
    if sector:
        df = df[df.sector.str.lower() == sector.lower()]
    return df.symbol.tolist()
