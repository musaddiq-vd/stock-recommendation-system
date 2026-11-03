"""Intraday quotes from Yahoo (about 15 min delayed): today's high / low / last."""
import logging
import time

import pandas as pd
import yfinance as yf

from .prices import _clean, yf_ticker

log = logging.getLogger(__name__)
_cache: dict = {}


def live_quotes(symbols: list[str], max_age: int = 60) -> dict[str, dict]:
    now = time.time()
    need = [s for s in symbols if s not in _cache or now - _cache[s]["ts"] > max_age]
    if need:
        try:
            raw = yf.download([yf_ticker(s) for s in need], period="1d", interval="5m",
                              group_by="ticker", auto_adjust=True, progress=False, threads=True)
            for s in need:
                t = yf_ticker(s)
                multi = isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0)
                df = _clean((raw[t] if multi else raw).copy(), daily=False)
                if len(df):
                    _cache[s] = {"last": float(df.close.iloc[-1]), "high": float(df.high.max()),
                                 "low": float(df.low.min()), "time": str(df.index[-1]),
                                 "date": str(df.index[-1].date()), "ts": now}
        except Exception as e:
            log.warning("live quotes failed: %s", e)
    return {s: _cache[s] for s in symbols if s in _cache}
