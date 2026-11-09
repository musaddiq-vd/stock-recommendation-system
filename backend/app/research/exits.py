"""Does managing the trade or capping sectors beat the plain 3.5 / 2.5 ATR bracket? (survivorship-free test)

    python -m app.research.exits pit

Every variant uses the live entry rule (ML top 2% + breadth > 50%), next-day-open entries, gap fills and 0.3% cost.
"""
import json
import logging
import sys

import numpy as np
import pandas as pd

from ..features import compute
from .experiment import portfolio
from .v2 import COST_REAL, HORIZON, OUT, SPLIT, STOP, TARGET, _benchmark, _sources


def managed_outcomes(o, h, l, c, atr, target_atr, stop_atr, horizon, be_at=None, trail=None):
    """Like features.next_open_outcomes, plus stop management decided at each close for the next bar:
    be_at: once the high reaches entry + be_at x ATR, the stop moves up to the entry (breakeven).
    trail: the stop follows the highest high minus trail x ATR (never moves down)."""
    o, h, l, c, atr = (np.asarray(x, dtype=float) for x in (o, h, l, c, atr))
    n = len(c)
    entry = np.full(n, np.nan)
    entry[: n - 1] = o[1:]
    tgt = entry + target_atr * atr
    stp = entry - stop_atr * atr
    hi = entry.copy()
    ret, days = np.full(n, np.nan), np.full(n, float(horizon))
    open_ = ~np.isnan(entry)
    for j in range(1, horizon + 1):
        oj, hj, lj = (np.full(n, np.nan) for _ in range(3))
        oj[: n - j], hj[: n - j], lj[: n - j] = o[j:], h[j:], l[j:]
        gap_stop = open_ & (j > 1) & (oj <= stp)
        stop_hit = open_ & ~gap_stop & (lj <= stp)
        gap_tgt = open_ & ~gap_stop & ~stop_hit & (j > 1) & (oj >= tgt)
        tgt_hit = open_ & ~gap_stop & ~stop_hit & ~gap_tgt & (hj >= tgt)
        for mask, price in ((gap_stop, oj), (stop_hit, stp), (gap_tgt, oj), (tgt_hit, tgt)):
            ret[mask] = price[mask] / entry[mask] - 1
            days[mask] = j
        open_ &= ~(gap_stop | stop_hit | gap_tgt | tgt_hit)
        hi = np.where(open_, np.fmax(hi, hj), hi)
        if be_at is not None:
            stp = np.where(open_ & (hi >= entry + be_at * atr), np.fmax(stp, entry), stp)
        if trail is not None:
            stp = np.where(open_, np.fmax(stp, hi - trail * atr), stp)
    still = open_.copy()
    still[n - horizon:] = False
    ret[still] = c[np.where(still)[0] + horizon] / entry[still] - 1
    ret[n - horizon:] = np.nan
    days[n - horizon:] = np.nan
    return ret, days


EXITS = {"be1": {"be_at": 1.0}, "be1.5": {"be_at": 1.5}, "trail2": {"trail": 2.0}, "trail3": {"trail": 3.0}}


def run(source: str = "pit") -> dict:
    data = pd.read_parquet(OUT / f"dataset_{source}.parquet")
    walk = pd.read_parquet(OUT / f"walk_{source}.parquet")
    df = walk.merge(data, on=["date", "symbol"], how="inner")
    df = df[(df.traded_value_20 > 1e7)].dropna(subset=["r_open"])
    calendar = sorted(data.date.unique())
    sig = df[(df.pct_gbm >= 0.98) & (df.breadth50 > 0.5)].assign(prob=lambda x: x.p_gbm)

    # path-dependent exits need full OHLC for the signalled stocks
    prices, _ = _sources(source)
    bench = _benchmark(prices)
    extra = []
    for s in sig.symbol.unique():
        p = prices.get(s)
        if p is None:
            continue
        f = compute(p[["open", "high", "low", "close", "volume"]], bench)
        out = pd.DataFrame(index=f.index)
        for name, kw in EXITS.items():
            r, d = managed_outcomes(f.open, f.high, f.low, f.close, f.atr, TARGET, STOP, HORIZON, **kw)
            out[f"r_{name}"], out[f"d_{name}"] = r, d
        out["symbol"] = s
        extra.append(out.reset_index().rename(columns={"index": "date", "Date": "date"}))
    ex = pd.concat(extra)
    ex["date"] = pd.to_datetime(ex["date"])
    sig = sig.merge(ex, on=["date", "symbol"], how="left")

    rows = []

    def add(name, rcol, dcol, **kw):
        s = sig.dropna(subset=[rcol])
        res = portfolio(s, rcol, dcol, cost=COST_REAL, calendar=calendar, **kw)
        held = portfolio(s[s.date >= SPLIT], rcol, dcol, cost=COST_REAL, calendar=calendar, **kw)
        rows.append({"variant": name, **{k: v for k, v in res.items() if k not in ("yearly", "equity")},
                     "holdout_win": held.get("win_rate"), "holdout_pf": held.get("profit_factor"),
                     "holdout_dd": held.get("max_dd")})

    add("Live rule, plain bracket (full calendar)", "r_open", "d_open")
    for cap in (2, 3):
        add(f"Live rule + max {cap} per sector", "r_open", "d_open", max_per_sector=cap)
    for name in EXITS:
        add(f"Live rule + exit {name}", f"r_{name}", f"d_{name}")
    out = {"source": source, "rows": rows}
    (OUT / f"exits_{source}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    r = run(sys.argv[1] if len(sys.argv) > 1 else "pit")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(r["rows"]).drop(columns=["losing_years"]).to_string(index=False))
