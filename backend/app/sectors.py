"""Sector heatmap: returns, breadth and money flow per NSE sector."""
import time

import numpy as np
import pandas as pd

from .data import nse
from .data.prices import benchmark, load_many
from .universe import get_index, get_universe

_cache: dict = {}


def _stock_row(df: pd.DataFrame, bench: pd.Series, dl: pd.DataFrame | None) -> dict | None:
    if len(df) < 70:
        return None
    c, v = df.close, df.volume
    tv = c * v
    up = np.sign(c.diff())
    last = df.index[-1]
    b = bench.reindex(df.index).ffill()
    # Money flow: turnover on up days minus down days over 20 sessions, as a share of total turnover (-1..+1)
    flow20 = float((up * tv).tail(20).sum() / tv.tail(20).sum())
    flow5 = float((up * tv).tail(5).sum() / tv.tail(5).sum())
    row = {
        "ret_1d": float(c.iloc[-1] / c.iloc[-2] - 1), "ret_5d": float(c.iloc[-1] / c.iloc[-6] - 1),
        "ret_20d": float(c.iloc[-1] / c.iloc[-21] - 1), "ret_60d": float(c.iloc[-1] / c.iloc[-61] - 1),
        "rs_20d": float((c.iloc[-1] / c.iloc[-21]) - (b.iloc[-1] / b.iloc[-21])),
        "above_ema50": float(c.iloc[-1] > c.ewm(span=50, adjust=False).mean().iloc[-1]),
        "turnover_surge": float(tv.tail(5).mean() / tv.tail(50).mean()),
        "flow_20d": flow20, "flow_5d": flow5, "turnover_cr": float(tv.tail(20).mean() / 1e7), "date": last,
    }
    if dl is not None and len(dl.dropna(subset=["deliv_qty"])) > 30:
        q = dl.deliv_qty.dropna()
        row["deliv_surge"] = float(q.tail(5).mean() / q.tail(50).mean())
    return row


def rotation(closes: pd.DataFrame, sectors: pd.Series, weeks: int = 6) -> dict:
    """Relative-rotation numbers per sector, like a relative rotation graph (RRG).

    A sector's daily return is the median of its stocks; its relative-strength line is that index divided by
    the whole universe measured the same way (equal weight), so a large-cap rally in Nifty does not push every
    sector into "lagging". rs_ratio > 100: beat the market over ~3 months; rs_momentum > 100: still gaining.
    Returns one point per week (oldest first) so the chart can draw each sector's recent path."""
    rets = closes.pct_change().clip(-0.2, 0.2)
    b = rets.median(axis=1).fillna(0)
    out = {}
    for sector, cols in sectors.groupby(sectors).groups.items():
        cols = [c for c in cols if c in rets]
        if len(cols) < 3:
            continue
        idx = (1 + rets[cols].median(axis=1).fillna(0)).cumprod()
        rs = idx / (1 + b).cumprod()
        ratio = (100 * rs / rs.rolling(60).mean()).ewm(span=5).mean()
        mom = (100 * ratio / ratio.shift(10)).ewm(span=5).mean()
        pts = pd.DataFrame({"x": ratio, "y": mom}).dropna().iloc[::-1].iloc[::5].head(weeks).iloc[::-1]
        if len(pts):
            out[sector] = [{"x": round(float(r.x), 2), "y": round(float(r.y), 2)} for r in pts.itertuples()]
    return out


def quadrant(x: float, y: float) -> str:
    if x >= 100:
        return "leading" if y >= 100 else "weakening"
    return "improving" if y >= 100 else "lagging"


def sector_heatmap(index: str = "NIFTY500") -> dict:
    key = index.upper()
    if key in _cache and time.time() - _cache[key]["ts"] < 900:
        return _cache[key]["data"]
    uni = get_universe()
    members = set(get_index(key).symbol)
    uni = uni[uni.symbol.isin(members)]
    prices, bench = load_many(uni.symbol.tolist()), benchmark().close
    rows = []
    for s, sector in zip(uni.symbol, uni.sector, strict=False):
        df = prices.get(s)
        if df is None:
            continue
        r = _stock_row(df, bench, nse.for_symbol(s))
        if r:
            rows.append({"symbol": s, "sector": sector, **r})
    d = pd.DataFrame(rows)
    d = d[d.date >= d.date.max() - pd.Timedelta(days=5)]
    closes = pd.DataFrame({s: prices[s].close.tail(200) for s in d.symbol}).ffill(limit=3)
    rot = rotation(closes, d.set_index("symbol").sector)
    out = []
    for sector, g in d.groupby("sector"):
        w = g.turnover_cr / g.turnover_cr.sum()
        leaders = g.sort_values("ret_20d", ascending=False)
        out.append({
            "sector": sector, "stocks": len(g),
            "ret_1d": round(float(g.ret_1d.median()) * 100, 2), "ret_5d": round(float(g.ret_5d.median()) * 100, 2),
            "ret_20d": round(float(g.ret_20d.median()) * 100, 2), "ret_60d": round(float(g.ret_60d.median()) * 100, 2),
            "rs_20d": round(float(g.rs_20d.median()) * 100, 2),
            "breadth": round(float(g.above_ema50.mean()) * 100, 1),
            "flow_20d": round(float((g.flow_20d * w).sum()) * 100, 1),
            "flow_5d": round(float((g.flow_5d * w).sum()) * 100, 1),
            "turnover_surge": round(float(g.turnover_surge.median()), 2),
            "deliv_surge": round(float(g.deliv_surge.median()), 2) if "deliv_surge" in g and g.deliv_surge.notna().any() else None,
            "turnover_cr": round(float(g.turnover_cr.sum()), 0),
            "leaders": leaders.symbol.head(3).tolist(), "laggards": leaders.symbol.tail(3).tolist()[::-1],
            "members": [{"symbol": r.symbol, "ret_1d": round(r.ret_1d * 100, 2), "ret_5d": round(r.ret_5d * 100, 2),
                         "ret_20d": round(r.ret_20d * 100, 2), "ret_60d": round(r.ret_60d * 100, 2),
                         "flow_20d": round(r.flow_20d * 100, 1),
                         "turnover_cr": round(r.turnover_cr, 1)}
                        for r in g.sort_values("turnover_cr", ascending=False).head(30).itertuples()],
        })
        trail = rot.get(sector)
        if trail:
            out[-1]["rotation"] = trail
            out[-1]["quadrant"] = quadrant(trail[-1]["x"], trail[-1]["y"])
            out[-1]["prev_quadrant"] = quadrant(trail[0]["x"], trail[0]["y"])
    # Tiny sectors (1-2 stocks) are noisy: list them after the real ones
    out.sort(key=lambda x: (x["stocks"] < 3, -x["flow_20d"]))
    data = {"index": key, "as_of": str(d.date.max().date()), "sectors": out}
    _cache[key] = {"ts": time.time(), "data": data}
    return data
