"""Technical indicators and swing-trade labels (triple barrier)."""
import numpy as np
import pandas as pd

from .config import HOLD_MAX_DAYS, STOP_ATR, TARGET_ATR

FEATURES = [
    "rsi14", "macd_hist_n", "adx14", "atr_pct", "dist_ema20", "dist_ema50", "dist_ema200",
    "ema20_slope", "vol_surge", "dist_52w_high", "dist_52w_low", "ret_5", "ret_20", "ret_60",
    "rs_20", "rs_60", "bb_width", "bb_pos", "volatility_20", "up_days_10",
    "mkt_dist_ema200", "mkt_ret_20",
]
RANK_SOURCES = ["ret_20", "ret_60", "rs_60", "dist_52w_high", "vol_surge", "atr_pct", "adx14"]
XS_FEATURES = [f"xr_{c}" for c in RANK_SOURCES] + ["sector_ret_20", "sector_rank"]
MARKET_FEATURES = ["mkt_dist_ema200", "mkt_ret_20"]
BASE_FEATURES = [c for c in FEATURES if c not in MARKET_FEATURES] + XS_FEATURES
ALL_FEATURES = BASE_FEATURES


def add_cross_sectional(data: "pd.DataFrame") -> "pd.DataFrame":
    """Per-date ranks across the universe + sector momentum. Needs columns date, sector."""
    g = data.groupby("date")
    for c in RANK_SOURCES:
        data[f"xr_{c}"] = g[c].rank(pct=True)
    data["sector_ret_20"] = data.groupby(["date", "sector"]).ret_20.transform("mean")
    data["sector_rank"] = data.groupby("date").sector_ret_20.rank(pct=True)
    return data


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _adx(df, n=14):
    up = df.high.diff()
    dn = -df.low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = _atr(df, n)
    pdi = 100 * pd.Series(plus, df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def compute(df: pd.DataFrame, bench: pd.DataFrame | None = None) -> pd.DataFrame:
    """Only uses data up to each row's date (no look-ahead)."""
    f = df.copy()
    c = f.close
    f["rsi14"] = _rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist_n"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    f["adx14"] = _adx(f)
    f["atr"] = _atr(f)
    f["atr_pct"] = f.atr / c
    for n in (20, 50, 200):
        f[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
        f[f"dist_ema{n}"] = c / f[f"ema{n}"] - 1
    f["ema20_slope"] = f.ema20.pct_change(5)
    f["vol_surge"] = f.volume.rolling(5).mean() / f.volume.rolling(50).mean()
    f["dist_52w_high"] = c / f.high.rolling(252, min_periods=120).max() - 1
    f["dist_52w_low"] = c / f.low.rolling(252, min_periods=120).min() - 1
    for n in (5, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)
    if bench is not None:
        b = bench.close.reindex(f.index).ffill()
        f["rs_20"] = f.ret_20 - b.pct_change(20)
        f["rs_60"] = f.ret_60 - b.pct_change(60)
        f["mkt_dist_ema200"] = b / b.ewm(span=200, adjust=False).mean() - 1
        f["mkt_ret_20"] = b.pct_change(20)
    else:
        f["rs_20"] = f["rs_60"] = f["mkt_dist_ema200"] = f["mkt_ret_20"] = 0.0
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_width"] = 4 * sd20 / sma20
    f["bb_pos"] = (c - (sma20 - 2 * sd20)) / (4 * sd20)
    f["volatility_20"] = c.pct_change().rolling(20).std()
    f["up_days_10"] = (c.diff() > 0).rolling(10).sum()
    f["traded_value_20"] = (c * f.volume).rolling(20).mean()
    f["low_20"] = f.low.rolling(20).min()
    return f


def barrier_outcomes(c, h, l, atr, target_atr: float, stop_atr: float, horizon: int, side: str = "long"):
    """Vectorised triple barrier over all start days.

    Returns (y, ret, days): y=1 if the target is hit before the stop within `horizon` bars, the realised
    return (positive = profit) and bars held. Stop wins on same-bar ties; unresolved trades exit at the close
    of the last bar. The last `horizon` rows are NaN (outcome unknown).
    """
    c, h, l, atr = (np.asarray(x, dtype=float) for x in (c, h, l, atr))
    n = len(c)
    sgn = 1.0 if side == "long" else -1.0
    tgt = c + sgn * target_atr * atr
    stp = c - sgn * stop_atr * atr
    y = np.zeros(n)
    ret = np.full(n, np.nan)
    days = np.full(n, float(horizon))
    open_ = np.ones(n, dtype=bool)
    for j in range(1, horizon + 1):
        hj = np.full(n, np.nan)
        lj = np.full(n, np.nan)
        hj[: n - j], lj[: n - j] = h[j:], l[j:]
        stop_hit = open_ & ((lj <= stp) if sgn > 0 else (hj >= stp))
        tgt_hit = open_ & ~stop_hit & ((hj >= tgt) if sgn > 0 else (lj <= tgt))
        ret[stop_hit] = sgn * (stp[stop_hit] / c[stop_hit] - 1)
        days[stop_hit] = j
        ret[tgt_hit] = sgn * (tgt[tgt_hit] / c[tgt_hit] - 1)
        y[tgt_hit] = 1
        days[tgt_hit] = j
        open_ &= ~(stop_hit | tgt_hit)
    still = open_.copy()
    still[n - horizon:] = False
    ret[still] = sgn * (c[np.where(still)[0] + horizon] / c[still] - 1)
    y[n - horizon:] = np.nan
    ret[n - horizon:] = np.nan
    days[n - horizon:] = np.nan
    return y, ret, days


def label(f: pd.DataFrame, horizon: int = HOLD_MAX_DAYS, side: str = "long",
          target_atr: float = TARGET_ATR, stop_atr: float = STOP_ATR) -> pd.DataFrame:
    """Adds y / trade_ret / days_held columns using the triple barrier (see barrier_outcomes)."""
    y, ret, days = barrier_outcomes(f.close.values, f.high.values, f.low.values, f.atr.values,
                                    target_atr, stop_atr, horizon, side)
    f = f.copy()
    f["y"], f["trade_ret"], f["days_held"] = y, ret, days
    return f


def trade_levels(row: pd.Series, side: str = "long") -> dict:
    entry = float(row.close)
    sgn = 1 if side == "long" else -1
    return {
        "side": side,
        "entry": round(entry, 2),
        "target": round(entry + sgn * TARGET_ATR * row.atr, 2),
        "stop_loss": round(entry - sgn * STOP_ATR * row.atr, 2),
        "target_pct": round(TARGET_ATR * row.atr / entry * 100, 2),
        "stop_pct": round(STOP_ATR * row.atr / entry * 100, 2),
    }
