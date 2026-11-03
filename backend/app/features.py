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
# NSE archive features: delivery % (all stocks) and F&O open interest / PCR (F&O stocks only, else NaN)
NSE_FEATURES = ["deliv_per", "deliv_rel20", "deliv_qty_surge", "fut_oi_chg5", "oi_buildup", "pcr", "pcr_chg5",
                "xr_deliv_rel20"]
ALL_FEATURES = BASE_FEATURES + NSE_FEATURES


def add_nse(f: pd.DataFrame, nse: pd.DataFrame | None) -> pd.DataFrame:
    """Merge per-day NSE data (indexed by date) and derive features. Missing data stays NaN."""
    cols = ["deliv_per", "deliv_rel20", "deliv_qty_surge", "fut_oi_chg5", "oi_buildup", "pcr", "pcr_chg5"]
    if nse is None or nse.empty:
        for c in cols:
            f[c] = np.nan
        return f
    n = nse.reindex(f.index).ffill(limit=2)
    f["deliv_per"] = n.deliv_per
    f["deliv_rel20"] = n.deliv_per / n.deliv_per.rolling(20, min_periods=10).mean()
    f["deliv_qty_surge"] = n.deliv_qty.rolling(5).mean() / n.deliv_qty.rolling(50, min_periods=20).mean()
    if "fut_oi" in n:
        oi = n.fut_oi.where(n.fut_oi > 0)
        f["fut_oi_chg5"] = oi.pct_change(5, fill_method=None)
        # + = long buildup (price up, OI up) / short covering; - = short buildup (price down, OI up)
        f["oi_buildup"] = f.fut_oi_chg5 * np.sign(f.close.pct_change(5))
        pcr = n.put_oi / n.call_oi.where(n.call_oi > 0)
        f["pcr"] = pcr
        f["pcr_chg5"] = pcr.diff(5)
    else:
        f["fut_oi_chg5"] = f["oi_buildup"] = f["pcr"] = f["pcr_chg5"] = np.nan
    return f


def add_cross_sectional(data: "pd.DataFrame") -> "pd.DataFrame":
    """Per-date ranks across the universe + sector momentum. Needs columns date, sector."""
    g = data.groupby("date")
    for c in RANK_SOURCES:
        data[f"xr_{c}"] = g[c].rank(pct=True)
    data["sector_ret_20"] = data.groupby(["date", "sector"]).ret_20.transform("mean")
    data["sector_rank"] = data.groupby("date").sector_ret_20.rank(pct=True)
    if "deliv_rel20" in data:
        data["xr_deliv_rel20"] = data.groupby("date").deliv_rel20.rank(pct=True)
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
    return f


def label(f: pd.DataFrame, horizon: int = HOLD_MAX_DAYS, side: str = "long") -> pd.DataFrame:
    """Triple barrier: 1 if target hit before stop within `horizon` days.
    side="short": target below entry, stop above. Records realised trade return (positive = profit)
    and days held. Stop wins on same-day ties."""
    c, h, l, atr = f.close.values, f.high.values, f.low.values, f.atr.values
    sgn = 1 if side == "long" else -1
    n = len(f)
    y = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    days = np.full(n, np.nan)
    for i in range(n - horizon):
        tgt, stp = c[i] + sgn * TARGET_ATR * atr[i], c[i] - sgn * STOP_ATR * atr[i]
        res, r, d = 0, sgn * (c[i + horizon] / c[i] - 1), horizon
        for j in range(i + 1, i + horizon + 1):
            stop_hit = l[j] <= stp if sgn == 1 else h[j] >= stp
            tgt_hit = h[j] >= tgt if sgn == 1 else l[j] <= tgt
            if stop_hit:
                res, r, d = 0, sgn * (stp / c[i] - 1), j - i
                break
            if tgt_hit:
                res, r, d = 1, sgn * (tgt / c[i] - 1), j - i
                break
        y[i], ret[i], days[i] = res, r, d
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
