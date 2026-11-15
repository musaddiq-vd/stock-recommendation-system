"""Round 4 of the improvement list, judged on the honest 22-year test before anything changes in the app.

Same setup as edge_lab: point-in-time universe (delisted stocks included), yearly walk-forward, next-open
entries with gap fills, 0.3% cost, 10-slot portfolio, live market gate (incl. the Nifty trend). Pick on
pre-2016, confirm on 2016+.

    python -m app.research.pattern_lab extra    # candlestick patterns and support / resistance features
    python -m app.research.pattern_lab run      # do they improve the model?
    python -m app.research.pattern_lab models   # LightGBM vs XGBoost vs CatBoost vs their average
    python -m app.research.pattern_lab rules    # sector caps, entry near support, room to resistance, candle confirmation

Result (Nov 2026): candlestick and support / resistance features, entry near support, room to resistance, candle
confirmation, XGBoost, CatBoost and their ensemble did not beat the live rule. A cap of 2 open trades per sector
kept the profit factor and lowered the 2016+ worst drawdown (-22% to -18%), so the app flags a 3rd Buy call in a sector.
"""
import json
import logging
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..features import BASE_FEATURES
from .experiment import portfolio
from .feature_lab import load
from .v2 import COST_REAL, HORIZON, OUT, PARAMS, SPLIT, live_gate

log = logging.getLogger(__name__)
CANDLES = ["c_bull_engulf", "c_hammer", "c_morning_star", "c_bull_marubozu", "c_bear_engulf", "c_shooting_star",
           "c_doji", "c_inside", "c_body_n", "c_upper_wick_n", "c_lower_wick_n", "c_bull_score_5"]
LEVELS = ["sr_support_dist", "sr_resist_dist", "sr_support_touches", "sr_resist_touches", "sr_room_atr", "sr_pivot_pos"]


# ---------------------------------------------------------------- features (only data up to each day)
def candle_features(df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    """Classic single and three-bar patterns plus the shape of the day's candle, scaled by ATR."""
    o, h, l, c = df.open, df.high, df.low, df.close
    body, rng = (c - o), (h - l).replace(0, np.nan)
    up_wick, lo_wick = h - np.maximum(o, c), np.minimum(o, c) - l
    po, pc = o.shift(), c.shift()
    trend_down = c.shift() < c.shift(10)  # patterns only mean something after a move
    trend_up = c.shift() > c.shift(10)
    out = pd.DataFrame(index=df.index)
    out["c_bull_engulf"] = ((pc < po) & (c > o) & (c >= po) & (o <= pc) & trend_down).astype(float)
    out["c_bear_engulf"] = ((pc > po) & (c < o) & (c <= po) & (o >= pc) & trend_up).astype(float)
    out["c_hammer"] = ((lo_wick >= 2 * body.abs()) & (up_wick <= 0.3 * body.abs().clip(lower=1e-9) + 0.1 * rng) & trend_down).astype(float)
    out["c_shooting_star"] = ((up_wick >= 2 * body.abs()) & (lo_wick <= 0.1 * rng) & trend_up).astype(float)
    b2, mid1 = (c.shift(2) - o.shift(2)), (o.shift(2) + c.shift(2)) / 2
    small1 = (c.shift() - o.shift()).abs() <= 0.3 * (h.shift() - l.shift())
    out["c_morning_star"] = ((b2 < 0) & small1 & (c > o) & (c > mid1) & (c.shift(3) > c.shift(2))).astype(float)
    out["c_bull_marubozu"] = ((body > 0.9 * rng) & (body > atr)).astype(float)
    out["c_doji"] = (body.abs() <= 0.1 * rng).astype(float)
    out["c_inside"] = ((h <= h.shift()) & (l >= l.shift())).astype(float)
    out["c_body_n"] = body / atr
    out["c_upper_wick_n"] = up_wick / atr
    out["c_lower_wick_n"] = lo_wick / atr
    bull = out.c_bull_engulf + out.c_hammer + out.c_morning_star + out.c_bull_marubozu
    bear = out.c_bear_engulf + out.c_shooting_star
    out["c_bull_score_5"] = (bull - bear).rolling(5).sum()
    return out


def level_features(df: pd.DataFrame, atr: pd.Series, look: int = 120, k: int = 5) -> pd.DataFrame:
    """Support = highest confirmed swing low below the close, resistance = lowest confirmed swing high above it,
    from the last `look` sessions. A swing point is only known `k` bars after it happened (no look-ahead)."""
    h, l, c = df.high.values, df.low.values, df.close.values
    n = len(c)
    win = 2 * k + 1
    hi_max = pd.Series(h).rolling(win, center=True).max().values
    lo_min = pd.Series(l).rolling(win, center=True).min().values
    is_sh, is_sl = (h == hi_max), (l == lo_min)
    a = atr.values
    sup_d, res_d, sup_t, res_t, room = (np.full(n, np.nan) for _ in range(5))
    for i in range(look + k, n):
        lo_i, hi_i = i - look, i - k  # swing points confirmed by day i
        sh = h[lo_i:hi_i][is_sh[lo_i:hi_i]]
        sl = l[lo_i:hi_i][is_sl[lo_i:hi_i]]
        below, above = sl[sl < c[i]], sh[sh > c[i]]
        if len(below):
            s = below.max()
            sup_d[i] = (c[i] - s) / a[i]
            sup_t[i] = np.sum(np.abs(sl - s) <= 0.5 * a[i])
        if len(above):
            r = above.min()
            res_d[i] = (r - c[i]) / a[i]
            res_t[i] = np.sum(np.abs(sh - r) <= 0.5 * a[i])
        room[i] = res_d[i] if len(above) else 10.0  # no resistance overhead: open sky
    out = pd.DataFrame(index=df.index)
    out["sr_support_dist"], out["sr_resist_dist"] = sup_d, res_d
    out["sr_support_touches"], out["sr_resist_touches"] = sup_t, res_t
    out["sr_room_atr"] = room
    p = (df.high.shift() + df.low.shift() + df.close.shift()) / 3  # classic floor pivot of the previous day
    out["sr_pivot_pos"] = (df.close - p) / atr
    return out


def extra(source: str = "pit") -> pd.DataFrame:
    from ..features import _atr
    from .pit import load_prices, load_universe

    keys = pd.read_parquet(OUT / f"dataset_{source}.parquet", columns=["date", "symbol"])
    frames = []
    for i, (s, df) in enumerate(load_prices(set(load_universe().symbol)).items(), 1):
        if len(df) < 260:
            continue
        a = _atr(df)
        f = pd.concat([candle_features(df, a), level_features(df, a)], axis=1).replace([np.inf, -np.inf], np.nan)
        frames.append(f.astype("float32").rename_axis("date").reset_index().assign(symbol=s))
        if i % 250 == 0:
            log.info("stocks %d", i)
    ex = keys.merge(pd.concat(frames), on=["date", "symbol"], how="left")
    ex.to_parquet(OUT / f"extra4_{source}.parquet")
    log.info("pattern features: %d rows; bullish engulfing on %.2f%% of days, support found %.0f%%", len(ex),
             100 * ex.c_bull_engulf.mean(), 100 * ex.sr_support_dist.notna().mean())
    return ex


# ---------------------------------------------------------------- shared
def _data(source: str) -> pd.DataFrame:
    d = load(source).merge(pd.read_parquet(OUT / f"extra4_{source}.parquet"), on=["date", "symbol"], how="left")
    d["gate"] = live_gate(d)
    return d


def _rows(name, sig, calendar, extra=None, cap=None):
    """Each period on its own calendar (so CAGR is per year of that period); `cap` = max open trades per sector."""
    out, split = [], pd.Timestamp(SPLIT)
    sig = sig.dropna(subset=["r_open"])
    for period, part, cal in (("pre-2016", sig[sig.date < SPLIT], [d for d in calendar if d < split]),
                              ("2016+", sig[sig.date >= SPLIT], [d for d in calendar if d >= split]), ("all", sig, calendar)):
        r = portfolio(part, "r_open", "d_open", cost=COST_REAL, calendar=cal, max_per_sector=cap)
        if r:
            out.append({"variant": name, "period": period, "trades": r["trades"], "win_rate": r["win_rate"],
                        "profit_factor": r["profit_factor"], "avg_trade": r["avg_trade"], "cagr": r["cagr"],
                        "max_dd": r["max_dd"], **(extra or {})})
    return out


def _walk(data: pd.DataFrame, variants: dict, first_year: int = 2008) -> pd.DataFrame:
    scored = []
    for year in range(first_year, data.date.max().year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        test = data[(data.date >= start) & (data.date < pd.Timestamp(f"{year + 1}-01-01"))]
        if test.empty:
            continue
        tr = data[(data.date < start - pd.Timedelta(days=HORIZON * 2)) & data.y_open.notna()]
        tr = tr[tr.date.dt.dayofyear % 3 == 0]
        s = test[["date", "symbol", "sector", "y_open", "r_open", "d_open", "gate"]].copy()
        t0 = time.time()
        for i, fn in enumerate(variants.values()):
            s[f"p{i}"] = fn(tr, test)
            s[f"pct{i}"] = s.groupby("date")[f"p{i}"].rank(pct=True)
        log.info("%d: %d variants (%.0fs)", year, len(variants), time.time() - t0)
        scored.append(s)
    return pd.concat(scored)


def _report(df: pd.DataFrame, variants: dict, calendar, tag: str) -> dict:
    d = df.dropna(subset=["y_open"])
    rows = []
    for i, name in enumerate(variants):
        sig = df[(df[f"pct{i}"] >= 0.98) & df.gate].assign(prob=df[f"p{i}"])
        rows += _rows(name, sig, calendar, {"auc": round(float(roc_auc_score(d.y_open, d[f"p{i}"])), 4)})
    out = {"rows": rows}
    (OUT / f"pattern_lab_{tag}.json").write_text(json.dumps(out, indent=1))
    return out


def _lgb(feats):
    return lambda tr, te: lgb.LGBMClassifier(**PARAMS).fit(tr[feats], tr.y_open).predict_proba(te[feats])[:, 1]


# ---------------------------------------------------------------- 1. features
def run(source: str = "pit") -> dict:
    data = _data(source)
    variants = {"live features": _lgb(BASE_FEATURES), "+ candlestick patterns": _lgb(BASE_FEATURES + CANDLES),
                "+ support / resistance": _lgb(BASE_FEATURES + LEVELS),
                "+ candles + support / resistance": _lgb(BASE_FEATURES + CANDLES + LEVELS)}
    return _report(_walk(data, variants), variants, sorted(data.date.unique()), "run")


# ---------------------------------------------------------------- 2. other model types
def models(source: str = "pit") -> dict:
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    data = _data(source)
    f = BASE_FEATURES
    # a few ratios divide by zero; LightGBM treats inf as large, XGBoost refuses it: all three get the same gaps
    data[f] = data[f].replace([np.inf, -np.inf], np.nan)

    def xgb(tr, te):
        m = XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=6, min_child_weight=300, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=-1, verbosity=0)
        return m.fit(tr[f], tr.y_open).predict_proba(te[f])[:, 1]

    def cat(tr, te):
        m = CatBoostClassifier(iterations=400, learning_rate=0.06, depth=6, l2_leaf_reg=3, verbose=0, thread_count=-1)
        return m.fit(tr[f], tr.y_open).predict_proba(te[f])[:, 1]

    cache = {}

    def remember(name, fn):
        def run_(tr, te):
            cache[name] = fn(tr, te)
            return cache[name]
        return run_

    def rank_avg(tr, te):  # average of the three models' daily ranks (scales differ, ranks do not)
        day = te.date.values
        r = [pd.Series(cache[k]).groupby(day).rank(pct=True).values for k in ("lgb", "xgb", "cat")]
        return np.mean(r, axis=0)

    variants = {"LightGBM (live)": remember("lgb", _lgb(f)), "XGBoost": remember("xgb", xgb),
                "CatBoost": remember("cat", cat), "Ensemble: average rank of all three": rank_avg}
    return _report(_walk(data, variants), variants, sorted(data.date.unique()), "models")


# ---------------------------------------------------------------- 3. rules on the live model's scores
def rules(source: str = "pit") -> dict:
    df = pd.read_parquet(OUT / f"model_lab_{source}.parquet",
                         columns=["date", "symbol", "y_open", "r_open", "d_open", "mkt_above_200", "breadth200", "p0", "pct0"])
    df = df.merge(pd.read_parquet(OUT / f"extra4_{source}.parquet"), on=["date", "symbol"], how="left")
    df = df.merge(pd.read_parquet(OUT / f"dataset_{source}.parquet", columns=["date", "symbol", "sector"]),
                  on=["date", "symbol"], how="left")
    calendar = sorted(df.date.unique())
    live = df[(df.pct0 >= 0.98) & live_gate(df)].assign(prob=lambda x: x.p0)
    bull_candle = (live.c_bull_engulf + live.c_hammer + live.c_morning_star + live.c_bull_marubozu) > 0
    variants = {
        "live rule": (live, None),
        "max 2 per sector": (live, 2),
        "max 3 per sector": (live, 3),
        "entry near support (<= 1.5 ATR above it)": (live[live.sr_support_dist <= 1.5], None),
        "room to resistance >= 3.5 ATR (the target)": (live[live.sr_room_atr >= 3.5], None),
        "no resistance within 2 ATR": (live[live.sr_room_atr >= 2], None),
        "bullish candle on the signal day": (live[bull_candle], None),
        "no bearish candle on the signal day": (live[(live.c_bear_engulf + live.c_shooting_star) == 0], None),
    }
    rows = []
    for name, (sig, cap) in variants.items():
        rows += _rows(name, sig, calendar, cap=cap)
    out = {"rows": rows}
    (OUT / f"pattern_lab_rules_{source}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pd.set_option("display.width", 220)
    cmd = sys.argv[1]
    if cmd == "extra":
        extra()
    else:
        print(pd.DataFrame({"run": run, "models": models, "rules": rules}[cmd]()["rows"]).to_string(index=False))
