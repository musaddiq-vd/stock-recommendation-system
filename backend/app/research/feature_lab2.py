"""Second round of candidate features, all from the NSE daily price history (what the chart shows).

Same test as feature_lab: yearly walk-forward on the point-in-time universe, live Buy rule, pick on pre-2016,
confirm on 2016+.

    python -m app.research.feature_lab2 extra   # compute the candidates
    python -m app.research.feature_lab2 run     # walk-forward comparison
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
from .v2 import COST_REAL, HORIZON, OUT, PARAMS, SPLIT, _benchmark

log = logging.getLogger(__name__)
GAP = ["gap_5", "overnight_20", "intraday_20"]                     # who is buying: overnight gaps or the session
FLOW = ["upval_20", "amihud_20"]                                    # money on up days, price impact per rupee
RISK = ["beta_120", "idio_vol_60", "max_ret_20", "skew_60"]         # market sensitivity, own noise, lottery days
SHAPE = ["ret_3", "breakout_20", "range_5_60", "inside_10"]         # short pullback, breakout, range squeeze
XS = ["xr_idio_vol_60", "xr_max_ret_20", "xr_upval_20"]
CONTEXT = ["breadth200", "breadth50"]                               # market state as a model input
PRICE = GAP + FLOW + RISK + SHAPE


def price_features(df: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    """Uses only data up to each day. df: open, high, low, close, traded (rupees); mkt: benchmark daily return."""
    o, h, l, c, v = df.open, df.high, df.low, df.close, df.traded
    r = c.pct_change()
    gap, intraday = o / c.shift() - 1, c / o - 1
    m = mkt.reindex(df.index)
    out = pd.DataFrame(index=df.index)
    out["gap_5"] = gap.rolling(5).sum()
    out["overnight_20"] = gap.rolling(20).sum()
    out["intraday_20"] = intraday.rolling(20).sum()
    out["upval_20"] = v.where(r > 0, 0).rolling(20).sum() / v.rolling(20).sum()
    out["amihud_20"] = np.log1p((r.abs() / v.replace(0, np.nan)).rolling(20).mean() * 1e9)
    # the index history has a few missing days: allow partial windows instead of losing 120 days per gap
    beta = r.rolling(120, min_periods=60).cov(m) / m.rolling(120, min_periods=60).var()
    out["beta_120"] = beta
    out["idio_vol_60"] = (r - beta * m).rolling(60, min_periods=30).std()
    out["max_ret_20"] = r.rolling(20).max()
    out["skew_60"] = r.rolling(60).skew()
    out["ret_3"] = c.pct_change(3)
    out["breakout_20"] = c / h.rolling(20).max().shift() - 1
    rng = h - l
    out["range_5_60"] = rng.rolling(5).mean() / rng.rolling(60).mean()
    inside = (h <= h.shift()) & (l >= l.shift())
    out["inside_10"] = inside.rolling(10).mean()
    return out.replace([np.inf, -np.inf], np.nan).astype("float32")


def extra(source: str = "pit") -> pd.DataFrame:
    from .pit import load_prices, load_universe

    keys = pd.read_parquet(OUT / f"dataset_{source}.parquet", columns=["date", "symbol"])
    prices = load_prices(set(load_universe().symbol))
    bench = _benchmark({s: p.assign(volume=p.traded / p.close) for s, p in prices.items()})
    mkt = bench.close.pct_change()
    frames = []
    for i, (s, df) in enumerate(prices.items(), 1):
        if len(df) < 260:
            continue
        frames.append(price_features(df, mkt).rename_axis("date").reset_index().assign(symbol=s))
        if i % 500 == 0:
            log.info("features %d/%d", i, len(prices))
    ex = keys.merge(pd.concat(frames), on=["date", "symbol"], how="left")
    g = ex.groupby("date")
    for c in ("idio_vol_60", "max_ret_20", "upval_20"):
        ex[f"xr_{c}"] = g[c].rank(pct=True).astype("float32")
    ex.to_parquet(OUT / f"extra2_{source}.parquet")
    log.info("candidate features: %d rows, %.1f%% filled", len(ex), 100 * ex.beta_120.notna().mean())
    return ex


VARIANTS = {
    "live features": BASE_FEATURES,
    "+ gap": BASE_FEATURES + GAP,
    "+ flow": BASE_FEATURES + FLOW,
    "+ risk": BASE_FEATURES + RISK,
    "+ shape": BASE_FEATURES + SHAPE,
    "+ all price candidates": BASE_FEATURES + PRICE,
    "+ all price + ranks": BASE_FEATURES + PRICE + XS,
    "+ market context": BASE_FEATURES + CONTEXT,
}


def run(first_year: int = 2008, source: str = "pit", only: list[str] | None = None) -> dict:
    data = load(source).merge(pd.read_parquet(OUT / f"extra2_{source}.parquet"), on=["date", "symbol"], how="left")
    calendar = sorted(data.date.unique())
    variants = {k: v for k, v in VARIANTS.items() if not only or k in only}
    scored = []
    for year in range(first_year, data.date.max().year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        test = data[(data.date >= start) & (data.date < pd.Timestamp(f"{year + 1}-01-01"))]
        if test.empty:
            continue
        tr = data[(data.date < start - pd.Timedelta(days=HORIZON * 2)) & data.y_open.notna()]
        tr = tr[tr.date.dt.dayofyear % 3 == 0]
        s = test[["date", "symbol", "y_open", "r_open", "d_open", "mkt_above_200", "breadth200"]].copy()
        t0 = time.time()
        for i, feats in enumerate(variants.values()):
            s[f"p{i}"] = lgb.LGBMClassifier(**PARAMS).fit(tr[feats], tr.y_open).predict_proba(test[feats])[:, 1]
            s[f"pct{i}"] = s.groupby("date")[f"p{i}"].rank(pct=True)
        log.info("%d: %d variants (%.0fs)", year, len(variants), time.time() - t0)
        scored.append(s)
    df = pd.concat(scored)
    df.to_parquet(OUT / f"feature_lab2_{source}.parquet")
    rows = []
    gate = df.mkt_above_200 & (df.breadth200 > 0.5)
    d = df.dropna(subset=["y_open"])
    for i, name in enumerate(variants):
        sig = df[(df[f"pct{i}"] >= 0.98) & gate].assign(prob=df[f"p{i}"]).dropna(subset=["r_open"])
        auc = round(float(roc_auc_score(d.y_open, d[f"p{i}"])), 4)
        for period, part in (("pre-2016", sig[sig.date < SPLIT]), ("2016+", sig[sig.date >= SPLIT]), ("all", sig)):
            r = portfolio(part, "r_open", "d_open", cost=COST_REAL, calendar=calendar)
            rows.append({"variant": name, "period": period, "trades": r["trades"], "win_rate": r["win_rate"],
                         "profit_factor": r["profit_factor"], "max_dd": r["max_dd"], "avg_trade": r["avg_trade"],
                         "auc": auc})
    out = {"first_year": first_year, "rows": rows}
    (OUT / f"feature_lab2_{source}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if sys.argv[1] == "extra":
        extra()
    else:
        res = run(only=sys.argv[2:] or None)
        pd.set_option("display.width", 200)
        print(pd.DataFrame(res["rows"]).to_string(index=False))
