"""Can new features or a different model type make the live Buy rule better?

Every variant is trained year by year on data ending before each test year and judged on the point-in-time
universe (delisted stocks included) with the live rule: ML top 2% of the day + Nifty and 200-day breadth gate,
next-open entries, gap fills, 0.3% cost, 10-slot portfolio. Pick on pre-2016 results, confirm on 2016+.

    python -m app.research.feature_lab extra   # compute candidate features from the NSE price history
    python -m app.research.feature_lab run     # walk-forward comparison
"""
import json
import logging
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..features import NEW_PRICE_FEATURES, NEW_XS_FEATURES, V8_FEATURES
from .experiment import portfolio
from .v2 import COST_REAL, HORIZON, OUT, PARAMS, SPLIT

log = logging.getLogger(__name__)
NEW_FEATURES = NEW_PRICE_FEATURES + NEW_XS_FEATURES  # now part of the live model (scoring version 9)


def price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate features from one stock's daily bars, using only data up to each day (same as features.compute)."""
    c, h, l = df.close, df.high, df.low
    r = c.pct_change()
    out = pd.DataFrame(index=df.index)
    out["ret_120"] = c.pct_change(120)
    out["mom_12_1"] = c.shift(21) / c.shift(252) - 1  # classic 12-month momentum, skipping the last month
    out["vol_ratio"] = r.rolling(20).std() / r.rolling(120).std()  # volatility contraction < 1
    pos = pd.Series(np.arange(len(c)), index=c.index, dtype=float)
    at_high = c >= c.rolling(252, min_periods=120).max()
    out["days_since_high"] = (pos - pos.where(at_high).ffill()) / 252
    rng = (h - l).replace(0, np.nan)
    out["clv_10"] = (((c - l) - (h - c)) / rng).rolling(10).mean()  # closes near the high = buying pressure
    out["dd_60"] = c / c.rolling(60).max() - 1
    out["sharpe_60"] = r.rolling(60).mean() / r.rolling(60).std()  # smooth trends rank above jumpy ones
    return out.astype("float32")


def extra(source: str = "pit") -> pd.DataFrame:
    from .pit import load_prices, load_universe

    keys = pd.read_parquet(OUT / f"dataset_{source}.parquet", columns=["date", "symbol"])
    prices = load_prices(set(load_universe().symbol))
    frames = []
    for i, (s, df) in enumerate(prices.items(), 1):
        if len(df) < 260:
            continue
        f = price_features(df)
        f.index.name = "date"
        frames.append(f.reset_index().assign(symbol=s))
        if i % 500 == 0:
            log.info("features %d/%d", i, len(prices))
    ex = pd.concat(frames)
    ex = keys.merge(ex, on=["date", "symbol"], how="left")
    ex.to_parquet(OUT / f"extra_{source}.parquet")
    log.info("extra features: %d rows, %.1f%% filled", len(ex), 100 * ex.mom_12_1.notna().mean())
    return ex


def load(source: str = "pit") -> pd.DataFrame:
    d = pd.read_parquet(OUT / f"dataset_{source}.parquet")
    if "mom_12_1" not in d:  # datasets built before version 9 lack the new features
        ex = pd.read_parquet(OUT / f"extra_{source}.parquet")
        d = d.merge(ex, on=["date", "symbol"], how="left")
        g = d.groupby("date")
        d["xr_mom_12_1"] = g.mom_12_1.rank(pct=True)
        d["xr_sharpe_60"] = g.sharpe_60.rank(pct=True)
        d["xr_liquidity"] = g.traded_value_20.rank(pct=True)
        d["rel_sector_20"] = d.ret_20 - d.sector_ret_20
    return d[d.traded_value_20 > 1e7].dropna(subset=V8_FEATURES)


def _rank_labels(tr: pd.DataFrame) -> np.ndarray:
    """Per-day quintile of the realised trade return (0 = worst, 4 = best) for the ranking objective."""
    return tr.groupby("date").r_open.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False)).values


VARIANTS = {
    "live (base features)": ("clf", V8_FEATURES),
    "base + new features": ("clf", V8_FEATURES + NEW_FEATURES),
    "ranker, base + new": ("rank", V8_FEATURES + NEW_FEATURES),
    "regressor, base + new": ("reg", V8_FEATURES + NEW_FEATURES),
}


def _fit_predict(kind: str, feats: list[str], tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    if kind == "clf":
        return lgb.LGBMClassifier(**PARAMS).fit(tr[feats], tr.y_open).predict_proba(te[feats])[:, 1]
    if kind == "reg":
        y = tr.r_open.clip(-0.2, 0.3)
        return lgb.LGBMRegressor(**PARAMS).fit(tr[feats], y).predict(te[feats])
    tr = tr.sort_values("date")
    groups = tr.groupby("date", sort=True).size().values
    m = lgb.LGBMRanker(**{**PARAMS, "objective": "lambdarank", "lambdarank_truncation_level": 30})
    return m.fit(tr[feats], _rank_labels(tr), group=groups).predict(te[feats])


def run(first_year: int = 2008, source: str = "pit", only: list[str] | None = None) -> dict:
    data = load(source)
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
        for i, (name, (kind, feats)) in enumerate(variants.items()):
            t0 = time.time()
            s[f"p{i}"] = _fit_predict(kind, feats, tr, test)
            s[f"pct{i}"] = s.groupby("date")[f"p{i}"].rank(pct=True)
            log.info("%d %s (%.0fs)", year, name, time.time() - t0)
        scored.append(s)
    df = pd.concat(scored)
    df.to_parquet(OUT / f"feature_lab_{source}.parquet")
    rows = []
    gate = df.mkt_above_200 & (df.breadth200 > 0.5)
    for i, name in enumerate(variants):
        sig = df[(df[f"pct{i}"] >= 0.98) & gate].assign(prob=df[f"p{i}"]).dropna(subset=["r_open"])
        d = df.dropna(subset=["y_open"])
        auc = round(float(roc_auc_score(d.y_open, d[f"p{i}"])), 4)
        for period, part in (("pre-2016", sig[sig.date < SPLIT]), ("2016+", sig[sig.date >= SPLIT]), ("all", sig)):
            r = portfolio(part, "r_open", "d_open", cost=COST_REAL, calendar=calendar)
            rows.append({"variant": name, "period": period, "trades": r["trades"], "win_rate": r["win_rate"],
                         "profit_factor": r["profit_factor"], "max_dd": r["max_dd"], "avg_trade": r["avg_trade"],
                         "auc": auc})
    out = {"first_year": first_year, "rows": rows}
    (OUT / f"feature_lab_{source}.json").write_text(json.dumps(out, indent=1, default=str))
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
