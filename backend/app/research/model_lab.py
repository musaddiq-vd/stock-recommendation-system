"""Can the same features be trained better? Training-recipe variants for the live Buy rule.

Same test as feature_lab: every variant is trained year by year on data ending before each test year and
judged on the point-in-time universe with the live rule (ML top 2% of the day + Nifty and 200-day breadth
gate, next-open entries, gap fills, 0.3% cost, 10-slot portfolio). Pick on pre-2016, confirm on 2016+.

    python -m app.research.model_lab            # all variants
    python -m app.research.model_lab "seed bag x3" "offset bag x3"
    python -m app.research.model_lab seeds      # live recipe vs time decay on 4 more seeds each

Result (Nov 2026): nothing beat the live recipe. Time decay looked best in one run (profit factor 1.51 vs 1.37)
but averaged 1.42 against 1.38 over five seeds, with a deeper drawdown after 2016, so the live training is unchanged.
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
from .v2 import COST_REAL, HORIZON, OUT, PARAMS, SPLIT

log = logging.getLogger(__name__)
SLOW = {**PARAMS, "n_estimators": 800, "learning_rate": 0.02, "min_child_samples": 500, "colsample_bytree": 0.6}


def _third(tr: pd.DataFrame, k: int = 0) -> pd.DataFrame:
    """Labels of neighbouring days overlap, so one day in three keeps most of the information."""
    return tr[tr.date.dt.dayofyear % 3 == k]


def _fit(tr, te, feats, params=PARAMS, seed=0, weight=None, y=None) -> np.ndarray:
    m = lgb.LGBMClassifier(**{**params, "random_state": seed})
    m.fit(tr[feats], tr.y_open if y is None else y, sample_weight=weight)
    return m.predict_proba(te[feats])[:, 1]


def baseline(tr, te, feats):  # the live recipe
    return _fit(_third(tr), te, feats)


def seed_bag(tr, te, feats):  # average of 3 seeds: less luck in which stocks reach the top 2%
    t = _third(tr)
    return np.mean([_fit(t, te, feats, seed=s) for s in range(3)], axis=0)


def offset_bag(tr, te, feats):  # 3 models, each on a different third of the days: uses every row once
    return np.mean([_fit(_third(tr, k), te, feats, seed=k) for k in range(3)], axis=0)


def recent_10y(tr, te, feats):  # markets change: forget data older than 10 years
    t = _third(tr)
    return _fit(t[t.date >= t.date.max() - pd.DateOffset(years=10)], te, feats)


def time_decay(tr, te, feats, half_life=5.0, seeds=1, seed0=0):
    """Keep all history but halve a row's weight every `half_life` years."""
    t = _third(tr)
    w = 0.5 ** ((t.date.max() - t.date).dt.days / 365.25 / half_life)
    return np.mean([_fit(t, te, feats, weight=w, seed=seed0 + s) for s in range(seeds)], axis=0)


def slow(tr, te, feats):  # more, smaller steps and stronger regularisation
    return _fit(_third(tr), te, feats, params=SLOW)


def profit_label(tr, te, feats):  # learn "closed in profit after costs" instead of "hit the full target"
    t = _third(tr)
    return _fit(t, te, feats, y=(t.r_open > COST_REAL).astype(int))


VARIANTS = {"live recipe": baseline, "seed bag x3": seed_bag, "offset bag x3": offset_bag,
            "recent 10 years": recent_10y, "time decay (5y half-life)": time_decay,
            "slow + regularised": slow, "profit label": profit_label,
            # is the time-decay gain stable, or one lucky setting?
            "time decay 3y": lambda tr, te, f: time_decay(tr, te, f, 3.0),
            "time decay 8y": lambda tr, te, f: time_decay(tr, te, f, 8.0),
            "time decay 5y, seed bag x3": lambda tr, te, f: time_decay(tr, te, f, 5.0, seeds=3)}
# One run of the same recipe moves the profit factor by about +-0.08 (LightGBM row subsampling), so a recipe
# is only better if its seeds beat the live recipe's seeds as a group:  model_lab seeds
SEEDS = range(1, 5)
VARIANTS |= {f"live seed {s}": (lambda tr, te, f, s=s: _fit(_third(tr), te, f, seed=s)) for s in SEEDS}
VARIANTS |= {f"decay seed {s}": (lambda tr, te, f, s=s: time_decay(tr, te, f, seed0=s)) for s in SEEDS}


def run(first_year: int = 2008, source: str = "pit", only: list[str] | None = None) -> dict:
    data = load(source)
    feats = BASE_FEATURES
    calendar = sorted(data.date.unique())
    variants = {k: v for k, v in VARIANTS.items() if not only or k in only}
    scored = []
    for year in range(first_year, data.date.max().year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        test = data[(data.date >= start) & (data.date < pd.Timestamp(f"{year + 1}-01-01"))]
        if test.empty:
            continue
        tr = data[(data.date < start - pd.Timedelta(days=HORIZON * 2)) & data.y_open.notna()]
        s = test[["date", "symbol", "y_open", "r_open", "d_open", "mkt_above_200", "breadth200"]].copy()
        for i, (name, fn) in enumerate(variants.items()):
            t0 = time.time()
            s[f"p{i}"] = fn(tr, test, feats)
            s[f"pct{i}"] = s.groupby("date")[f"p{i}"].rank(pct=True)
            log.info("%d %s (%.0fs)", year, name, time.time() - t0)
        scored.append(s)
    df = pd.concat(scored)
    tag = "_part" if only else ""  # a partial run never overwrites the full comparison
    df.to_parquet(OUT / f"model_lab_{source}{tag}.parquet")
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
    (OUT / f"model_lab_{source}{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = sys.argv[1:]
    if args == ["seeds"]:
        args = [k for k in VARIANTS if " seed " in k and "bag" not in k]
    res = run(only=args or [k for k in VARIANTS if " seed " not in k or "bag" in k])
    pd.set_option("display.width", 200)
    print(pd.DataFrame(res["rows"]).to_string(index=False))
