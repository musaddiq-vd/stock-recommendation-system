"""Should the live model be trained on survivors (today's Nifty 500) or on the point-in-time universe?

Both models are trained year by year on data before each test year and judged on the SAME point-in-time test
universe with the live rule (ML top 2% + Nifty & 200-day breadth), next-open entries, gap fills, 0.3% cost.

    python -m app.research.training_data
"""
import json
import logging
import time

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..features import BASE_FEATURES, V8_FEATURES
from .experiment import portfolio
from .v2 import COST_REAL, HORIZON, OUT, PARAMS, SPLIT

log = logging.getLogger(__name__)


def run(first_year: int = 2010) -> dict:
    feats = BASE_FEATURES
    pit = pd.read_parquet(OUT / "dataset_pit.parquet")
    yh = pd.read_parquet(OUT / "dataset_yahoo.parquet")
    pit = pit[pit.traded_value_20 > 1e7].dropna(subset=V8_FEATURES)
    yh = yh[yh.traded_value_20 > 1e7].dropna(subset=V8_FEATURES)
    calendar = sorted(pit.date.unique())
    scored = []
    for year in range(first_year, pit.date.max().year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        cut = start - pd.Timedelta(days=HORIZON * 2)
        test = pit[(pit.date >= start) & (pit.date < pd.Timestamp(f"{year + 1}-01-01"))]
        if test.empty:
            continue
        s = test[["date", "symbol", "y_open", "r_open", "d_open", "mkt_above_200", "breadth200"]].copy()
        for name, src in (("pit", pit), ("survivors", yh)):
            tr = src[(src.date < cut) & src.y_open.notna()]
            tr = tr[tr.date.dt.dayofyear % 3 == 0]
            t0 = time.time()
            m = lgb.LGBMClassifier(**PARAMS).fit(tr[feats], tr.y_open)
            s[f"p_{name}"] = m.predict_proba(test[feats])[:, 1]
            s[f"pct_{name}"] = s.groupby("date")[f"p_{name}"].rank(pct=True)
            log.info("%d %s: train %d rows (%.0fs)", year, name, len(tr), time.time() - t0)
        scored.append(s)
    df = pd.concat(scored)
    rows = []
    for name in ("pit", "survivors"):
        sig = df[(df[f"pct_{name}"] >= 0.98) & df.mkt_above_200 & (df.breadth200 > 0.5)].assign(prob=df[f"p_{name}"])
        sig = sig.dropna(subset=["r_open"])
        for period, part in (("all", sig), ("2016+", sig[sig.date >= SPLIT]), ("pre-2016", sig[sig.date < SPLIT])):
            r = portfolio(part, "r_open", "d_open", cost=COST_REAL, calendar=calendar)
            rows.append({"trained_on": name, "period": period, "trades": r["trades"], "win_rate": r["win_rate"],
                         "profit_factor": r["profit_factor"], "max_dd": r["max_dd"], "avg_trade": r["avg_trade"]})
        d = df.dropna(subset=["y_open"])
        rows[-1]["auc_all"] = round(float(roc_auc_score(d.y_open, d[f"p_{name}"])), 4)
    out = {"first_year": first_year, "rows": rows}
    (OUT / "training_data.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    r = run()
    pd.set_option("display.width", 200)
    print(pd.DataFrame(r["rows"]).to_string(index=False))
