"""Compare the ML strategy against a random baseline and test regime / rubric filters.

    python -m app.research.analyze long
    python -m app.research.analyze short
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd

from .experiment import OUT, SCORES, load_dataset, portfolio
from .rubric_proxy import proxy_score

warnings.filterwarnings("ignore")
CONFIGS = [(3.5, 2.0), (3.5, 2.5), (2.5, 2.5), (2.0, 2.5)]
SPLIT = "2016-01-01"  # filters are tuned on data before this date and checked on data after it


def summarize(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("yearly", "equity")}


def main(side: str):
    data = load_dataset()
    data = data[data.traded_value_20 > 1e7]
    scores = pd.read_parquet(str(SCORES).format(side=side))
    df = scores.merge(data, on=["date", "symbol"], how="inner")

    rng = np.random.default_rng(0)
    rows, yearly = [], {}
    for t, s in CONFIGS:
        k = f"{side}_t{t:g}_s{s:g}"
        stop = s
        base = df.dropna(subset=[f"r_{k}"]).copy()
        base["proxy"] = proxy_score(base, side, stop).values
        top = base[base.pct >= 0.98]
        # random baseline: same number of picks per day, drawn at random from all liquid stocks
        rnd = base.copy()
        rnd["prob"] = rng.random(len(rnd))
        rnd = rnd[rnd.groupby("date").prob.rank(pct=True) >= 0.98]
        regime = top.mkt_above_200 if side == "long" else ~top.mkt_above_200
        variants = {
            "random baseline": rnd,
            "ML top 2%": top,
            "ML top 2% + regime": top[regime],
        }
        # rubric threshold tuned only on the pre-2016 period
        pre = top[regime & (top.date < SPLIT)]
        best_th, best_q = None, -9
        for th in (40, 50, 55, 60, 65, 70):
            sub = pre[pre.proxy >= th]
            if len(sub) < 300:
                continue
            q = sub[f"r_{k}"].mean()
            if q > best_q:
                best_th, best_q = th, q
        if best_th is not None:
            variants[f"ML top 2% + regime + rubric>={best_th}"] = top[regime & (top.proxy >= best_th)]
        for name, sig in variants.items():
            for period, part in (("all", sig), ("pre-2016", sig[sig.date < SPLIT]), ("2016+", sig[sig.date >= SPLIT])):
                res = portfolio(part, f"r_{k}", f"d_{k}")
                if not res:
                    continue
                rows.append({"cfg": f"t{t:g}/s{s:g}", "variant": name, "period": period, **summarize(res)})
                if period == "all":
                    yearly[f"t{t:g}/s{s:g} | {name}"] = res["yearly"]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"analysis_{side}.csv", index=False)
    (OUT / f"analysis_{side}_yearly.json").write_text(json.dumps(yearly, indent=1))
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))




def export(side: str = "long", target: float = 3.5, stop: float = 2.5, rubric_min: float = 70) -> dict:
    """Write the chosen strategy's 22-year walk-forward results for the dashboard."""
    data = load_dataset()
    data = data[data.traded_value_20 > 1e7]
    scores = pd.read_parquet(str(SCORES).format(side=side))
    k = f"{side}_t{target:g}_s{stop:g}"
    base = scores.merge(data, on=["date", "symbol"], how="inner").dropna(subset=[f"r_{k}"])
    base["proxy"] = proxy_score(base, side, stop).values
    rng = np.random.default_rng(0)
    rnd = base.copy()
    rnd["prob"] = rng.random(len(rnd))
    rnd = rnd[rnd.groupby("date").prob.rank(pct=True) >= 0.98]
    top = base[base.pct >= 0.98]
    chosen = top[top.mkt_above_200 & (top.proxy >= rubric_min)]
    variants = {"Random pick (same universe)": rnd, "ML top 2%": top, "ML + regime + rubric (live setting)": chosen}
    out = {"side": side, "target_atr": target, "stop_atr": stop, "rubric_min": rubric_min, "split": SPLIT,
           "from": str(base.date.min().date()), "to": str(base.date.max().date()), "variants": []}
    for name, sig in variants.items():
        res = portfolio(sig, f"r_{k}", f"d_{k}")
        held = portfolio(sig[sig.date >= SPLIT], f"r_{k}", f"d_{k}")
        eq = res["equity"].resample("W").last().dropna()
        out["variants"].append({
            "name": name, **summarize(res), "yearly": res["yearly"],
            "holdout": summarize(held) if held else None,
            "equity": [{"date": str(d.date()), "value": round(float(v), 4)} for d, v in eq.items()],
        })
    (OUT / "summary.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        r = export()
        for v in r["variants"]:
            print(v["name"], {k: v[k] for k in ("trades", "win_rate", "profit_factor", "cagr", "max_dd", "losing_years")})
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "long")
