"""Train a LightGBM model on pooled Nifty 500 data with a time-based split."""
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import HOLD_MAX_DAYS, STORAGE
from ..data.prices import benchmark, load_many
from ..features import BASE_FEATURES, add_cross_sectional, compute, label
from ..universe import get_universe

log = logging.getLogger(__name__)
PARAMS = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
)


def model_path(side: str):
    return STORAGE / ("model.joblib" if side == "long" else f"model_{side}.joblib")


def build_dataset(symbols: list[str] | None = None, side: str = "long") -> pd.DataFrame:
    uni = get_universe()
    symbols = symbols or uni.symbol.tolist()
    sector = dict(zip(uni.symbol, uni.sector, strict=False))
    prices, bench = load_many(symbols), benchmark()
    frames = []
    for s, df in prices.items():
        f = label(compute(df, bench), side=side)
        f["symbol"], f["sector"] = s, sector.get(s, "")
        frames.append(f)
    data = pd.concat(frames)
    data.index.name = "date"
    return add_cross_sectional(data.reset_index())


def backtest(test: pd.DataFrame, prob_col="prob", thresholds=(0.35, 0.4, 0.45, 0.5, 0.55, 0.6)) -> dict:
    """Non-overlapping trades per stock: after a signal, skip until the trade exits."""
    out = {"base_win_rate": round(float(test.y.mean()), 4), "base_avg_ret": round(float(test.trade_ret.mean()), 4)}
    rows = []
    for th in thresholds:
        trades = []
        for _, g in test[test[prob_col] >= th].groupby("symbol"):
            next_ok = None
            for r in g.itertuples():
                if next_ok is None or r.date >= next_ok:
                    trades.append(r)
                    next_ok = r.date + pd.Timedelta(days=int(r.days_held * 1.5) + 1)
        if not trades:
            continue
        t = pd.DataFrame(trades)
        rows.append({
            "threshold": th, "trades": len(t), "win_rate": round(float(t.y.mean()), 4),
            "avg_ret": round(float(t.trade_ret.mean()), 4), "avg_days": round(float(t.days_held.mean()), 1),
            "worst": round(float(t.trade_ret.min()), 4),
        })
    out["by_threshold"] = rows
    return out


def _fit_eval(data: pd.DataFrame, feats: list[str], test_months: int, side: str) -> tuple[dict, object]:
    cutoff = data.date.max() - pd.DateOffset(months=test_months)
    embargo = cutoff - pd.Timedelta(days=HOLD_MAX_DAYS * 2)  # avoid label overlap leak
    tr, te = data[data.date < embargo], data[data.date >= cutoff].copy()
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(tr[feats], tr.y)
    te["prob"] = m.predict_proba(te[feats])[:, 1]
    report = {
        "side": side, "trained_at": str(pd.Timestamp.now().date()), "train_rows": len(tr), "test_rows": len(te),
        "auc": round(float(roc_auc_score(te.y, te.prob)), 4),
        "test_from": str(cutoff.date()), "test_to": str(data.date.max().date()),
        "prob_quantiles": {q: round(float(te.prob.quantile(q)), 3) for q in (0.5, 0.9, 0.99)},
        "backtest": backtest(te),
        "portfolio": simulate_portfolio(te, _pick_threshold(te)),
        # Percentile map: lets the UI show how rare a probability is vs history
        "prob_percentiles": [round(float(te.prob.quantile(q / 100)), 4) for q in range(101)],
    }
    return report, m


def _pick_threshold(te: pd.DataFrame) -> float:
    """Probability at the top 2% of the test set: selective, but enough signals to fill 10 slots."""
    return round(float(te.prob.quantile(0.98)), 3)


def simulate_portfolio(te: pd.DataFrame, threshold: float, slots: int = 10, cost: float = 0.003) -> dict:
    """Realistic-ish portfolio: max `slots` open trades, each 1/slots of equity at entry, best probability
    first, one position per stock, round-trip cost `cost` (0.2% charges + 0.1% slippage). Trades are entered at
    the next day's open with gap fills (see features.next_open_outcomes). Positions are carried at cost until exit."""
    dates = np.array(sorted(te.date.unique()))
    sig = te[te.prob >= threshold].sort_values(["date", "prob"], ascending=[True, False])
    by_day = {d: g for d, g in sig.groupby("date")}
    cash, open_pos, held, curve, trades = 1.0, [], set(), [], []
    for i, d in enumerate(dates):
        still = []
        for p in open_pos:
            if p["exit_i"] <= i:
                cash += p["alloc"] * (1 + p["ret"] - cost)
                held.discard(p["symbol"])
                trades.append({"exit": str(pd.Timestamp(d).date()), "ret": p["ret"] - cost, "symbol": p["symbol"]})
            else:
                still.append(p)
        open_pos = still
        equity = cash + sum(p["alloc"] for p in open_pos)
        for r in (by_day[d].itertuples() if d in by_day else []):
            if len(open_pos) >= slots or r.symbol in held:
                continue
            alloc = equity / slots
            if cash < alloc:
                break
            cash -= alloc
            open_pos.append({"symbol": r.symbol, "alloc": alloc, "ret": float(r.trade_ret),
                             "exit_i": i + max(1, int(r.days_held))})
            held.add(r.symbol)
        curve.append((pd.Timestamp(d), cash + sum(p["alloc"] for p in open_pos)))
    eq = pd.Series([v for _, v in curve], index=[d for d, _ in curve])
    dd = eq / eq.cummax() - 1
    monthly = eq.resample("ME").last().pct_change()
    monthly.iloc[0] = eq.resample("ME").last().iloc[0] - 1
    bench = None
    try:
        b = benchmark().close
        b = b[(b.index >= eq.index[0]) & (b.index <= eq.index[-1])]
        bench = (b / b.iloc[0]).resample("W").last()
    except Exception:
        pass
    weekly = eq.resample("W").last()
    t = pd.DataFrame(trades)
    return {
        "threshold": threshold, "slots": slots, "cost": cost,
        "total_return": round(float(eq.iloc[-1] - 1), 4), "max_drawdown": round(float(dd.min()), 4),
        "trades": len(t), "win_rate": round(float((t.ret > 0).mean()), 4) if len(t) else None,
        "avg_trade": round(float(t.ret.mean()), 4) if len(t) else None,
        "nifty_return": round(float(bench.iloc[-1] - 1), 4) if bench is not None and len(bench) else None,
        "equity": [{"date": str(d.date()), "equity": round(float(v), 4),
                    "nifty": round(float(bench.asof(d)), 4) if bench is not None else None,
                    "drawdown": round(float(dd.asof(d)), 4)} for d, v in weekly.items()],
        "monthly": [{"month": d.strftime("%Y-%m"), "ret": round(float(v), 4)} for d, v in monthly.items()],
    }


def quality(report: dict) -> float:
    """Average per-trade return of the backtest buckets with enough trades (what we actually trade)."""
    rows = [r["avg_ret"] for r in report["backtest"]["by_threshold"] if r["trades"] >= 100]
    return float(np.mean(rows)) if rows else -1.0


def train(symbols: list[str] | None = None, test_months: int = 12, side: str = "long",
          feature_sets: dict | None = None) -> dict:
    """Trains each candidate feature set on the same split; keeps the one with the better backtest."""
    data = build_dataset(symbols, side)
    data = data.dropna(subset=BASE_FEATURES + ["y"])
    data = data[data.traded_value_20 > 1e7]
    feature_sets = feature_sets or {"base": BASE_FEATURES}
    results = {name: _fit_eval(data, feats, test_months, side)[0] for name, feats in feature_sets.items()}
    best = max(results, key=lambda k: quality(results[k]))
    feats = feature_sets[best]
    report = results[best]
    report["feature_set"] = best
    report["comparison"] = {k: {"auc": v["auc"], "quality": round(quality(v), 4),
                                "by_threshold": v["backtest"]["by_threshold"]} for k, v in results.items()}
    final = lgb.LGBMClassifier(**PARAMS)
    final.fit(data[feats], data.y)
    report["feature_importance"] = sorted(zip(feats, final.feature_importances_.tolist(), strict=False), key=lambda x: -x[1])
    joblib.dump({"model": final, "features": feats, "report": report}, model_path(side))
    name = "model_report.json" if side == "long" else f"model_report_{side}.json"
    (STORAGE / name).write_text(json.dumps(report, indent=2, default=str))
    _cache.pop(side, None)
    return report


def train_all() -> dict:
    """Retrain the live (long) model on fresh data."""
    return {"long": train(side="long")}


_cache = {}


def load_model(side: str = "long"):
    if side not in _cache:
        path = model_path(side)
        if not path.exists():
            raise FileNotFoundError("Model not trained. Run: python -m app.model.train")
        _cache[side] = joblib.load(path)
    return _cache[side]


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    for side, r in train_all().items():
        print(side, "chosen:", r["feature_set"])
        for k, v in r["comparison"].items():
            print("  ", k, "auc", v["auc"], "quality", v["quality"])
            for row in v["by_threshold"]:
                print("     ", row)
