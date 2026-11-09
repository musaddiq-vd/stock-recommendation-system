"""Long-horizon research: walk-forward ML + barrier grid + filters, 1996 onwards.

Usage:
    python -m app.research.experiment build      # features + outcomes for every barrier config (cached)
    python -m app.research.experiment walk       # yearly walk-forward model scores (cached)
    python -m app.research.experiment grid       # compare barrier configs and filters
"""
import logging
import sys
import time

import lightgbm as lgb
import pandas as pd

from ..config import HOLD_MAX_DAYS, STORAGE
from ..features import BASE_FEATURES, add_cross_sectional, barrier_outcomes, compute
from ..universe import get_universe
from .history import benchmark, load_all

log = logging.getLogger(__name__)
OUT = STORAGE / "research"
DATASET = OUT / "dataset.parquet"
SCORES = OUT / "walk_scores_{side}.parquet"

TARGETS = [1.0, 1.5, 2.0, 2.5, 3.5]
STOPS = [1.0, 1.5, 2.0, 2.5]
LABEL_CFGS = [(3.5, 2.0), (2.0, 2.0)]
HORIZON = HOLD_MAX_DAYS
COST = 0.002  # round-trip brokerage + STT + slippage
PARAMS = dict(n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=300, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def cfg_name(t, s):
    return f"t{t:g}_s{s:g}"


# ---------------------------------------------------------------- build
def build() -> pd.DataFrame:
    prices = load_all()
    bench = benchmark(prices)
    b200 = bench.close / bench.close.ewm(span=200, adjust=False).mean() - 1
    sector = dict(zip(get_universe().symbol, get_universe().sector, strict=False))
    frames = []
    for i, (s, df) in enumerate(prices.items(), 1):
        f = compute(df, bench)
        f = f[f.traded_value_20 > 0]
        keep = BASE_FEATURES[:-9] + ["close", "atr", "traded_value_20", "ema20", "ema50", "ema200", "low_20"]
        f["low_20"] = f.low.rolling(20).min()
        out = f[[c for c in dict.fromkeys(keep) if c in f]].astype("float32")
        for side in ("long", "short"):
            for t in TARGETS:
                for st in STOPS:
                    y, r, d = barrier_outcomes(f.close.values, f.high.values, f.low.values, f.atr.values,
                                               t, st, HORIZON, side)
                    k = f"{side}_{cfg_name(t, st)}"
                    out[f"r_{k}"], out[f"d_{k}"] = r.astype("float32"), d.astype("float32")
                    if (t, st) in LABEL_CFGS:  # training labels only for the candidate label configs
                        out[f"y_{k}"] = y.astype("float32")
        out["symbol"], out["sector"] = s, sector.get(s, "")
        frames.append(out)
        if i % 50 == 0:
            log.info("features %d/%d", i, len(prices))
    data = pd.concat(frames)
    data.index.name = "date"
    data = data.reset_index()
    data = data[data.date >= "1997-06-01"]
    data = add_cross_sectional(data)
    data["mkt_above_200"] = data.date.map(b200).values > 0
    for c in data.select_dtypes("float64").columns:
        data[c] = data[c].astype("float32")
    data.to_parquet(DATASET)
    log.info("dataset rows %d, stocks %d, %s -> %s", len(data), data.symbol.nunique(), data.date.min(), data.date.max())
    return data


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATASET)


# ---------------------------------------------------------------- walk-forward
def walk_forward(side: str = "long", label_cfg: str = "t3.5_s2", first_test_year: int = 2004) -> pd.DataFrame:
    """Each year is scored by a model trained only on data that ended 40 days before that year began.
    The training label is one barrier config; scores are then evaluated on every config."""
    data = load_dataset()
    feats = BASE_FEATURES
    ycol = f"y_{side}_{label_cfg}"
    data = data[data.traded_value_20 > 1e7].dropna(subset=feats)
    scored = []
    for year in range(first_test_year, data.date.max().year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        train = data[(data.date < start - pd.Timedelta(days=HORIZON * 2)) & data[ycol].notna()]
        test = data[(data.date >= start) & (data.date < pd.Timestamp(f"{year + 1}-01-01"))]
        if len(train) < 50_000 or test.empty:
            continue
        # labels of neighbouring days overlap heavily; every 3rd day keeps the information at a third of the cost
        train = train[train.date.dt.dayofyear % 3 == 0]
        t0 = time.time()
        m = lgb.LGBMClassifier(**PARAMS).fit(train[feats], train[ycol])
        s = test[["date", "symbol"]].copy()
        s["prob"] = m.predict_proba(test[feats])[:, 1]
        s["pct"] = s.groupby("date").prob.rank(pct=True)
        scored.append(s)
        log.info("%s %d: train %d rows (%.0fs), test %d rows", side, year, len(train), time.time() - t0, len(test))
    out = pd.concat(scored)
    out.to_parquet(str(SCORES).format(side=side) if label_cfg == "t3.5_s2"
                   else str(SCORES).format(side=f"{side}_{label_cfg}"))
    return out


# ---------------------------------------------------------------- evaluation
def portfolio(sig: pd.DataFrame, rcol: str, dcol: str, slots: int = 10) -> dict:
    """10 equal slots, best-ranked first, one position per stock, positions held until their barrier exit."""
    sig = (sig[["date", "symbol", "prob", rcol, dcol]].rename(columns={rcol: "ret_", dcol: "days_"})
           .sort_values(["date", "prob"], ascending=[True, False]))
    rcol, dcol = "ret_", "days_"
    by_day = {d: g for d, g in sig.groupby("date")}
    all_days = pd.DatetimeIndex(sorted(set(sig.date)))  # signal days are trading days
    cash, open_pos, held, trades, curve = 1.0, [], set(), [], []
    day_idx = {d: i for i, d in enumerate(all_days)}
    for d in all_days:
        i = day_idx[d]
        keep = []
        for p in open_pos:
            if p["exit_i"] <= i:
                cash += p["alloc"] * (1 + p["ret"] - COST)
                held.discard(p["symbol"])
                trades.append((d, p["ret"] - COST))
            else:
                keep.append(p)
        open_pos = keep
        equity = cash + sum(p["alloc"] for p in open_pos)
        if d in by_day:
            for r in by_day[d].itertuples():
                if len(open_pos) >= slots:
                    break
                if r.symbol in held:
                    continue
                alloc = equity / slots
                cash -= alloc
                open_pos.append({"symbol": r.symbol, "alloc": alloc, "ret": float(getattr(r, rcol)),
                                 "exit_i": i + max(1, int(getattr(r, dcol)))})
                held.add(r.symbol)
        curve.append((d, cash + sum(p["alloc"] for p in open_pos)))
    eq = pd.Series(dict(curve))
    t = pd.DataFrame(trades, columns=["date", "ret"])
    if t.empty:
        return {}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    wins, losses = t.ret[t.ret > 0].sum(), -t.ret[t.ret <= 0].sum()
    yearly = eq.resample("YE").last().pct_change()
    yearly.iloc[0] = eq.resample("YE").last().iloc[0] - 1
    return {
        "trades": len(t), "win_rate": round(float((t.ret > 0).mean()), 4),
        "avg_trade": round(float(t.ret.mean()), 4),
        "profit_factor": round(float(wins / losses), 2) if losses > 0 else None,
        "cagr": round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4) if yrs > 0 else None,
        "max_dd": round(float((eq / eq.cummax() - 1).min()), 4),
        "losing_years": int((yearly < 0).sum()), "years": len(yearly),
        "yearly": {str(k.year): round(float(v), 4) for k, v in yearly.items()},
        "equity": eq,
    }


def grid(side: str = "long", top_pct: float = 0.98, filters: dict | None = None) -> pd.DataFrame:
    data = load_dataset()
    scores = pd.read_parquet(str(SCORES).format(side=side))
    df = scores.merge(data, on=["date", "symbol"], how="left")
    df = df[df.pct >= top_pct]
    rows = []
    for fname, fn in (filters or {"none": lambda d: d}).items():
        sub = fn(df)
        for t in TARGETS:
            for s in STOPS:
                k = f"{side}_{cfg_name(t, s)}"
                sig = sub.dropna(subset=[f"r_{k}"])
                res = portfolio(sig, f"r_{k}", f"d_{k}")
                if not res:
                    continue
                rows.append({"filter": fname, "target": t, "stop": s,
                             **{k2: v for k2, v in res.items() if k2 not in ("yearly", "equity")}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if cmd == "build":
        build()
    elif cmd == "walk":
        for side in ("long", "short"):
            walk_forward(side)
    elif cmd == "grid":
        side = sys.argv[2] if len(sys.argv) > 2 else "long"
        res = grid(side)
        pd.set_option("display.width", 200)
        print(res.sort_values("cagr", ascending=False).to_string(index=False))
        res.to_csv(OUT / f"grid_{side}.csv", index=False)
