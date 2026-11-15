"""Realistic walk-forward research (v2): next-day-open entries, gap fills, slippage, ensembles,
market-breadth regimes, probability calibration, on either universe:

    yahoo  today's Nifty 500 (survivors only, flatters results)
    pit    point-in-time top-500 from NSE bhavcopies, including later-delisted stocks

    python -m app.research.v2 build pit
    python -m app.research.v2 walk pit
    python -m app.research.v2 report pit
"""
import json
import logging
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import BREADTH_MIN, HOLD_MAX_DAYS, STORAGE, TREND_ADX_MIN
from ..features import (BASE_FEATURES, PRICE_FEATURES, V8_FEATURES, _adx, add_cross_sectional, barrier_outcomes, compute,
                        next_open_outcomes)
from ..universe import get_universe
from .experiment import portfolio
from .rubric_proxy import proxy_score

log = logging.getLogger(__name__)
OUT = STORAGE / "research" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
TARGET, STOP, HORIZON = 3.5, 2.5, HOLD_MAX_DAYS
COST_OLD, COST_REAL = 0.002, 0.003  # 0.2% brokerage+taxes, plus 0.1% slippage in the realistic setting
SPLIT = "2016-01-01"
FIRST_TEST_YEAR = 2005
PARAMS = dict(n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=300, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


# ---------------------------------------------------------------- data
def _sources(source: str):
    if source == "pit":
        from .pit import load_prices, load_universe

        uni = load_universe()
        members = {(r.month, r.symbol) for r in uni.itertuples()}
        prices = load_prices(set(uni.symbol))  # only stocks that were ever in the universe (memory)
        for df in prices.values():  # bhavcopies give traded value, not volume
            df["volume"] = df.traded / df.close
        return prices, members
    from .history import load_all

    return load_all(), None


def _benchmark(prices: dict) -> pd.DataFrame:
    from .history import _path, benchmark

    if _path("^NSEI").exists():
        return benchmark(prices)
    rets = pd.DataFrame({s: d.close.pct_change() for s, d in prices.items()}).clip(-0.25, 0.25)
    close = (1 + rets.median(axis=1).fillna(0)).cumprod()
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": np.nan})


def build(source: str = "pit") -> pd.DataFrame:
    prices, members = _sources(source)
    bench = _benchmark(prices)
    b200 = bench.close / bench.close.ewm(span=200, adjust=False).mean() - 1
    sector = dict(zip(get_universe().symbol, get_universe().sector, strict=False))
    keep = PRICE_FEATURES + ["close", "atr", "traded_value_20", "ema20", "ema50", "ema200", "low_20"]
    frames = []
    for i, (s, df) in enumerate(prices.items(), 1):
        if len(df) < 260:
            continue
        f = compute(df[["open", "high", "low", "close", "volume"]], bench)
        out = f[[c for c in dict.fromkeys(keep) if c in f]].astype("float32")
        y, r, d = barrier_outcomes(f.close.values, f.high.values, f.low.values, f.atr.values, TARGET, STOP, HORIZON)
        out["r_close"], out["d_close"] = r.astype("float32"), d.astype("float32")
        y, r, d, _ = next_open_outcomes(f.open.values, f.high.values, f.low.values, f.close.values, f.atr.values,
                                        TARGET, STOP, HORIZON)
        out["y_open"], out["r_open"], out["d_open"] = y.astype("float32"), r.astype("float32"), d.astype("float32")
        out["symbol"], out["sector"] = s, sector.get(s, "Other")
        if members is not None:  # keep only the months in which the stock was in the point-in-time universe
            months = out.index.to_period("M").astype(str)
            out = out[[(m, s) in members for m in months]]
        frames.append(out)
        if i % 250 == 0:
            log.info("features %d/%d", i, len(prices))
    data = pd.concat(frames)
    data.index.name = "date"
    data = data.reset_index()
    data = data[data.date >= "1999-01-01"]
    data = add_cross_sectional(data)
    data["mkt_above_200"] = data.date.map(b200).values > 0
    # market breadth: share of universe stocks above their 50 / 200-day averages that day
    g = data.groupby("date")
    data["breadth50"] = g.dist_ema50.transform(lambda x: (x > 0).mean()).astype("float32")
    data["breadth200"] = g.dist_ema200.transform(lambda x: (x > 0).mean()).astype("float32")
    data["rubric"] = proxy_score(data, "long", STOP).astype("float32").values
    data.to_parquet(OUT / f"dataset_{source}.parquet")
    log.info("%s dataset: %d rows, %d symbols, %s -> %s", source, len(data), data.symbol.nunique(),
             data.date.min().date(), data.date.max().date())
    return data


# ---------------------------------------------------------------- walk-forward
def walk(source: str = "pit") -> pd.DataFrame:
    """Yearly walk-forward. Each year is scored by models trained only on data ending 40 days earlier."""
    data = pd.read_parquet(OUT / f"dataset_{source}.parquet")
    feats = BASE_FEATURES
    data = data[data.traded_value_20 > 1e7].dropna(subset=V8_FEATURES)  # newer features may be empty (no 1-year high yet); the model handles that
    scored = []
    for year in range(FIRST_TEST_YEAR, data.date.max().year + 1):
        start = pd.Timestamp(f"{year}-01-01")
        train = data[(data.date < start - pd.Timedelta(days=HORIZON * 2)) & data.y_open.notna()]
        test = data[(data.date >= start) & (data.date < pd.Timestamp(f"{year + 1}-01-01"))]
        if len(train) < 50_000 or test.empty:
            continue
        train = train[train.date.dt.dayofyear % 3 == 0]  # overlapping labels: a third keeps the information
        t0 = time.time()
        gbm = lgb.LGBMClassifier(**PARAMS).fit(train[feats], train.y_open)
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
        med = train[feats].median()  # logistic regression needs no gaps; LightGBM keeps them
        lr.fit(train[feats].clip(-50, 50).fillna(med), train.y_open)
        s = test[["date", "symbol"]].copy()
        s["p_gbm"] = gbm.predict_proba(test[feats])[:, 1]
        s["p_lr"] = lr.predict_proba(test[feats].clip(-50, 50).fillna(med))[:, 1]
        s["p_ens"] = (s.p_gbm + s.p_lr) / 2
        for c in ("p_gbm", "p_lr", "p_ens"):
            s[c.replace("p_", "pct_")] = s.groupby("date")[c].rank(pct=True)
        scored.append(s)
        log.info("%s %d: train %d (%.0fs), test %d", source, year, len(train), time.time() - t0, len(test))
    out = pd.concat(scored)
    out.to_parquet(OUT / f"walk_{source}.parquet")
    return out


# ---------------------------------------------------------------- evaluation
def _summ(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("yearly", "equity")}


def calibration(df: pd.DataFrame) -> dict:
    """Isotonic calibration fitted before SPLIT, judged after it (target-hit probability)."""
    d = df.dropna(subset=["y_open"])
    fit, hold = d[d.date < SPLIT], d[d.date >= SPLIT]
    iso = IsotonicRegression(out_of_bounds="clip").fit(fit.p_gbm, fit.y_open)
    cal = iso.predict(hold.p_gbm)
    bins = pd.qcut(hold.p_gbm.rank(method="first"), 10, labels=False)
    table = pd.DataFrame({"raw": hold.p_gbm.values, "cal": cal, "hit": hold.y_open.values, "bin": bins.values})
    rel = table.groupby("bin").agg(raw=("raw", "mean"), calibrated=("cal", "mean"), actual=("hit", "mean"))
    return {
        "brier_raw": round(float(brier_score_loss(hold.y_open, hold.p_gbm)), 5),
        "brier_calibrated": round(float(brier_score_loss(hold.y_open, cal)), 5),
        "top_decile": {k: round(float(v), 4) for k, v in rel.iloc[-1].items()},
        "deciles": rel.round(4).to_dict("records"),
    }


def report(source: str = "pit") -> dict:
    data = pd.read_parquet(OUT / f"dataset_{source}.parquet")
    walk_df = pd.read_parquet(OUT / f"walk_{source}.parquet")
    df = walk_df.merge(data, on=["date", "symbol"], how="inner")
    df = df[df.traded_value_20 > 1e7]
    calendar = sorted(data.date.unique())  # every trading day, so holding periods are counted correctly
    rng = np.random.default_rng(0)
    df["p_rand"] = rng.random(len(df))
    df["pct_rand"] = df.groupby("date").p_rand.rank(pct=True)
    def top(col):  # top 2% each day by this model; its own score ranks the picks
        t = df[df[col] >= 0.98]
        return t.assign(prob=t[col.replace("pct_", "p_")])

    gbm, ens = top("pct_gbm"), top("pct_ens")
    variants = {
        "Random pick": (top("pct_rand"), "open", COST_REAL),
        "ML (LightGBM) top 2% · old assumptions (close entry, 0.2% cost)": (gbm, "close", COST_OLD),
        "ML (LightGBM) top 2%": (gbm, "open", COST_REAL),
        "ML (LR) top 2%": (top("pct_lr"), "open", COST_REAL),
        "ML (ensemble) top 2%": (ens, "open", COST_REAL),
        "LightGBM + Nifty>200EMA": (gbm[gbm.mkt_above_200], "open", COST_REAL),
        "LightGBM + breadth200>50%": (gbm[gbm.breadth200 > 0.5], "open", COST_REAL),
        "LightGBM + breadth50>50%": (gbm[gbm.breadth50 > 0.5], "open", COST_REAL),
        "LightGBM + Nifty & breadth200": (gbm[gbm.mkt_above_200 & (gbm.breadth200 > 0.5)], "open", COST_REAL),
        "LightGBM + Nifty + rubric>=70": (gbm[gbm.mkt_above_200 & (gbm.rubric >= 70)], "open", COST_REAL),
        "LightGBM + breadth200 + rubric>=70": (gbm[(gbm.breadth200 > 0.5) & (gbm.rubric >= 70)], "open", COST_REAL),
        "LightGBM + breadth50 + rubric>=70": (gbm[(gbm.breadth50 > 0.5) & (gbm.rubric >= 70)], "open", COST_REAL),
        "LightGBM + breadth50 + rubric>=50": (gbm[(gbm.breadth50 > 0.5) & (gbm.rubric >= 50)], "open", COST_REAL),
        "LightGBM + breadth50>60%": (gbm[gbm.breadth50 > 0.6], "open", COST_REAL),
        "LightGBM + breadth50>50% + max 2 per sector": (gbm[gbm.breadth50 > 0.5], "open", COST_REAL, 2),
        "Ensemble + Nifty + rubric>=70": (ens[ens.mkt_above_200 & (ens.rubric >= 70)], "open", COST_REAL),
    }
    rows, yearly = [], {}
    for name, (sig, mode, cost, *cap) in variants.items():
        rcol, dcol = f"r_{mode}", f"d_{mode}"
        sig = sig.dropna(subset=[rcol])
        kw = {"cost": cost, "calendar": calendar, "max_per_sector": cap[0] if cap else None}
        res = portfolio(sig, rcol, dcol, **kw)
        held = portfolio(sig[sig.date >= SPLIT], rcol, dcol, **kw)
        if not res:
            continue
        rows.append({"variant": name, **_summ(res),
                     "holdout_win": held.get("win_rate") if held else None,
                     "holdout_pf": held.get("profit_factor") if held else None,
                     "holdout_dd": held.get("max_dd") if held else None})
        yearly[name] = res["yearly"]
    aucs = {}
    d = df.dropna(subset=["y_open"])
    for c in ("p_gbm", "p_lr", "p_ens"):
        aucs[c] = {"all": round(float(roc_auc_score(d.y_open, d[c])), 4),
                   "2016+": round(float(roc_auc_score(d[d.date >= SPLIT].y_open, d[d.date >= SPLIT][c])), 4)}
    out = {"source": source, "rows": rows, "auc": aucs, "calibration": calibration(d), "yearly": yearly,
           "universe_symbols": int(data.symbol.nunique())}
    (OUT / f"report_{source}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def nifty_adx() -> pd.Series:
    """Nifty's 14-day ADX by date (Yahoo has the index from Sept 2007; earlier dates are missing)."""
    from .history import _path

    n = pd.read_parquet(_path("^NSEI"))
    return _adx(n).rename("n_adx")


def live_gate(df: pd.DataFrame, breadth_min: float = BREADTH_MIN, trend_min: float = TREND_ADX_MIN) -> pd.Series:
    """The live market gate on research rows: Nifty and breadth above their 200-day averages, and Nifty trending.
    Days before the index has an ADX (2005-2007) pass the trend check, like the app does without the number."""
    gate = df.mkt_above_200 & (df.breadth200 > breadth_min)
    if trend_min:
        adx = df.date.map(nifty_adx())
        gate &= adx.isna() | (adx >= trend_min)
    return gate


def export(source: str = "pit", breadth_min: float = BREADTH_MIN) -> dict:
    """Dashboard summary of the live setting (ML top 2% + Nifty and breadth above their 200-day averages + Nifty
    trending), honest universe, realistic fills. Market filters were chosen on pre-2016 data."""
    data = pd.read_parquet(OUT / f"dataset_{source}.parquet")
    df = pd.read_parquet(OUT / f"walk_{source}.parquet").merge(data, on=["date", "symbol"], how="inner")
    df = df[df.traded_value_20 > 1e7].dropna(subset=["r_open"])
    calendar = sorted(data.date.unique())
    rng = np.random.default_rng(0)
    df["p_rand"] = rng.random(len(df))
    rnd = df[df.groupby("date").p_rand.rank(pct=True) >= 0.98].assign(prob=lambda x: x.p_rand)
    top = df[df.pct_gbm >= 0.98].assign(prob=lambda x: x.p_gbm)
    variants = {"Random pick (same universe)": rnd, "ML top 2%": top,
                "ML + Nifty & breadth above 200-day averages (rule until Nov 2026)":
                    top[live_gate(top, breadth_min, trend_min=0)],
                "ML + market gate + Nifty trending (live setting)": top[live_gate(top, breadth_min)]}
    out = {"source": source, "target_atr": TARGET, "stop_atr": STOP, "split": SPLIT, "cost": COST_REAL,
           "entry": "next-day open with gap fills", "breadth_min": breadth_min, "trend_adx_min": TREND_ADX_MIN,
           "survivorship_free": source == "pit", "symbols": int(data.symbol.nunique()),
           "from": str(df.date.min().date()), "to": str(df.date.max().date()), "variants": []}
    for name, sig in variants.items():
        res = portfolio(sig, "r_open", "d_open", cost=COST_REAL, calendar=calendar)
        held = portfolio(sig[sig.date >= SPLIT], "r_open", "d_open", cost=COST_REAL, calendar=calendar)
        eq = res["equity"].resample("W").last().dropna()
        out["variants"].append({"name": name, **_summ(res), "yearly": res["yearly"],
                                "holdout": _summ(held) if held else None,
                                "equity": [{"date": str(d.date()), "value": round(float(v), 4)} for d, v in eq.items()]})
    (STORAGE / "research" / "summary.json").write_text(json.dumps(out))
    return out


ODDS_PATH = STORAGE / "research" / "odds.json"
ODDS_BUCKETS = ((99, "top 1%"), (98, "top 2%"), (95, "top 3-5%"), (90, "top 6-10%"), (0, "below top 10%"))


def odds(source: str = "pit") -> dict:
    """What happened to past signals like this one: every walk-forward signal (scored by a model that had not
    seen it), grouped by the day's ML rank and whether the market gate was open. Bought at the next open with
    the live target and stop, 0.3% cost. The app shows these observed rates next to each pick as its chance of
    profit; they are frequencies of past outcomes, not a promise for one trade."""
    cols = ["date", "symbol", "y_open", "r_open", "d_open", "mkt_above_200", "breadth200", "traded_value_20"]
    df = pd.read_parquet(OUT / f"walk_{source}.parquet").merge(
        pd.read_parquet(OUT / f"dataset_{source}.parquet", columns=cols), on=["date", "symbol"], how="inner")
    df = df[df.traded_value_20 > 1e7].dropna(subset=["r_open"])
    df["net"] = df.r_open - COST_REAL
    df["pct"] = np.floor(df.pct_gbm * 100)
    gate = live_gate(df)  # same gate as the app, so "gate open" odds describe the days a Buy can happen
    rows, hi = [], 101
    for lo, label in ODDS_BUCKETS:
        for is_open in (True, False):
            part = df[(df.pct >= lo) & (df.pct < hi) & (gate == is_open)]
            recent = part[part.date >= SPLIT]
            if len(part) < 200:
                continue
            rows.append({"gate_open": is_open, "min_pct": lo, "label": label, "signals": len(part),
                         "win": round(float((part.net > 0).mean()), 4),
                         "target": round(float(part.y_open.mean()), 4),
                         "avg_ret": round(float(part.net.mean()), 4),
                         "avg_win": round(float(part.net[part.net > 0].mean()), 4),
                         "avg_loss": round(float(part.net[part.net <= 0].mean()), 4),
                         "avg_days": round(float(part.d_open.mean()), 1),
                         "recent_signals": len(recent),
                         "recent_win": round(float((recent.net > 0).mean()), 4) if len(recent) >= 200 else None})
        hi = lo
    out = {"source": source, "from": str(df.date.min().date()), "to": str(df.date.max().date()),
           "recent_from": SPLIT, "cost": COST_REAL, "rows": rows}
    ODDS_PATH.write_text(json.dumps(out, indent=1))
    return out


CALIBRATION_PATH = STORAGE / "research" / "calibration.json"


def calibration_table(source: str = "pit", liquid: float = 5e7) -> dict:
    """Is the model's probability honest? Out-of-sample signals (each scored by a model that had not seen it)
    grouped by predicted probability: the predicted target-hit rate next to the one that actually happened, and
    the share that closed in profit after the 0.3% cost. Shown in the Backtest tab."""
    cols = ["date", "symbol", "y_open", "r_open", "traded_value_20"]
    df = pd.read_parquet(OUT / f"walk_{source}.parquet", columns=["date", "symbol", "p_gbm", "pct_gbm"]).merge(
        pd.read_parquet(OUT / f"dataset_{source}.parquet", columns=cols), on=["date", "symbol"], how="inner")
    df = df[df.traded_value_20 >= liquid].dropna(subset=["y_open", "r_open"])

    def table(d: pd.DataFrame) -> list[dict]:
        bins = pd.qcut(d.p_gbm.rank(method="first"), 10, labels=False)
        rows = []
        for b, g in d.groupby(bins):
            rows.append({"decile": int(b) + 1, "signals": len(g),
                         "predicted": round(float(g.p_gbm.mean()), 4), "actual_target": round(float(g.y_open.mean()), 4),
                         "actual_profit": round(float((g.r_open - COST_REAL > 0).mean()), 4)})
        top = d[d.pct_gbm >= 0.98]
        rows.append({"decile": "top 2%", "signals": len(top), "predicted": round(float(top.p_gbm.mean()), 4),
                     "actual_target": round(float(top.y_open.mean()), 4),
                     "actual_profit": round(float((top.r_open - COST_REAL > 0).mean()), 4)})
        return rows

    out = {"from": str(df.date.min().date()), "to": str(df.date.max().date()), "cost": COST_REAL, "split": SPLIT,
           "all": table(df), "since_split": table(df[df.date >= SPLIT])}
    CALIBRATION_PATH.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cmd, src = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "pit")
    if cmd == "build":
        build(src)
    elif cmd == "walk":
        walk(src)
    elif cmd == "calib":
        r = calibration_table(src)
        print(pd.DataFrame(r["all"]).to_string(index=False))
        print(pd.DataFrame(r["since_split"]).to_string(index=False))
    elif cmd == "odds":
        print(pd.DataFrame(odds(src)["rows"]).to_string(index=False))
    elif cmd == "export":
        for v in export(src)["variants"]:
            print(v["name"], {k: v[k] for k in ("trades", "win_rate", "profit_factor", "max_dd", "losing_years")})
    elif cmd == "report":
        r = report(src)
        pd.set_option("display.width", 250)
        print(pd.DataFrame(r["rows"]).drop(columns=["losing_years"]).to_string(index=False))
        print("AUC", r["auc"])
        print("calibration", {k: v for k, v in r["calibration"].items() if k != "deciles"})
