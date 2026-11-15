"""Ideas from the improvement list, each judged on the honest 22-year test before anything changes in the app.

Same setup as feature_lab / trade_lab: point-in-time universe (delisted stocks included), yearly walk-forward,
next-open entries with gap fills, 0.3% cost, 10-slot portfolio. Pick on pre-2016, confirm on 2016+.

    python -m app.research.edge_lab extra   # weekly and monthly chart features
    python -m app.research.edge_lab run     # do weekly / monthly features improve the model?
    python -m app.research.edge_lab rules   # regime (incl. sideways), liquidity, trend alignment, 1:2 reward
    python -m app.research.edge_lab luck    # is the live rule better than luck? and vs Nifty buy-and-hold

Result (Nov 2026): only the Nifty trend filter (ADX >= 20) beat the live rule in both halves of the test; it is an
optional switch (config.TREND_ADX_MIN) because it trades ~30% less. Weekly / monthly features, timeframe alignment,
stricter liquidity and 1:2 brackets did not improve it. The live rule beat 200 random portfolios. See the README.
"""
import json
import logging
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import STORAGE
from ..features import BASE_FEATURES, _adx
from .exits import managed_outcomes
from .experiment import portfolio
from .feature_lab import load
from .v2 import COST_REAL, HORIZON, OUT, PARAMS, SPLIT

log = logging.getLogger(__name__)
WEEKLY = ["w_rsi", "w_macd_n", "w_dist_10", "w_trend", "w_ret_4", "w_up_8"]
MONTHLY = ["m_rsi", "m_dist_10", "m_trend", "m_ret_3"]
NIFTY = STORAGE / "research" / "prices_max" / "IDX_NSEI.parquet"


# ---------------------------------------------------------------- multi-timeframe features
def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _to_daily(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Each day gets the last COMPLETED week / month (labelled on its last calendar day): no look-ahead."""
    return frame.reindex(frame.index.union(index)).ffill().reindex(index)


def mtf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly and monthly trend of one stock, as a chart reader would check before a daily entry."""
    c = df.close
    w, m = c.resample("W-FRI").last().dropna(), c.resample("ME").last().dropna()
    wf = pd.DataFrame(index=w.index)
    wf["w_rsi"] = _rsi(w)
    macd = w.ewm(span=12, adjust=False).mean() - w.ewm(span=26, adjust=False).mean()
    wf["w_macd_n"] = (macd - macd.ewm(span=9, adjust=False).mean()) / w
    s10, s30 = w.rolling(10).mean(), w.rolling(30).mean()
    wf["w_dist_10"] = w / s10 - 1
    wf["w_trend"] = ((w > s10).astype(float) + (s10 > s30).astype(float)).where(s30.notna())
    wf["w_ret_4"] = w.pct_change(4)
    wf["w_up_8"] = (w.diff() > 0).astype(float).rolling(8).mean()
    mf = pd.DataFrame(index=m.index)
    mf["m_rsi"] = _rsi(m)
    m10 = m.rolling(10).mean()
    mf["m_dist_10"] = m / m10 - 1
    mf["m_trend"] = ((m > m10).astype(float) + (m10.diff() > 0).astype(float)).where(m10.notna())
    mf["m_ret_3"] = m.pct_change(3)
    out = pd.concat([_to_daily(wf, df.index), _to_daily(mf, df.index)], axis=1)
    # 0-3: daily close above its 20-day average, weekly close above its 10-week, monthly above its 10-month
    out["mtf_align"] = ((c > c.ewm(span=20, adjust=False).mean()).astype(float)
                        + (out.w_dist_10 > 0).astype(float) + (out.m_dist_10 > 0).astype(float))
    return out.replace([np.inf, -np.inf], np.nan).astype("float32")


def extra(source: str = "pit") -> pd.DataFrame:
    from .pit import load_prices, load_universe

    keys = pd.read_parquet(OUT / f"dataset_{source}.parquet", columns=["date", "symbol"])
    frames = []
    for i, (s, df) in enumerate(load_prices(set(load_universe().symbol)).items(), 1):
        if len(df) < 260:
            continue
        frames.append(mtf_features(df).rename_axis("date").reset_index().assign(symbol=s))
        if i % 500 == 0:
            log.info("features %d", i)
    ex = keys.merge(pd.concat(frames), on=["date", "symbol"], how="left")
    ex.to_parquet(OUT / f"extra3_{source}.parquet")
    log.info("multi-timeframe features: %d rows, weekly %.1f%% / monthly %.1f%% filled", len(ex),
             100 * ex.w_trend.notna().mean(), 100 * ex.m_trend.notna().mean())
    return ex


def _rows(name, sig, calendar, rcol="r_open", dcol="d_open", extra=None):
    out = []
    sig = sig.dropna(subset=[rcol])
    split = pd.Timestamp(SPLIT)
    for period, part, cal in (("pre-2016", sig[sig.date < SPLIT], [d for d in calendar if d < split]),
                              ("2016+", sig[sig.date >= SPLIT], [d for d in calendar if d >= split]), ("all", sig, calendar)):
        r = portfolio(part, rcol, dcol, cost=COST_REAL, calendar=cal)  # each period on its own calendar, so CAGR is per year of that period
        if r:
            out.append({"variant": name, "period": period, "trades": r["trades"], "win_rate": r["win_rate"],
                        "profit_factor": r["profit_factor"], "avg_trade": r["avg_trade"], "cagr": r["cagr"],
                        "max_dd": r["max_dd"], **(extra or {})})
    return out


# ---------------------------------------------------------------- do weekly / monthly features help the model?
VARIANTS = {"live features": BASE_FEATURES, "+ weekly": BASE_FEATURES + WEEKLY,
            "+ monthly": BASE_FEATURES + MONTHLY, "+ weekly + monthly": BASE_FEATURES + WEEKLY + MONTHLY}


def run(first_year: int = 2008, source: str = "pit") -> dict:
    data = load(source).merge(pd.read_parquet(OUT / f"extra3_{source}.parquet"), on=["date", "symbol"], how="left")
    calendar = sorted(data.date.unique())
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
        for i, feats in enumerate(VARIANTS.values()):
            s[f"p{i}"] = lgb.LGBMClassifier(**PARAMS).fit(tr[feats], tr.y_open).predict_proba(test[feats])[:, 1]
            s[f"pct{i}"] = s.groupby("date")[f"p{i}"].rank(pct=True)
        log.info("%d: %d variants (%.0fs)", year, len(VARIANTS), time.time() - t0)
        scored.append(s)
    df = pd.concat(scored)
    gate = df.mkt_above_200 & (df.breadth200 > 0.5)
    d = df.dropna(subset=["y_open"])
    rows = []
    for i, name in enumerate(VARIANTS):
        sig = df[(df[f"pct{i}"] >= 0.98) & gate].assign(prob=df[f"p{i}"])
        rows += _rows(name, sig, calendar, extra={"auc": round(float(roc_auc_score(d.y_open, d[f"p{i}"])), 4)})
    out = {"rows": rows}
    (OUT / f"edge_lab_run_{source}.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- filters on the live model's own scores
def nifty_regime() -> pd.DataFrame:
    n = pd.read_parquet(NIFTY)
    c = n.close
    e50, e200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    out = pd.DataFrame(index=n.index)
    out["n_bull"] = (c > e50) & (e50 > e200)
    out["n_adx"] = _adx(n)
    out["n_above50"] = c > e50
    out["n_sideways"] = (out.n_adx < 20) & (c.pct_change(60).abs() < 0.05)  # no trend and going nowhere
    return out


def _signals(source: str = "pit") -> tuple[pd.DataFrame, list]:
    """Live-recipe walk-forward scores (model_lab variant 0) with what the filters need."""
    df = pd.read_parquet(OUT / f"model_lab_{source}.parquet",
                         columns=["date", "symbol", "y_open", "r_open", "d_open", "mkt_above_200", "breadth200", "p0", "pct0"])
    df = df.merge(pd.read_parquet(OUT / f"dataset_{source}.parquet", columns=["date", "symbol", "traded_value_20"]),
                  on=["date", "symbol"], how="left")
    df = df.merge(pd.read_parquet(OUT / f"extra3_{source}.parquet", columns=["date", "symbol", "mtf_align"]),
                  on=["date", "symbol"], how="left")
    df = df.merge(nifty_regime(), left_on="date", right_index=True, how="left")
    for c in ("n_bull", "n_above50", "n_sideways"):  # days without a Nifty bar count as "no"
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    return df.rename(columns={"p0": "prob"}), sorted(df.date.unique())


def rules(source: str = "pit") -> dict:
    df, calendar = _signals(source)
    top = df[df.pct0 >= 0.98]
    live = top[top.mkt_above_200 & (top.breadth200 > 0.5)]
    variants = {
        "live rule (ML top 2% + market gate)": live,
        "+ Nifty in an uptrend (close > 50-day > 200-day)": live[live.n_bull],
        "+ Nifty trending (ADX >= 20)": live[live.n_adx >= 20],
        "+ not sideways (skip ADX < 20 and |60d| < 5%)": live[~live.n_sideways],
        "+ Nifty above 50-day": live[live.n_above50],
        "liquidity >= Rs 10 cr/day": live[live.traded_value_20 >= 1e8],
        "liquidity >= Rs 25 cr/day": live[live.traded_value_20 >= 2.5e8],
        "liquidity >= Rs 50 cr/day": live[live.traded_value_20 >= 5e8],
        "+ daily, weekly and monthly trends all up": live[live.mtf_align == 3],
        "+ at least 2 of 3 timeframes up": live[live.mtf_align >= 2],
    }
    rows = []
    for name, sig in variants.items():
        rows += _rows(name, sig, calendar)

    # 1:2 reward-to-risk brackets on the same live signals
    from .pit import load_prices

    from ..features import _atr
    brackets = {"live 3.5 / 2.5 ATR (1:1.4)": (3.5, 2.5), "5.0 / 2.5 ATR (1:2)": (5.0, 2.5), "4.0 / 2.0 ATR (1:2)": (4.0, 2.0)}
    extra = []
    for s, p in load_prices(set(live.symbol)).items():
        a = _atr(p)
        o = pd.DataFrame(index=p.index)
        for i, (t, st) in enumerate(brackets.values()):
            o[f"r{i}"], o[f"d{i}"] = managed_outcomes(p.open, p.high, p.low, p.close, a, t, st, HORIZON)
        extra.append(o.assign(symbol=s).rename_axis("date").reset_index())
    lv = live.merge(pd.concat(extra), on=["date", "symbol"], how="left")
    for i, name in enumerate(brackets):
        rows += _rows(name, lv, calendar, rcol=f"r{i}", dcol=f"d{i}")
    out = {"rows": rows}
    (OUT / f"edge_lab_rules_{source}.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- luck and benchmark
def luck(runs: int = 200, source: str = "pit") -> dict:
    """Overfitting check: the live rule against `runs` portfolios that pick 2% of the same stocks at random on
    the same days (market gate open, liquid). If the model only fitted noise, it would sit inside that spread."""
    df, calendar = _signals(source)
    pool = df[df.mkt_above_200 & (df.breadth200 > 0.5) & (df.traded_value_20 >= 5e7)].dropna(subset=["r_open"])
    live = pool[pool.pct0 >= 0.98]
    out = {"live": {}, "live_trend_filter": {}, "random": {}}
    cals = {"2008+": calendar, "2016+": [d for d in calendar if d >= pd.Timestamp(SPLIT)]}
    for period, mask_fn in (("2008+", lambda x: x), ("2016+", lambda x: x[x.date >= SPLIT])):
        for key, sig in (("live", live), ("live_trend_filter", live[live.n_adx >= 20])):
            r = portfolio(mask_fn(sig), "r_open", "d_open", cost=COST_REAL, calendar=cals[period])
            out[key][period] = {k: r[k] for k in ("trades", "win_rate", "profit_factor", "cagr", "max_dd")}
    rng = np.random.default_rng(7)
    pfs = {"2008+": [], "2016+": []}
    sub = pool[["date", "symbol", "r_open", "d_open"]].copy()
    for i in range(runs):
        sub["prob"] = rng.random(len(sub))
        pick = sub[sub.groupby("date").prob.rank(pct=True) >= 0.98]
        for period in pfs:
            part = pick if period == "2008+" else pick[pick.date >= SPLIT]
            pfs[period].append(portfolio(part, "r_open", "d_open", cost=COST_REAL, calendar=cals[period])["profit_factor"])
        if (i + 1) % 50 == 0:
            log.info("random portfolios %d/%d", i + 1, runs)
    for period, v in pfs.items():
        v = np.array(v, dtype=float)
        live_pf = out["live"][period]["profit_factor"]
        out["random"][period] = {"median_pf": round(float(np.median(v)), 2), "p95_pf": round(float(np.percentile(v, 95)), 2),
                                 "best_pf": round(float(v.max()), 2),
                                 "share_of_random_at_or_above_live": round(float((v >= live_pf).mean()), 3)}
    n = pd.read_parquet(NIFTY).close
    for period, start in (("2008+", "2008-01-01"), ("2016+", SPLIT)):
        s = n[n.index >= start]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        out.setdefault("nifty_buy_and_hold", {})[period] = {
            "cagr": round(float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1), 4),
            "max_dd": round(float((s / s.cummax() - 1).min()), 4)}
    (OUT / f"edge_lab_luck_{source}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pd.set_option("display.width", 220)
    cmd = sys.argv[1]
    if cmd == "extra":
        extra()
    elif cmd in ("run", "rules"):
        print(pd.DataFrame({"run": run, "rules": rules}[cmd]()["rows"]).to_string(index=False))
    elif cmd == "luck":
        print(json.dumps(luck(), indent=1))
