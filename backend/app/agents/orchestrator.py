"""Scan pipeline: universe → ML screener → research agents → ranker → DB."""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from .. import alerts, db
from ..data import nse
from ..data.events import window_event
from ..data.live import live_quotes
from ..config import (AGENT_TOP_N, HOLD_MAX_DAYS, HOLD_MIN_DAYS, MIN_AVG_TRADED_VALUE, REGIME_FILTER, RUBRIC_MIN,
                      SHORTS_ENABLED, STOP_ATR)
from ..research.rubric_proxy import proxy_score
from ..data.prices import benchmark, load_many
from ..features import add_cross_sectional, add_nse, compute, trade_levels
from ..model.train import load_model, prob_percentile
from ..universe import get_fo_lots, get_index, get_universe
from .research_agent import research
from .tools import get_technicals

log = logging.getLogger(__name__)
_lock = threading.Lock()


def market_regime() -> dict:
    b = benchmark()
    c = b.close
    ema200 = c.ewm(span=200, adjust=False).mean().iloc[-1]
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    last, ret20 = float(c.iloc[-1]), float(c.pct_change(20).iloc[-1])
    if last > ema50 > ema200:
        regime = "bullish"
    elif last < ema200 and last < ema50:
        regime = "bearish"
    else:
        regime = "neutral"
    return {"nifty": round(last, 2), "ret_20d_pct": round(ret20 * 100, 2), "above_ema200": bool(last > ema200),
            "regime": regime, "date": str(c.index[-1].date())}


def score_universe() -> pd.DataFrame:
    """ML probability for every Nifty 500 stock on the latest bar."""
    uni = get_universe()
    prices, bench = load_many(uni.symbol.tolist()), benchmark()
    rows = []
    for s, df in prices.items():
        f = add_nse(compute(df, bench), nse.for_symbol(s))
        r = f.iloc[-1].copy()
        r["symbol"], r["as_of"] = s, f.index[-1]
        rows.append(r)
    data = pd.DataFrame(rows)
    data = data[data.as_of >= data.as_of.max() - pd.Timedelta(days=5)]  # drop stale/suspended
    data["date"] = data.as_of.max()
    data = data.merge(uni[["symbol", "name", "sector", "indices"]], on="symbol", how="left")
    data = add_cross_sectional(data)
    m = load_model()
    data["prob"] = m["model"].predict_proba(data[m["features"]].astype(float))[:, 1]
    data["proxy_long"] = proxy_score(data, "long", STOP_ATR).values
    data["proxy_short"] = proxy_score(data, "short", STOP_ATR).values
    try:
        ms = load_model("short")
        data["prob_short"] = ms["model"].predict_proba(data[ms["features"]].astype(float))[:, 1]
    except FileNotFoundError:
        data["prob_short"] = np.nan
    return data.sort_values("prob", ascending=False)


def hist_stats(prob: float, side: str = "long") -> dict:
    """Out-of-sample backtest stats for signals at or above this probability."""
    bt = load_model(side)["report"]["backtest"]
    best = None
    for r in bt["by_threshold"]:
        if prob >= r["threshold"] and r["trades"] >= 30:
            best = r
    return best or {"win_rate": bt["base_win_rate"], "avg_ret": 0.0}


def tier(score: float, verdict: str | None, gates_ok: bool = True) -> str:
    """strong = high score + agent agrees + every historically validated gate passes."""
    if verdict == "avoid":
        return "avoid"
    if score >= 85 and verdict in ("buy", "short", None) and gates_ok:
        return "strong"
    if score >= 72 and verdict in ("buy", "short", None):
        return "good"
    return "watch"


def run_scan(scan_id: int, universe: str = "NIFTY100", sector: str | None = None, mode: str = "news",
             top_n: int = AGENT_TOP_N):
    progress = []

    def log_step(msg):
        progress.append({"t": datetime.now().strftime("%H:%M:%S"), "msg": msg})
        db.update_scan(scan_id, progress=progress)

    try:
        log_step("Loading NSE price data and computing indicators")
        market = market_regime()
        try:
            market["flows"] = nse.fii_dii()
        except Exception:
            market["flows"] = {}
        db.update_scan(scan_id, market=market)
        try:
            nse.update_recent(5)
        except Exception:
            log.exception("NSE update failed")
        scored = score_universe()
        members = set(get_index(universe).symbol)
        pool = scored[scored.symbol.isin(members) & (scored.traded_value_20 >= MIN_AVG_TRADED_VALUE)]
        if sector:
            pool = pool[pool.sector.str.lower() == sector.lower()]
        jobs = [(r, "long") for _, r in pool.head(top_n).iterrows()]
        log_step(f"ML screener ranked {len(scored)} stocks; top {len(jobs)} long candidates from {universe} "
                 f"(market regime: {market['regime']})")
        # Short ideas only when the market is not bullish, and only F&O stocks (futures needed to hold a short)
        lots = get_fo_lots()
        if SHORTS_ENABLED and market["regime"] != "bullish" and lots and pool.prob_short.notna().any():
            shorts = pool[pool.symbol.isin(lots)].sort_values("prob_short", ascending=False)
            n_short = max(3, top_n // 2)
            jobs += [(r, "short") for _, r in shorts.head(n_short).iterrows()]
            log_step(f"Market {market['regime']}: adding top {min(n_short, len(shorts))} F&O short-sell candidates")

        def work(job):
            r, side = job
            snap = get_technicals(r.symbol, side)
            prob = float(r.prob if side == "long" else r.prob_short)

            def on_event(s, ev):
                args = ", ".join(map(str, ev["input"].values()))
                log_step(f"{s} [{side}]: {ev['name']}({args})")

            pct, hs0 = prob_percentile(prob, side), hist_stats(prob, side)
            ml_context = {"percentile": pct, "win_rate": hs0["win_rate"], "avg_ret": hs0["avg_ret"]}
            res = research(r.symbol, r["name"], r.sector, snap, prob, mode, on_event=on_event, side=side,
                           market=market, ml_context=ml_context)
            return r, side, prob, snap, res

        picks = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for fu in as_completed(futs):
                try:
                    r, side, prob, snap, res = fu.result()
                except Exception as e:
                    log.exception("agent failed")
                    log_step(f"Agent error: {e}")
                    continue
                ml_pct = prob_percentile(prob, side)
                proxy = float(r.proxy_long if side == "long" else r.proxy_short)
                # Score uses only parts validated on 22 years of walk-forward data: the ML rank and the
                # rule-based rubric. An anonymised test showed the LLM's chart reading adds no edge, so the
                # agent acts as a veto (red flags, bad news, results inside the window) instead.
                score = 0.6 * ml_pct + 0.4 * proxy
                if market["regime"] == ("bearish" if side == "long" else "bullish"):
                    score -= 8  # trading against the market trend
                elif side == "short" and market["regime"] == "bearish":
                    score += 4  # shorting with the market trend
                # Official NSE calendar overrides the agent's guess when it lists an event
                official = window_event(r.symbol)
                if official:
                    res["event_in_window"] = True
                    res["event_risk"] = f"{official['purpose']} on {official['date']} (official NSE calendar)"
                if res.get("event_in_window") is True:
                    score -= 4
                score = round(score, 1)
                hs = hist_stats(prob, side)
                # Gates that lifted win rate and cut drawdown over 22 years of walk-forward tests
                regime_ok = (not REGIME_FILTER) or (market["above_ema200"] if side == "long" else not market["above_ema200"])
                news = (res.get("scores") or {}).get("news_catalyst", 7.5)
                agent_ok = not res.get("red_flags") and news > 4 and res.get("event_in_window") is not True
                gates = {"regime": bool(regime_ok), "rubric": proxy >= RUBRIC_MIN, "agent": bool(agent_ok)}
                verdict = "avoid" if res.get("red_flags") else ("buy" if side == "long" else "short") if agent_ok else "watch"
                res["agent_verdict"], res["verdict"] = res.get("verdict"), verdict
                picks.append({
                    "symbol": r.symbol, "name": r["name"], "sector": r.sector, "indices": r.indices, "side": side,
                    "lot_size": lots.get(r.symbol) if side == "short" else None, "official_event": official,
                    "as_of": str(pd.Timestamp(r.as_of).date()), "ml_prob": round(prob, 4),
                    "ml_percentile": ml_pct, "hist_win_rate": hs["win_rate"], "hist_avg_ret": hs["avg_ret"],
                    "score": score, "tier": tier(score, verdict, all(gates.values())), "levels": snap["trade_levels"],
                    "rubric_score": round(proxy, 1), "gates": gates,
                    "hold_days": f"{HOLD_MIN_DAYS}-{HOLD_MAX_DAYS}",
                    "technicals": {k: v for k, v in snap.items() if k not in ("last_30_closes", "trade_levels")},
                    "research": res,
                })
                log_step(f"{r.symbol} [{side}]: {verdict} (ML top {max(1, 100 - ml_pct)}%, rubric {proxy:.0f}, "
                         f"gates {gates}) → score {score}")
        picks.sort(key=lambda p: -p["score"])
        db.save_picks(scan_id, picks)
        db.update_scan(scan_id, status="done")
        for side in ("long", "short"):
            sp = [p for p in picks if p["side"] == side]
            if sp:
                log_step(f"Done {side}: {sum(p['tier'] == 'strong' for p in sp)} strong, "
                         f"{sum(p['tier'] == 'good' for p in sp)} good")
        try:
            alerts.scan_summary(db.get_scan(scan_id))
        except Exception:
            log.exception("scan alert failed")
    except Exception as e:
        log.exception("scan failed")
        db.update_scan(scan_id, status="error", error=str(e))


def start_scan(universe="NIFTY100", sector=None, mode="news", top_n=AGENT_TOP_N) -> int:
    scan_id = db.create_scan(universe, sector, mode)
    threading.Thread(target=run_scan, args=(scan_id, universe, sector, mode, top_n), daemon=True).start()
    return scan_id


def update_outcomes(use_live: bool = False):
    """Check open picks against prices since recommendation: target / stop / timeout."""
    picks = db.open_picks()
    if not picks:
        return 0
    prices = load_many(sorted({p["symbol"] for p in picks}))
    watch = sorted({p["symbol"] for p in picks if p["tier"] in ("strong", "good")})
    live = live_quotes(watch) if use_live and watch else {}
    n = 0
    for p in picks:
        df = prices.get(p["symbol"])
        if df is None:
            continue
        sgn = 1 if json.loads(p["data"]).get("side", "long") == "long" else -1
        after = df[df.index > pd.Timestamp(p["as_of"])].head(HOLD_MAX_DAYS)
        outcome, ret, days = "open", None, len(after)
        for i, row in enumerate(after.itertuples(), 1):
            stop_hit = row.low <= p["stop_loss"] if sgn == 1 else row.high >= p["stop_loss"]
            tgt_hit = row.high >= p["target"] if sgn == 1 else row.low <= p["target"]
            if stop_hit:
                outcome, ret, days = "stop", sgn * (p["stop_loss"] / p["entry"] - 1), i
                break
            if tgt_hit:
                outcome, ret, days = "target", sgn * (p["target"] / p["entry"] - 1), i
                break
        else:
            if len(after) >= HOLD_MAX_DAYS:
                outcome, ret = "timeout", sgn * float(after.close.iloc[-1] / p["entry"] - 1)
            elif len(after):
                ret = sgn * float(after.close.iloc[-1] / p["entry"] - 1)
        # Today's intraday bar (not yet in daily history) for open strong/good picks
        q = live.get(p["symbol"])
        last_daily = str(after.index[-1].date()) if len(after) else p["as_of"]
        if outcome == "open" and q and q["date"] > last_daily:
            stop_hit = q["low"] <= p["stop_loss"] if sgn == 1 else q["high"] >= p["stop_loss"]
            tgt_hit = q["high"] >= p["target"] if sgn == 1 else q["low"] <= p["target"]
            if stop_hit:
                outcome, ret, days = "stop", sgn * (p["stop_loss"] / p["entry"] - 1), days + 1
            elif tgt_hit:
                outcome, ret, days = "target", sgn * (p["target"] / p["entry"] - 1), days + 1
        db.set_outcome(p["id"], outcome, None if ret is None else round(ret, 4), days)
        alerts.pick_outcome(p, outcome, ret)
        n += 1
    return n
