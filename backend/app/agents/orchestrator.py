"""Scan pipeline: universe → ML screener → validated rubric → research-agent veto → tiers → DB.

Only parts validated on 22 years of walk-forward data move the score (ML rank and the rule-based rubric).
The research agent can only hold a pick back (red flags, negative news, results inside the holding window).
"""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from .. import alerts, db
from ..config import (AGENT_TOP_N, HOLD_MAX_DAYS, HOLD_MIN_DAYS, MIN_AVG_TRADED_VALUE, REGIME_FILTER, RUBRIC_MIN,
                      SCORING_VERSION,
                      STOP_ATR)
from ..data import nse
from ..data.events import window_event
from ..data.live import live_quotes
from ..data.prices import benchmark, load_many
from ..features import add_cross_sectional, compute
from ..model.train import load_model, prob_percentile
from ..research.rubric_proxy import proxy_components
from ..universe import get_index, get_universe
from .research_agent import research
from .tools import get_technicals

log = logging.getLogger(__name__)
_scan_lock = threading.Lock()  # one scan at a time: scans are long, hit Bedrock and share the price cache
STRONG_AT, GOOD_AT = 85, 72


class ScanBusy(RuntimeError):
    pass


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
    """ML probability and rule-based rubric for every Nifty 500 stock on the latest bar."""
    uni = get_universe()
    prices, bench = load_many(uni.symbol.tolist()), benchmark()
    rows = []
    for s, df in prices.items():
        f = compute(df, bench)
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
    comps = proxy_components(data, "long", STOP_ATR)
    data["rubric"] = (comps.sum(axis=1) / 60 * 100).values
    data["rubric_parts"] = comps.round(1).to_dict("records")
    return data.sort_values("prob", ascending=False)


def hist_stats(prob: float) -> dict:
    """Out-of-sample target-hit rate of past signals at or above this probability."""
    bt = load_model()["report"]["backtest"]
    best = None
    for r in bt["by_threshold"]:
        if prob >= r["threshold"] and r["trades"] >= 30:
            best = r
    return best or {"win_rate": bt["base_win_rate"], "avg_ret": 0.0}


def tier(score: float, verdict: str, gates: dict) -> str:
    """Each tier maps to one variant of the 22-year walk-forward, so its published stats are real:
    strong = ML top + regime + rubric (+ agent clear); good = ML top + regime (+ agent clear)."""
    if verdict == "avoid":
        return "avoid"
    if verdict != "buy":
        return "watch"
    if score >= STRONG_AT and all(gates.values()):
        return "strong"
    if score >= GOOD_AT and gates.get("regime"):
        return "good"
    return "watch"


def judge(row, market: dict, res: dict, ml_pct: int) -> dict:
    """Combine the validated score, the gates and the agent's veto into the published pick fields."""
    rubric = float(row.rubric)
    score = 0.6 * ml_pct + 0.4 * rubric
    if market["regime"] == "bearish":
        score -= 8  # buying against the market trend
    official = window_event(row.symbol)  # the official NSE calendar overrides the agent's guess
    if official:
        res["event_in_window"] = True
        res["event_risk"] = f"{official['purpose']} on {official['date']} (official NSE calendar)"
    if res.get("event_in_window"):
        score -= 4
    agent_ok = (not res.get("parse_error") and not res.get("red_flags") and res.get("news_sentiment") != "negative"
                and not res.get("event_in_window"))
    gates = {"regime": bool(market["above_ema200"]) if REGIME_FILTER else True,
             "rubric": rubric >= RUBRIC_MIN, "agent": bool(agent_ok)}
    verdict = "avoid" if res.get("red_flags") else "buy" if agent_ok else "watch"
    score = round(score, 1)
    return {"score": score, "verdict": verdict, "gates": gates, "tier": tier(score, verdict, gates),
            "official_event": official}


def run_scan(scan_id: int, universe: str = "NIFTY100", sector: str | None = None, top_n: int = AGENT_TOP_N):
    progress = []
    plock = threading.Lock()  # agents log from several threads

    def log_step(msg):
        with plock:
            progress.append({"t": datetime.now().strftime("%H:%M:%S"), "msg": msg})
            db.update_scan(scan_id, progress=list(progress))

    try:
        log_step("Loading NSE price data and computing indicators")
        market = market_regime()
        try:
            market["flows"] = nse.fii_dii()
        except Exception:
            market["flows"] = {}
        db.update_scan(scan_id, market=market)
        scored = score_universe()
        members = set(get_index(universe).symbol)
        pool = scored[scored.symbol.isin(members) & (scored.traded_value_20 >= MIN_AVG_TRADED_VALUE)]
        if sector:
            pool = pool[pool.sector.str.lower() == sector.lower()]
        cands = [r for _, r in pool.head(top_n).iterrows()]
        log_step(f"ML screener ranked {len(scored)} stocks; top {len(cands)} from {universe} sent to research "
                 f"(market regime: {market['regime']})")

        def work(r):
            snap = get_technicals(r.symbol)

            def on_event(s, ev):
                log_step(f"{s}: {ev['name']}({', '.join(map(str, ev['input'].values()))})")

            return r, snap, research(r.symbol, r["name"], r.sector, snap, market=market, on_event=on_event)

        picks = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for fu in as_completed([ex.submit(work, r) for r in cands]):
                try:
                    r, snap, res = fu.result()
                except Exception as e:
                    log.exception("agent failed")
                    log_step(f"Agent error: {e}")
                    continue
                if "trade_levels" not in snap:
                    log_step(f"{r.symbol}: no price data, skipped")
                    continue
                ml_pct = prob_percentile(float(r.prob))
                j = judge(r, market, res, ml_pct)
                hs = hist_stats(float(r.prob))
                picks.append({
                    "symbol": r.symbol, "name": r["name"], "sector": r.sector, "indices": r.indices,
                    "scoring_version": SCORING_VERSION,
                    "as_of": str(pd.Timestamp(r.as_of).date()), "ml_prob": round(float(r.prob), 4),
                    "ml_percentile": ml_pct, "hist_win_rate": hs["win_rate"], "hist_avg_ret": hs["avg_ret"],
                    "score": j["score"], "tier": j["tier"], "gates": j["gates"], "official_event": j["official_event"],
                    "rubric_score": round(float(r.rubric), 1), "rubric_parts": r.rubric_parts,
                    "levels": snap["trade_levels"], "hold_days": f"{HOLD_MIN_DAYS}-{HOLD_MAX_DAYS}",
                    "technicals": {k: v for k, v in snap.items() if k not in ("last_30_closes", "trade_levels")},
                    "research": {**res, "verdict": j["verdict"]},
                })
                log_step(f"{r.symbol}: {j['tier']} (ML top {max(1, 100 - ml_pct)}%, rubric {r.rubric:.0f}, "
                         f"agent {'clear' if j['gates']['agent'] else 'flagged'}) → score {j['score']}")
        picks.sort(key=lambda p: -p["score"])
        db.save_picks(scan_id, picks)
        db.update_scan(scan_id, status="done")
        log_step(f"Done: {sum(p['tier'] == 'strong' for p in picks)} strong, {sum(p['tier'] == 'good' for p in picks)} good")
        try:
            alerts.scan_summary(db.get_scan(scan_id))
        except Exception:
            log.exception("scan alert failed")
    except Exception as e:
        log.exception("scan failed")
        db.update_scan(scan_id, status="error", error=str(e)[:500])


def scan_running() -> bool:
    return _scan_lock.locked()


def start_scan(universe="NIFTY100", sector=None, top_n=AGENT_TOP_N) -> int:
    """Start a scan in the background. Raises ScanBusy if another scan is still running."""
    if not _scan_lock.acquire(blocking=False):
        raise ScanBusy("A scan is already running")
    try:
        scan_id = db.create_scan(universe, sector, "news")
    except Exception:
        _scan_lock.release()
        raise

    def _run():
        try:
            run_scan(scan_id, universe, sector, top_n)
        finally:
            _scan_lock.release()

    threading.Thread(target=_run, daemon=True, name=f"scan-{scan_id}").start()
    return scan_id


def evaluate(p: dict, bars: pd.DataFrame, quote: dict | None) -> tuple[str, float | None, int]:
    """Outcome of one long pick: target / stop / timeout / open, its return and bars held.
    Bars strictly after the recommendation day count; stop wins when both are touched in one bar."""
    after = bars[bars.index > pd.Timestamp(p["as_of"])].head(HOLD_MAX_DAYS)
    for i, row in enumerate(after.itertuples(), 1):
        if row.low <= p["stop_loss"]:
            return "stop", p["stop_loss"] / p["entry"] - 1, i
        if row.high >= p["target"]:
            return "target", p["target"] / p["entry"] - 1, i
    if len(after) >= HOLD_MAX_DAYS:
        return "timeout", float(after.close.iloc[-1] / p["entry"] - 1), len(after)
    # today's intraday bar is not in the daily history yet
    last_daily = str(after.index[-1].date()) if len(after) else p["as_of"]
    if quote and quote["date"] > last_daily:
        if quote["low"] <= p["stop_loss"]:
            return "stop", p["stop_loss"] / p["entry"] - 1, len(after) + 1
        if quote["high"] >= p["target"]:
            return "target", p["target"] / p["entry"] - 1, len(after) + 1
    ret = float(after.close.iloc[-1] / p["entry"] - 1) if len(after) else None
    return "open", ret, len(after)


def update_outcomes(use_live: bool = False) -> int:
    """Check every open pick against prices since it was recommended."""
    picks = db.open_picks()
    if not picks:
        return 0
    prices = load_many(sorted({p["symbol"] for p in picks}))
    watch = sorted({p["symbol"] for p in picks if p["tier"] in ("strong", "good")})
    live = live_quotes(watch) if use_live and watch else {}
    n = 0
    for p in picks:
        if json.loads(p["data"]).get("side", "long") != "long":
            db.set_outcome(p["id"], "retired", None, 0)  # legacy short picks are no longer tracked
            continue
        bars = prices.get(p["symbol"])
        if bars is None:
            continue
        outcome, ret, days = evaluate(p, bars, live.get(p["symbol"]))
        db.set_outcome(p["id"], outcome, None if ret is None else round(ret, 4), days)
        alerts.pick_outcome(p, outcome, ret)
        n += 1
    return n
