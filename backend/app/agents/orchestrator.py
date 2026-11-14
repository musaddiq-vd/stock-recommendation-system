"""Scan pipeline: universe → ML screener → validated rubric → research-agent veto → tiers → DB.

Only parts validated on 22 years of walk-forward data move the score (ML rank and the rule-based rubric).
The research agent can only hold a pick back (red flags, negative news, results inside the holding window).
"""
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from .. import alerts, db
from ..config import (AGENT_TOP_N, BREADTH_MIN, HOLD_MAX_DAYS, HOLD_MIN_DAYS, MIN_AVG_TRADED_VALUE,
                      ML_TOP_PCT,
                      SCORING_VERSION,
                      STOP_ATR, STORAGE)
from ..data import nse
from ..data.events import window_event
from ..data.live import live_quotes
from ..data.prices import benchmark, load_many
from ..features import add_cross_sectional, compute
from ..model.train import load_model
from ..research.rubric_proxy import proxy_components
from ..universe import get_index, get_universe
from .research_agent import research
from .tools import get_technicals

log = logging.getLogger(__name__)
_scan_lock = threading.Lock()  # one scan at a time: scans are long, hit Bedrock and share the price cache


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
    # today's cross-sectional rank among liquid stocks, exactly how the backtest picked its "top 2%" each day
    liquid = data.traded_value_20 >= MIN_AVG_TRADED_VALUE
    data["xs_pct"] = np.floor(data.prob.where(liquid).rank(pct=True) * 100).fillna(0).astype(int)
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


_odds: dict = {}


def signal_odds(ml_pct: int, gate_open: bool) -> dict | None:
    """Observed outcomes of past walk-forward signals with the same ML rank and market state
    (app/research/v2.py odds). None until that research file exists."""
    path = STORAGE / "research" / "odds.json"
    if not path.exists():
        return None
    stamp = path.stat().st_mtime
    if _odds.get("stamp") != stamp:
        _odds.update(stamp=stamp, data=json.loads(path.read_text()))
    data = _odds["data"]
    for r in data["rows"]:  # rows are ordered best rank first
        if r["gate_open"] == gate_open and ml_pct >= r["min_pct"]:
            return {**r, "from": data["from"], "to": data["to"]}
    return None


def tier(verdict: str, gates: dict) -> str:
    """One Buy tier ("strong"), only when every gate passes: it maps exactly to the live-setting variant of
    the survivorship-free walk-forward (ML top 2% + Nifty and breadth above their 200-day averages)."""
    if verdict == "avoid":
        return "avoid"
    return "strong" if verdict == "buy" and all(gates.values()) else "watch"


def judge(row, market: dict, res: dict, ml_pct: int) -> dict:
    """Gates + the agent's veto -> the published pick fields. The score is the ML percentile: the rule-based
    rubric, an ensemble and probability calibration did not improve the survivorship-free backtest, so they
    do not move the decision (the rubric is still shown for context)."""
    official = window_event(row.symbol)  # the official NSE calendar overrides the agent's guess
    if official:
        res["event_in_window"] = True
        res["event_risk"] = f"{official['purpose']} on {official['date']} (official NSE calendar)"
    agent_ok = (not res.get("parse_error") and not res.get("red_flags") and res.get("news_sentiment") != "negative"
                and not res.get("event_in_window"))
    gates = {"ml_top": ml_pct >= ML_TOP_PCT,
             "market": bool(market.get("above_ema200")) and market.get("breadth200", 0) > BREADTH_MIN,
             "agent": bool(agent_ok)}
    verdict = "avoid" if res.get("red_flags") else "buy" if agent_ok else "watch"
    return {"score": float(ml_pct), "verdict": verdict, "gates": gates, "tier": tier(verdict, gates),
            "official_event": official}


def mark_top_pick(picks: list[dict]) -> dict | None:
    """Sort picks best first and flag the day's single best stock.

    Order: Buy tier first, then ML percentile, then the raw ML probability (many picks share the same
    percentile, so the probability breaks ties instead of the order the agents happened to finish in).
    The top pick is the best stock the research agent cleared; it is a Buy only if its tier says so."""
    rank = {"strong": 0, "watch": 1, "avoid": 2}
    picks.sort(key=lambda p: (rank.get(p["tier"], 3), -p["score"], -p.get("ml_prob", 0)))
    for p in picks:
        p["top_pick"] = False
    best = next((p for p in picks if p["tier"] != "avoid" and p.get("gates", {}).get("agent")), None)
    if best:
        best["top_pick"] = True
    return best


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
        # market breadth: share of liquid Nifty 500 stocks above their 200-day average
        liquid = scored[scored.traded_value_20 >= MIN_AVG_TRADED_VALUE]
        market["breadth200"] = round(float((liquid.dist_ema200 > 0).mean()), 3)
        db.update_scan(scan_id, market=market)
        members = set(get_index(universe).symbol)
        # only stocks with a bar for the latest session: a pick must be judged on today's data, not a stale bar
        fresh = scored.as_of == scored.date
        pool = scored[scored.symbol.isin(members) & (scored.traded_value_20 >= MIN_AVG_TRADED_VALUE) & fresh]
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
                ml_pct = int(r.xs_pct)
                j = judge(r, market, res, ml_pct)
                hs = hist_stats(float(r.prob))
                picks.append({
                    "symbol": r.symbol, "name": r["name"], "sector": r.sector, "indices": r.indices,
                    "scoring_version": SCORING_VERSION,
                    "as_of": str(pd.Timestamp(r.as_of).date()), "ml_prob": round(float(r.prob), 4),
                    "ml_percentile": ml_pct, "hist_win_rate": hs["win_rate"], "hist_avg_ret": hs["avg_ret"],
                    "score": j["score"], "tier": j["tier"], "gates": j["gates"], "official_event": j["official_event"],
                    "odds": signal_odds(ml_pct, j["gates"]["market"]),
                    "rubric_score": round(float(r.rubric), 1), "rubric_parts": r.rubric_parts,
                    "levels": snap["trade_levels"], "hold_days": f"{HOLD_MIN_DAYS}-{HOLD_MAX_DAYS}",
                    "technicals": {k: v for k, v in snap.items() if k not in ("last_30_closes", "trade_levels")},
                    "research": {**res, "verdict": j["verdict"]},
                })
                log_step(f"{r.symbol}: {'Buy' if j['tier'] == 'strong' else j['tier']} (ML top {max(1, 100 - ml_pct)}%, "
                         f"agent {'clear' if j['gates']['agent'] else 'flagged'})")
        mark_top_pick(picks)
        db.save_picks(scan_id, picks)
        db.update_scan(scan_id, status="done")
        log_step(f"Done: {sum(p['tier'] == 'strong' for p in picks)} Buy · Nifty above 200-day average: "
                 f"{'yes' if market['above_ema200'] else 'no'} · breadth {market['breadth200']:.0%}")
        try:
            alerts.scan_summary(db.get_scan(scan_id))
        except Exception:
            log.exception("scan alert failed")
    except Exception as e:
        log.exception("scan failed")
        db.update_scan(scan_id, status="error", error=str(e)[:500])
        alerts.ops_alert(f"scan:{scan_id}", f"scan #{scan_id} ({universe}) failed: {str(e)[:200]}")


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
    """Outcome of one long pick as a user could actually trade it: target / stop / timeout / skipped / open.

    Bought at the OPEN of the first session after the recommendation (not the close it was computed on),
    with the published target and stop. A gap through a level fills at that day's open, so a gap below the
    stop loses more than the stop. If the stock already opens beyond the target or stop on entry day, the
    trade is not taken ("skipped"). Stop wins when both levels are touched in one bar."""
    stop, target = p["stop_loss"], p["target"]
    after = bars[bars.index > pd.Timestamp(p["as_of"])].head(HOLD_MAX_DAYS)
    if after.empty:
        return "open", None, 0
    entry = float(after.open.iloc[0])
    if not (stop < entry < target):
        return "skipped", None, 0
    for i, row in enumerate(after.itertuples(), 1):
        if i > 1 and row.open <= stop:
            return "stop", row.open / entry - 1, i
        if row.low <= stop:
            return "stop", stop / entry - 1, i
        if i > 1 and row.open >= target:
            return "target", row.open / entry - 1, i
        if row.high >= target:
            return "target", target / entry - 1, i
    if len(after) >= HOLD_MAX_DAYS:
        return "timeout", float(after.close.iloc[-1] / entry - 1), len(after)
    # today's intraday bar is not in the daily history yet
    if quote and quote["date"] > str(after.index[-1].date()):
        if quote["low"] <= stop:
            return "stop", stop / entry - 1, len(after) + 1
        if quote["high"] >= target:
            return "target", target / entry - 1, len(after) + 1
    return "open", float(after.close.iloc[-1] / entry - 1), len(after)


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
