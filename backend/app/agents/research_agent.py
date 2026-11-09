"""Deep-research agent: investigates one stock and scores it on a fixed rubric.

The agent scores 7 components; the code adds them into `conviction` and derives the verdict,
so results are comparable across stocks instead of clustering around one "safe" number.
"""
import json

from ..config import HOLD_MAX_DAYS, HOLD_MIN_DAYS
from .llm import extract_json, run_agent
from .tools import HANDLERS, specs_for

# name: (max points, what to judge)
RUBRIC = {
    "trend": (20, "Setup structure, judged FOR THE GIVEN SETUP TYPE. Trend/breakout: alignment above EMA20/50/200, "
                  "ADX, higher highs. Pullback: holding EMA50/200 or prior breakout level. Mean-reversion bounce: "
                  "price at a strong support / 52w-low zone, oversold, selling exhausting (smaller red candles, "
                  "volume drying up or a reversal candle). Do NOT mark a bounce setup down just for being below EMAs."),
    "momentum": (15, "Relative strength vs Nifty and sector, 20/60-day returns, RSI regime (not overbought/oversold against the trade)."),
    "volume_flow": (15, "Volume surge, delivery % vs average, futures OI buildup in the trade direction, sector money flow."),
    "fundamentals": (15, "Earnings/revenue growth, margins, valuation vs peers, debt, analyst view. Matters less for a 3-week swing; 8 = unremarkable."),
    "news_catalyst": (15, "Last 30 days news and catalysts (orders, results, upgrades, policy). 8 = no material news."),
    "event_risk": (10, "10 = no results/major event inside the window; 4 = results inside the window; 0 = known negative event pending."),
    "risk_reward": (10, "Is the stop beyond a logical support (resistance for shorts) and is there room to the target before the next barrier?"),
}
RED_FLAGS = ("fraud, SEBI/regulatory action, auditor resignation, promoter pledge spike, default, delisting, "
             "trading suspension, ongoing tax raid")

_rubric_txt = "\n".join(f"  - {k} (0-{m}): {d}" for k, (m, d) in RUBRIC.items())
_score_keys = ", ".join(f'"{k}": <0-{m}>' for k, (m, _) in RUBRIC.items())

SYSTEM = f"""You are a senior equity research analyst for Indian markets (NSE), specialising in
swing trades held {HOLD_MIN_DAYS}-{HOLD_MAX_DAYS} trading days. A quant model has already shortlisted this stock
from ~480 by ranking it near the top; your job is to CONFIRM or REJECT the setup with real research.
{{side_note}}

Process:
1. Review the technical snapshot given (call get_technicals only if you need more).
2. Call get_fundamentals and get_corporate_events (OFFICIAL NSE result/dividend dates; trust them over news).
3. If news tools are available: call get_news for the company, and web_search for anything material.
4. Score every rubric component independently. Use the full range: half of the max means neutral /
   no information; near max means clearly supportive; near zero means clearly against the trade.
   Do not compress scores toward the middle to be safe, and do not anchor on the quant probability.

Rubric:
{_rubric_txt}

Red flags (any one means avoid regardless of score): {RED_FLAGS}.
Only list an item in red_flags if it is one of those exact categories AND confirmed by a source. Ordinary
concerns (weak OI, management change, valuation, soft results) go in bear_case and lower the scores instead.
Base every claim on tool data; if data is missing, give the neutral half score and say so.

Reply with ONLY a JSON object:
{{"scores": {{{_score_keys}}},
  "score_notes": {{"<component>": "<one short reason>", ...}},
  "red_flags": ["..."],
  "summary": "<2-3 sentences, plain English>",
  "bull_case": ["..."], "bear_case": ["..."],
  "news_sentiment": "positive" | "neutral" | "negative" | "unknown",
  "event_risk": "<e.g. results on 2026-11-12, or none>",
  "event_in_window": <true only if results/major event falls inside the next {HOLD_MAX_DAYS} trading days>,
  "key_levels": "<support / resistance comments>",
  "sources": ["<url>", ...]}}"""

LONG_NOTE = "Trade direction: LONG (buy now, sell at target or stop within the window)."
SHORT_NOTE = ("Trade direction: SHORT via stock futures (sell now, buy back lower). Target is BELOW entry, "
              "stop-loss is ABOVE entry. Score every component for a FALLING stock: a strong downtrend scores high "
              "on trend, weak results score high on fundamentals for the short. Watch short-squeeze risk "
              "(heavily sold stocks bouncing on good news). Use side='short' when calling get_technicals. "
              "In bull_case list reasons the short could FAIL; in bear_case list reasons it should WORK.")

GO_AT, WATCH_AT = 65, 45  # conviction thresholds for buy/short and watch


def score_result(out: dict, side: str, mode: str) -> dict:
    """Clamp component scores, sum to conviction, derive the verdict."""
    raw = out.get("scores") or {}
    scores = {}
    for k, (mx, _) in RUBRIC.items():
        try:
            v = float(raw.get(k, mx / 2))
        except (TypeError, ValueError):
            v = mx / 2
        scores[k] = max(0.0, min(float(mx), v))
    if mode != "news" and not raw.get("news_catalyst"):
        scores["news_catalyst"] = RUBRIC["news_catalyst"][0] / 2  # no news research in technical-only mode
    out["scores"] = {k: round(v, 1) for k, v in scores.items()}
    out["conviction"] = int(round(sum(scores.values())))
    flags = [f for f in (out.get("red_flags") or []) if f and str(f).strip().lower() not in ("none", "n/a")]
    out["red_flags"] = flags
    go = "buy" if side == "long" else "short"
    out["verdict"] = "avoid" if flags else go if out["conviction"] >= GO_AT else "watch" if out["conviction"] >= WATCH_AT else "avoid"
    return out


def setup_type(snap: dict, side: str) -> str:
    """Classify what kind of swing the quant model found, so the agent judges the right thing."""
    c, e20, e50, e200 = snap.get("close"), snap.get("ema20"), snap.get("ema50"), snap.get("ema200")
    rsi, hi = snap.get("rsi14", 50), snap.get("pct_from_52w_high", -20)
    if None in (c, e20, e50, e200):
        return "unclassified"
    if side == "short":
        if c < e50 < e200:
            return "trend continuation (downtrend)"
        if c > e20 and rsi > 60:
            return "fade of an overbought rally"
        return "breakdown / pullback short"
    if c > e50 > e200 and hi > -8:
        return "trend continuation / breakout"
    if c > e200 and c < e20:
        return "pullback within an uptrend"
    if c < e50 and rsi < 45:
        return "mean-reversion bounce from oversold"
    return "early reversal / base building"


def research(symbol: str, name: str, sector: str, snapshot: dict, ml_prob: float, mode: str = "news",
             on_event=None, side: str = "long", market: dict | None = None, ml_context: dict | None = None) -> dict:
    system = SYSTEM.replace("{side_note}", LONG_NOTE if side == "long" else SHORT_NOTE)
    ml = ml_context or {}
    ml_txt = (f"Quant screener: this stock ranks in the top {max(1, 100 - ml.get('percentile', 99))}% of NSE stocks for a "
              f"{side} setup. Historically, signals this strong hit the target before the stop "
              f"{ml.get('win_rate', 0) * 100:.0f}% of the time (vs ~21% for a random stock) with an average "
              f"{ml.get('avg_ret', 0) * 100:+.1f}% per trade, because the target is 2x the stop distance.")
    setup = setup_type(snapshot, side)
    user = (f"Stock: {symbol} ({name}), sector: {sector}. Direction: {side.upper()}. Setup type: {setup}.\n"
            f"Market context (Nifty regime, FII/DII net cash flows in Rs crore): {json.dumps(market or {}, default=str)}\n"
            f"{ml_txt}\nTechnical snapshot:\n{json.dumps(snapshot, default=str)}")
    msgs = [{"role": "user", "content": [{"text": user}]}]
    steps, final = [], ""
    for ev in run_agent(system, msgs, specs_for(mode), HANDLERS, max_steps=9):
        if ev["type"] == "tool":
            steps.append({"tool": ev["name"], "input": ev["input"]})
            if on_event:
                on_event(symbol, ev)
        else:
            final = ev["text"]
    try:
        out = extract_json(final)
        out["parse_error"] = False
    except Exception:
        out = {"summary": final[:600] or "The research agent did not return a usable answer.", "bull_case": [],
               "bear_case": [], "news_sentiment": "unknown", "event_risk": "unknown", "sources": [], "scores": {},
               "parse_error": True}
    out = score_result(out, side, mode)
    out["setup"] = setup
    out["steps"] = steps
    return out
