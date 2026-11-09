"""Research agent: checks a shortlisted stock for reasons NOT to take the trade.

History showed the LLM's chart reading adds no edge (anonymised test, AUC 0.48), so the agent does not score
the setup. Its job is the part a price model cannot see: news, filings, official event dates and red flags.
Its findings act as a veto on the validated ML + rubric score.
"""
import json

from ..config import HOLD_MAX_DAYS, HOLD_MIN_DAYS
from .llm import extract_json, run_agent
from .tools import HANDLERS, SPECS

RED_FLAGS = ("fraud, SEBI/regulatory action, auditor resignation, promoter pledge spike, debt default, delisting, "
             "trading suspension, ongoing tax raid")

SYSTEM = f"""You are a risk analyst for Indian equities (NSE). A validated quant model has shortlisted this stock for
a LONG swing trade held {HOLD_MIN_DAYS}-{HOLD_MAX_DAYS} trading days. The price setup is already judged; do not
second-guess the chart. Your only job is to find real-world reasons NOT to buy now.

Process:
1. Call get_corporate_events (official NSE result/dividend dates; trust them over news).
2. Call get_news for the company, then web_search for anything material you find (results, orders, block deals,
   downgrades, regulatory action, litigation, promoter pledging).
3. Call get_fundamentals to check for distress (losses, very high debt).

Rules:
- red_flags: list an item ONLY if it is one of: {RED_FLAGS}, it is serious enough to threaten the stock over the
  next few weeks, AND a source confirms it. NOT red flags (put them in bear_case): small procedural or disclosure
  fines (e.g. SEBI LODR penalties of a few lakh rupees), routine litigation, valuation, soft results, management
  changes. Serious means things like trading bans, fraud or forensic investigations, defaults, large asset seizures.
- news_sentiment: the overall tone of the last 30 days of company news for the next few weeks.
- Base every claim on tool data. If you find nothing material, say so; that is a normal outcome.

Reply with ONLY a JSON object:
{{"news_sentiment": "positive" | "neutral" | "negative" | "unknown",
  "red_flags": ["..."],
  "event_risk": "<e.g. results on 2026-11-12, or none>",
  "event_in_window": <true only if results or a major event falls inside the next {HOLD_MAX_DAYS} trading days>,
  "summary": "<2-3 sentences, plain English: what you found and whether anything argues against the trade>",
  "bull_case": ["..."], "bear_case": ["..."],
  "sources": ["<url>", ...]}}"""

SENTIMENTS = ("positive", "neutral", "negative", "unknown")


def clean(out: dict) -> dict:
    """Normalise the agent's JSON so downstream code can trust every field."""
    flags = [str(f).strip() for f in (out.get("red_flags") or []) if f and str(f).strip().lower() not in ("none", "n/a")]
    sentiment = str(out.get("news_sentiment", "unknown")).lower()
    return {
        "news_sentiment": sentiment if sentiment in SENTIMENTS else "unknown",
        "red_flags": flags,
        "event_risk": str(out.get("event_risk") or "unknown"),
        "event_in_window": out.get("event_in_window") is True,
        "summary": str(out.get("summary") or "")[:1200],
        "bull_case": [str(x) for x in (out.get("bull_case") or [])][:8],
        "bear_case": [str(x) for x in (out.get("bear_case") or [])][:8],
        "sources": [str(x) for x in (out.get("sources") or []) if str(x).startswith("http")][:10],
        "parse_error": bool(out.get("parse_error")),
    }


def research(symbol: str, name: str, sector: str, snapshot: dict, market: dict | None = None,
             on_event=None) -> dict:
    user = (f"Stock: {symbol} ({name}), sector: {sector}.\n"
            f"Market context (Nifty regime, FII/DII net cash flows in Rs crore): {json.dumps(market or {}, default=str)}\n"
            f"Price snapshot (for context only):\n{json.dumps(snapshot, default=str)}")
    msgs = [{"role": "user", "content": [{"text": user}]}]
    steps, final = [], ""
    # temperature 0: the same stock and news should give the same verdict on every run
    for ev in run_agent(SYSTEM, msgs, SPECS, HANDLERS, max_steps=9, temperature=0.0):
        if ev["type"] == "tool":
            steps.append({"tool": ev["name"], "input": ev["input"]})
            if on_event:
                on_event(symbol, ev)
        else:
            final = ev["text"]
    try:
        out = extract_json(final)
    except Exception:
        out = {"summary": final[:600] or "The research agent did not return a usable answer.", "parse_error": True}
    out = clean(out)
    out["steps"] = steps
    return out
