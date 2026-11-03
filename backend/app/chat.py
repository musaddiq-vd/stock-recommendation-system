"""Chat agent: answers questions using the same research tools + latest scan results."""
import json
from datetime import date

from . import db
from .agents.llm import run_agent
from .agents.tools import HANDLERS, SPECS
from .config import HOLD_MAX_DAYS, HOLD_MIN_DAYS

SYSTEM = f"""You are an AI swing-trading research assistant for Indian stocks (NSE: Nifty 50/100/500).
Today is {date.today()}. Swing trades here are held {HOLD_MIN_DAYS}-{HOLD_MAX_DAYS} trading days.
Use tools for real data; never invent prices or news. Use get_latest_picks for the system's current
recommendations. Answer in the user's language (Hindi/Hinglish/English), short and clear, with
entry / target / stop-loss when recommending. End recommendations with a one-line reminder that this
is not financial advice.
Formatting: the chat panel is narrow. Use markdown tables with at most 4 short columns
(e.g. Stock | Entry | Target | Stop) or short bullet lists; no long paragraphs."""


def get_latest_picks(universe: str | None = None, sector: str | None = None, limit: int = 10) -> dict:
    scan = db.latest_scan(universe.upper() if universe else None)
    if not scan:
        return {"error": "No completed scan yet. Ask the user to run a scan from the dashboard."}
    picks = [p for p in scan["picks"] if not sector or sector.lower() in (p["sector"] or "").lower()]
    return {"scan_date": scan["created_at"], "universe": scan["universe"], "market": scan["market"],
            "picks": [{"symbol": p["symbol"], "name": p["name"], "sector": p["sector"], "tier": p["tier"],
                       "score": p["score"], "levels": p["levels"], "verdict": p["research"].get("verdict"),
                       "summary": p["research"].get("summary")} for p in picks[:limit]]}


CHAT_SPECS = SPECS + [{
    "name": "get_latest_picks",
    "description": "Latest swing-trade recommendations from the system's scan, optionally filtered by universe (NIFTY50/NIFTY100/NIFTY500) and sector.",
    "schema": {"type": "object", "properties": {"universe": {"type": "string"}, "sector": {"type": "string"},
                                                "limit": {"type": "integer"}}},
}]
CHAT_HANDLERS = {**HANDLERS, "get_latest_picks": get_latest_picks}


def chat_stream(history: list[dict]):
    """history: [{role: user|assistant, content: str}]. Yields SSE-ready dicts."""
    msgs = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in history if m.get("content")]
    for ev in run_agent(SYSTEM, msgs, CHAT_SPECS, CHAT_HANDLERS, max_steps=8):
        yield json.dumps(ev, default=str)
