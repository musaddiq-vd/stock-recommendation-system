"""Official NSE corporate event calendar (board meetings for results, dividends, fund raising)."""
import json
import logging
import time
from datetime import date, datetime, timedelta

import requests

from .nse import NSE_DIR, UA

log = logging.getLogger(__name__)
PATH = NSE_DIR / "events.json"
MAX_AGE = 6 * 3600


def _fetch() -> list[dict]:
    s = requests.Session()
    s.headers.update(UA)
    s.get("https://www.nseindia.com/", timeout=20)
    rows = s.get("https://www.nseindia.com/api/event-calendar", timeout=30).json()
    return [{"symbol": r["symbol"], "purpose": r.get("purpose", ""), "desc": r.get("bm_desc", ""),
             "date": datetime.strptime(r["date"], "%d-%b-%Y").date().isoformat()} for r in rows]


def all_events() -> list[dict]:
    """Cached calendar. Keeps past entries we saw earlier so recent results stay known."""
    if not PATH.exists() or time.time() - PATH.stat().st_mtime > MAX_AGE:
        try:
            new = _fetch()
            old = json.loads(PATH.read_text()) if PATH.exists() else []
            seen = {(e["symbol"], e["date"], e["purpose"]) for e in new}
            cutoff = (date.today() - timedelta(days=90)).isoformat()
            merged = new + [e for e in old if (e["symbol"], e["date"], e["purpose"]) not in seen and e["date"] >= cutoff]
            PATH.write_text(json.dumps(merged))
        except Exception as e:
            log.warning("event calendar fetch failed: %s", e)
    return json.loads(PATH.read_text()) if PATH.exists() else []


def for_symbol(symbol: str) -> list[dict]:
    return sorted((e for e in all_events() if e["symbol"] == symbol), key=lambda e: e["date"])


def window_event(symbol: str, days: int = 28) -> dict | None:
    """First official results / dividend event between today and `days` calendar days ahead."""
    today, end = date.today().isoformat(), (date.today() + timedelta(days=days)).isoformat()
    for e in for_symbol(symbol):
        if today <= e["date"] <= end and ("Result" in e["purpose"] or "Dividend" in e["purpose"]):
            return e
    return None


def calendar_horizon() -> str | None:
    ev = all_events()
    return max((e["date"] for e in ev), default=None)


def get_corporate_events(symbol: str) -> dict:
    """Agent tool: official NSE calendar for a stock (upcoming + seen in the last 90 days)."""
    evs = for_symbol(symbol.upper())
    return {"symbol": symbol.upper(), "events": evs, "calendar_covers_until": calendar_horizon(),
            "note": "Official NSE board-meeting calendar. It only lists meetings already announced "
                    "(usually ~2 weeks ahead); no entry does not guarantee no results later in the window."}
