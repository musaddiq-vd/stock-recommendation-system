"""SQLite storage for scans, picks and outcome tracking."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, universe TEXT, sector TEXT, mode TEXT,
  status TEXT, progress TEXT, market TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS picks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER, symbol TEXT, data TEXT,
  score REAL, tier TEXT, entry REAL, target REAL, stop_loss REAL, as_of TEXT,
  outcome TEXT, outcome_ret REAL, outcome_days INTEGER);
CREATE INDEX IF NOT EXISTS picks_scan ON picks(scan_id);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=30000")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.execute("PRAGMA journal_mode=WAL")  # readers do not block the scan thread's writes
        c.executescript(SCHEMA)
        # a scan cannot survive a restart: close any left "running" so the dashboard stops polling it
        c.execute("UPDATE scans SET status='error', error='Interrupted by a server restart' WHERE status='running'")


def create_scan(universe, sector, mode) -> int:
    with conn() as c:
        cur = c.execute("INSERT INTO scans(created_at, universe, sector, mode, status, progress) VALUES (?,?,?,?,?,?)",
                        (datetime.now().isoformat(timespec="seconds"), universe, sector, mode, "running", "[]"))
        return cur.lastrowid


def update_scan(scan_id, **kw):
    if "progress" in kw or "market" in kw:
        kw = {k: json.dumps(v) if k in ("progress", "market") else v for k, v in kw.items()}
    sets = ", ".join(f"{k}=?" for k in kw)
    with conn() as c:
        c.execute(f"UPDATE scans SET {sets} WHERE id=?", (*kw.values(), scan_id))


def save_picks(scan_id, picks: list[dict]):
    with conn() as c:
        c.execute("DELETE FROM picks WHERE scan_id=?", (scan_id,))
        for p in picks:
            c.execute("INSERT INTO picks(scan_id, symbol, data, score, tier, entry, target, stop_loss, as_of) "
                      "VALUES (?,?,?,?,?,?,?,?,?)",
                      (scan_id, p["symbol"], json.dumps(p, default=str), p["score"], p["tier"],
                       p["levels"]["entry"], p["levels"]["target"], p["levels"]["stop_loss"], p["as_of"]))


def _scan_row(r):
    d = dict(r)
    d["progress"] = json.loads(d["progress"] or "[]")
    d["market"] = json.loads(d["market"]) if d["market"] else None
    return d


def get_scan(scan_id):
    with conn() as c:
        r = c.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
        if not r:
            return None
        scan = _scan_row(r)
        rows = c.execute("SELECT * FROM picks WHERE scan_id=? ORDER BY score DESC", (scan_id,)).fetchall()
    scan["picks"] = [{**json.loads(p["data"]), "outcome": p["outcome"], "outcome_ret": p["outcome_ret"]} for p in rows]
    return scan


def latest_scan(universe=None, done_only=True):
    q, args = "SELECT id FROM scans WHERE 1=1", []
    if done_only:
        q += " AND status='done'"
    if universe:
        q += " AND universe=?"
        args.append(universe)
    with conn() as c:
        r = c.execute(q + " ORDER BY id DESC LIMIT 1", args).fetchone()
    return get_scan(r["id"]) if r else None


def list_scans(limit=20):
    with conn() as c:
        rows = c.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{k: v for k, v in _scan_row(r).items() if k != "progress"} for r in rows]


def open_picks():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM picks WHERE outcome IS NULL OR outcome='open'").fetchall()]


def set_outcome(pick_id, outcome, ret, days):
    with conn() as c:
        c.execute("UPDATE picks SET outcome=?, outcome_ret=?, outcome_days=? WHERE id=?", (outcome, ret, days, pick_id))


def performance():
    # the same stock recommended on the same day by several scans is one trade, not many
    with conn() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT tier, outcome, outcome_ret FROM picks WHERE id IN ("
            " SELECT MAX(id) FROM picks WHERE outcome IN ('target','stop','timeout') GROUP BY symbol, as_of)"
        ).fetchall()]
    out = {}
    for r in rows:
        t = out.setdefault(r["tier"], {"trades": 0, "wins": 0, "sum_ret": 0.0})
        t["trades"] += 1
        t["wins"] += r["outcome"] == "target"
        t["sum_ret"] += r["outcome_ret"] or 0
    return {k: {"trades": v["trades"], "win_rate": round(v["wins"] / v["trades"], 3),
                "avg_ret": round(v["sum_ret"] / v["trades"], 4)} for k, v in out.items()}


# ---------- settings / paper trades / alerts ----------
EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS paper_trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, opened_on TEXT, scan_id INTEGER, symbol TEXT,
  side TEXT, qty INTEGER, lots INTEGER, entry REAL, target REAL, stop_loss REAL, status TEXT,
  exit_price REAL, exit_date TEXT, pnl REAL, last_price REAL, last_checked TEXT);
CREATE TABLE IF NOT EXISTS alerts_sent (key TEXT PRIMARY KEY, sent_at TEXT);
"""
DEFAULT_SETTINGS = {"capital": "500000", "risk_pct": "1", "margin_pct": "20", "alerts_enabled": "1",
                    "telegram_bot_token": "", "telegram_chat_id": ""}


def init_extra():
    with conn() as c:
        c.executescript(EXTRA_SCHEMA)


def get_settings() -> dict:
    with conn() as c:
        rows = dict(c.execute("SELECT key, value FROM settings").fetchall())
    return {**DEFAULT_SETTINGS, **rows}


def set_settings(values: dict):
    with conn() as c:
        for k, v in values.items():
            if k in DEFAULT_SETTINGS and v is not None:
                c.execute("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (k, str(v)))


def add_paper(t: dict) -> int:
    cols = ["created_at", "opened_on", "scan_id", "symbol", "side", "qty", "lots", "entry", "target", "stop_loss", "status"]
    t = {**t, "created_at": datetime.now().isoformat(timespec="seconds"), "status": "open"}
    with conn() as c:
        cur = c.execute(f"INSERT INTO paper_trades({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                        [t.get(k) for k in cols])
        return cur.lastrowid


def list_paper(status: str | None = None) -> list[dict]:
    q, args = "SELECT * FROM paper_trades", []
    if status:
        q += " WHERE status=?"
        args.append(status)
    with conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY id DESC", args).fetchall()]


def update_paper(trade_id: int, **kw):
    sets = ", ".join(f"{k}=?" for k in kw)
    with conn() as c:
        c.execute(f"UPDATE paper_trades SET {sets} WHERE id=?", (*kw.values(), trade_id))


def delete_paper(trade_id: int):
    with conn() as c:
        c.execute("DELETE FROM paper_trades WHERE id=?", (trade_id,))


def mark_alert(key: str) -> bool:
    """True if this alert was not sent before (and records it)."""
    with conn() as c:
        try:
            c.execute("INSERT INTO alerts_sent(key, sent_at) VALUES (?, ?)", (key, datetime.now().isoformat()))
            return True
        except sqlite3.IntegrityError:
            return False


def recent_open_picks(days: int = 30, tiers=("strong", "good")) -> list[dict]:
    q = (f"SELECT p.* FROM picks p WHERE (p.outcome IS NULL OR p.outcome='open') AND p.tier IN "
         f"({', '.join('?' * len(tiers))}) AND p.as_of >= date('now', ?)")
    with conn() as c:
        return [dict(r) for r in c.execute(q, (*tiers, f"-{days} days")).fetchall()]
