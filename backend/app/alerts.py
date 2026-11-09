"""Telegram alerts (free Bot API). WhatsApp needs a paid Business API, so Telegram is the default."""
import logging

import requests

from . import db

log = logging.getLogger(__name__)


def _creds():
    s = db.get_settings()
    return s.get("telegram_bot_token", "").strip(), s.get("telegram_chat_id", "").strip(), s.get("alerts_enabled") == "1"


def send(text: str, force: bool = False) -> bool:
    token, chat, enabled = _creds()
    if not token or not chat or (not enabled and not force):
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=20)
        if not r.ok:
            log.warning("telegram send failed: %s", r.text[:200])
        return r.ok
    except Exception as e:
        log.warning("telegram send failed: %s", e)
        return False


def send_once(key: str, text: str) -> bool:
    """Send only the first time for a given key, e.g. paper:12:target."""
    token, chat, enabled = _creds()
    if not (token and chat and enabled):
        return False
    if db.mark_alert(key):
        return send(text)
    return False


def find_chat_id(token: str) -> str | None:
    """After the user sends /start to their bot, read the chat id from getUpdates."""
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20).json()
    for u in reversed(r.get("result", [])):
        msg = u.get("message") or u.get("channel_post") or {}
        if msg.get("chat", {}).get("id"):
            return str(msg["chat"]["id"])
    return None


def fmt_pick(p: dict) -> str:
    L = p["levels"]
    tier = "🔥 HIGH PROBABILITY" if p["tier"] == "strong" else "✅ Good"
    summary = (p.get("research", {}).get("summary") or "")[:220]
    return (f"<b>🟢 BUY {p['symbol']}</b> ({tier}, score {p['score']:.0f})\n"
            f"Entry ₹{L['entry']} → Target ₹{L['target']} · Stop ₹{L['stop_loss']} · hold {p['hold_days']}d\n"
            f"<i>{summary}</i>")


def scan_summary(scan: dict) -> None:
    good = [p for p in scan["picks"] if p["tier"] in ("strong", "good")]
    if not good:
        return
    head = f"📊 <b>AI Swing scan #{scan['id']}</b> ({scan['universe']}, market {scan['market']['regime']})\n\n"
    body = "\n\n".join(fmt_pick(p) for p in good[:8])
    send_once(f"scan:{scan['id']}", head + body + "\n\n<i>Educational only, not financial advice.</i>")


def pick_outcome(pick_row: dict, outcome: str, ret: float | None) -> None:
    """Alert when a strong/good recommendation hits target or stop."""
    if outcome not in ("target", "stop") or pick_row.get("tier") not in ("strong", "good"):
        return
    emoji = "🎯" if outcome == "target" else "🛑"
    pct = f" ({ret * 100:+.1f}%)" if ret is not None else ""
    send_once(f"pick:{pick_row['symbol']}:{pick_row['as_of']}:{outcome}",
              f"{emoji} <b>{pick_row['symbol']}</b> recommendation hit <b>{outcome.upper()}</b>{pct}\n"
              f"Entry ₹{pick_row['entry']} · Target ₹{pick_row['target']} · Stop ₹{pick_row['stop_loss']}")
