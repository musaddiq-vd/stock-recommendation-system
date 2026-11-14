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
    summary = (p.get("research", {}).get("summary") or "")[:220]
    odds = p.get("odds") or {}
    # the same honest numbers as the dashboard card: a Buy is about a coin flip with a better payoff, not a sure thing
    chance = (f"Similar past signals: {odds['win'] * 100:.0f}% closed in profit, {odds['target'] * 100:.0f}% reached "
              f"the target (avg {odds['avg_days']:.0f} days)\n") if odds.get("win") is not None else ""
    top = " · top pick of the day" if p.get("top_pick") else ""
    return (f"<b>🟢 BUY {p['symbol']}</b> (ML top {max(1, 100 - int(p['score']))}%{top})\n"
            f"Entry ₹{L['entry']} → Target ₹{L['target']} · Stop ₹{L['stop_loss']} · "
            f"hold up to {str(p['hold_days']).split('-')[-1]} trading days\n"
            f"{chance}<i>{summary}</i>")


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


def ops_alert(key: str, text: str) -> bool:
    """Operational problem (failed scan, stale data). Sent once per key so a recurring issue does not spam."""
    return send_once(f"ops:{key}", f"⚠️ <b>AI Swing Trader</b>: {text}")


def weekly_summary(scan: dict | None, perf: dict, paper: dict) -> str:
    """Plain weekly report: market gate, the latest Buy calls, the live track record and paper P&L."""
    lines = ["🗓 <b>Weekly summary</b>"]
    if scan:
        m = scan.get("market") or {}
        gate = bool(m.get("above_ema200")) and (m.get("breadth200") or 0) > 0.5
        lines.append(f"Market gate: <b>{'open' if gate else 'closed'}</b> (Nifty "
                     f"{'above' if m.get('above_ema200') else 'below'} 200-day avg, breadth {round((m.get('breadth200') or 0) * 100)}%)")
        buys = [p["symbol"] for p in scan.get("picks", []) if p.get("tier") == "strong"]
        lines.append(f"Latest scan #{scan['id']}: {', '.join(buys) if buys else 'no Buy calls'}")
    t = (perf.get("tiers") or {}).get("strong")
    if t:
        rec = (f"{t['closed']} closed, {t['open']} open"
               + (f" · {round(t['win_rate'] * 100)}% profitable · avg {t['avg_ret'] * 100:+.1f}%" if t["closed"] else ""))
        lines.append(f"Buy track record (rules v{perf.get('version')}): {rec}")
    if paper.get("trades"):
        lines.append(f"Paper portfolio: ₹{paper['equity']:,.0f} ({paper['return_pct']:+.2f}%), {paper['open']} open")
    lines.append("<i>Educational only, not financial advice.</i>")
    return "\n".join(lines)
