"""Paper trading: virtual positions from picks, auto-closed at target / stop / 20-day timeout."""
from datetime import datetime

import pandas as pd

from . import alerts, db
from .config import HOLD_MAX_DAYS
from .data.live import live_quotes
from .data.prices import load_many

EMOJI = {"target": "🎯", "stop": "🛑", "timeout": "⏱", "closed": "✋"}


def _exit(t, status, price, date):
    sgn = 1 if t["side"] == "long" else -1
    pnl = round(sgn * (price - t["entry"]) * t["qty"], 2)
    db.update_paper(t["id"], status=status, exit_price=round(price, 2), exit_date=str(date), pnl=pnl,
                    last_price=round(price, 2))
    alerts.send_once(f"paper:{t['id']}:{status}",
                     f"{EMOJI[status]} <b>Paper trade {status.upper()}</b>: {t['side'].upper()} {t['symbol']} × {t['qty']}\n"
                     f"Entry ₹{t['entry']} → Exit ₹{price:.2f} · P&L <b>₹{pnl:,.0f}</b>")
    return pnl


def check_bar(t, high, low, open_: float | None = None):
    """('stop'|'target', fill price) if this bar touches a level. A gap through a level fills at the open
    (worse than the stop, better than the target). Stop first when both are touched (conservative)."""
    stop, target = t["stop_loss"], t["target"]
    if open_ is not None and open_ <= stop:
        return "stop", open_
    if low <= stop:
        return "stop", stop
    if open_ is not None and open_ >= target:
        return "target", open_
    if high >= target:
        return "target", target
    return None


def update_all(use_live: bool = True) -> None:
    trades = db.list_paper("open")
    if not trades:
        return
    syms = sorted({t["symbol"] for t in trades})
    daily = load_many(syms)
    live = live_quotes(syms) if use_live else {}
    now = datetime.now().isoformat(timespec="seconds")
    for t in trades:
        df = daily.get(t["symbol"])
        bars = df[df.index > pd.Timestamp(t["opened_on"])] if df is not None else pd.DataFrame()
        closed = False
        for i, (d, r) in enumerate(bars.iterrows(), 1):
            hit = check_bar(t, r.high, r.low, r.open)
            if hit:
                _exit(t, hit[0], hit[1], d.date())
                closed = True
                break
            if i >= HOLD_MAX_DAYS:
                _exit(t, "timeout", float(r.close), d.date())
                closed = True
                break
        if closed:
            continue
        q = live.get(t["symbol"])
        if q and q["date"] > t["opened_on"]:
            hit = check_bar(t, q["high"], q["low"])
            if hit:
                _exit(t, hit[0], hit[1], q["date"])
                continue
        last = q["last"] if q else (float(bars.close.iloc[-1]) if len(bars) else t["entry"])
        db.update_paper(t["id"], last_price=round(last, 2), last_checked=now)


def close(trade_id: int) -> None:
    t = next((x for x in db.list_paper("open") if x["id"] == trade_id), None)
    if not t:
        return
    q = live_quotes([t["symbol"]]).get(t["symbol"])
    price = q["last"] if q else (t["last_price"] or t["entry"])
    _exit(t, "closed", price, datetime.now().date())


def portfolio() -> dict:
    capital = float(db.get_settings()["capital"])
    trades = db.list_paper()
    realized = sum(t["pnl"] or 0 for t in trades if t["status"] != "open")
    unreal = 0.0
    for t in trades:
        if t["status"] == "open":
            sgn = 1 if t["side"] == "long" else -1
            lp = t["last_price"] or t["entry"]
            t["unrealized"] = round(sgn * (lp - t["entry"]) * t["qty"], 2)
            t["unrealized_pct"] = round(sgn * (lp / t["entry"] - 1) * 100, 2)
            unreal += t["unrealized"]
    closed = [t for t in trades if t["status"] != "open"]
    wins = [t for t in closed if (t["pnl"] or 0) > 0]
    return {
        "capital": capital, "realized": round(realized, 2), "unrealized": round(unreal, 2),
        "equity": round(capital + realized + unreal, 2),
        "return_pct": round((realized + unreal) / capital * 100, 2) if capital else 0.0,
        "open": sum(t["status"] == "open" for t in trades), "closed": len(closed),
        "win_rate": round(len(wins) / len(closed), 3) if closed else None,
        "trades": trades,
    }


class TradeRejected(ValueError):
    pass


def open_trade(symbol: str, qty: int, target: float, stop_loss: float, scan_id: int | None = None) -> dict:
    """Open at the current price, not the scan's close: a trade opened days later must not inherit moves
    that already happened. Rejected if the price has already passed the target or the stop."""
    daily = load_many([symbol]).get(symbol)
    if daily is None or daily.empty:
        raise TradeRejected(f"No price data for {symbol}")
    price, on = float(daily.close.iloc[-1]), str(daily.index[-1].date())
    q = live_quotes([symbol]).get(symbol)
    if q and q["date"] >= on:
        price, on = q["last"], q["date"]
    if not (stop_loss < price < target):
        raise TradeRejected(f"Price ₹{price:.2f} is already outside stop ₹{stop_loss} / target ₹{target}")
    tid = db.add_paper({"symbol": symbol, "side": "long", "qty": qty, "lots": None, "entry": round(price, 2),
                        "target": target, "stop_loss": stop_loss, "opened_on": on, "scan_id": scan_id})
    return {"id": tid, "entry": round(price, 2), "opened_on": on}
