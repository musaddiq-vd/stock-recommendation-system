"""Daily data flow and the day's top pick."""
import datetime as dt

import pandas as pd

from app import main
from app.agents.orchestrator import mark_top_pick
from app.data import prices


def _pick(sym, tier, score, prob, agent=True, market=True):
    return {"symbol": sym, "tier": tier, "score": score, "ml_prob": prob,
            "gates": {"ml_top": score >= 98, "market": market, "agent": agent}}


def test_top_pick_prefers_buy_then_probability():
    picks = [_pick("A", "watch", 99, 0.40, agent=False), _pick("B", "strong", 98, 0.30),
             _pick("C", "strong", 98, 0.35), _pick("D", "avoid", 99, 0.50, agent=False)]
    best = mark_top_pick(picks)
    assert best["symbol"] == "C"
    assert [p["symbol"] for p in picks] == ["C", "B", "A", "D"]
    assert [p["top_pick"] for p in picks] == [True, False, False, False]


def test_top_pick_when_market_gate_closed_is_watch_not_buy():
    picks = [_pick("A", "watch", 99, 0.40, market=False), _pick("B", "watch", 99, 0.45, market=False),
             _pick("C", "watch", 99, 0.60, agent=False, market=False)]
    best = mark_top_pick(picks)
    assert best["symbol"] == "B" and best["tier"] == "watch"  # C ranks higher but the agent flagged it


def test_no_top_pick_when_agent_clears_nothing():
    picks = [_pick("A", "watch", 99, 0.4, agent=False), _pick("B", "avoid", 98, 0.3, agent=False)]
    assert mark_top_pick(picks) is None and not any(p["top_pick"] for p in picks)


def test_expected_last_bar_uses_indian_time():
    fri = dt.datetime(2026, 11, 13)
    assert prices.expected_last_bar(fri.replace(hour=15)) == dt.date(2026, 11, 12)
    assert prices.expected_last_bar(fri.replace(hour=17)) == dt.date(2026, 11, 13)
    assert prices.expected_last_bar(dt.datetime(2026, 11, 15, 12)) == dt.date(2026, 11, 13)  # Sunday


def test_unfinished_intraday_bar_is_not_cached(monkeypatch):
    monkeypatch.setattr(prices, "expected_last_bar", lambda now=None: dt.date(2026, 11, 12))
    idx = pd.to_datetime(["2026-11-11", "2026-11-12", "2026-11-13"])  # 13th is today, market still open
    raw = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": [1.0, 2.0, 3.0], "Volume": 1}, index=idx)
    out = prices._clean(raw)
    assert out.index[-1] == pd.Timestamp("2026-11-12") and len(out) == 2


def test_signal_odds_match_rank_and_market_state(tmp_path, monkeypatch):
    import json

    from app.agents import orchestrator

    (tmp_path / "research").mkdir()
    rows = [{"gate_open": g, "min_pct": lo, "label": lab, "signals": 1000, "win": w}
            for lo, lab, w0 in ((99, "top 1%", 0.55), (98, "top 2%", 0.52), (0, "below top 10%", 0.45))
            for g, w in ((True, w0), (False, w0 - 0.05))]
    (tmp_path / "research" / "odds.json").write_text(json.dumps({"from": "2005-01-03", "to": "2026-10-01", "rows": rows}))
    monkeypatch.setattr(orchestrator, "STORAGE", tmp_path)
    orchestrator._odds.clear()
    assert orchestrator.signal_odds(100, True)["label"] == "top 1%"
    assert round(orchestrator.signal_odds(98, False)["win"], 2) == 0.47
    assert orchestrator.signal_odds(60, True)["label"] == "below top 10%"
    monkeypatch.setattr(orchestrator, "STORAGE", tmp_path / "missing")
    assert orchestrator.signal_odds(99, True) is None  # no research file: the card simply shows no odds


def test_live_record_counts_wins_after_costs():
    from app import db

    db.init()
    base = {"levels": {"entry": 100, "target": 110, "stop_loss": 95}, "as_of": "2026-02-02", "score": 99,
            "tier": "strong", "scoring_version": "cost-v"}
    sid = db.create_scan("NIFTY50", None, "news")
    db.save_picks(sid, [{**base, "symbol": "AAA"}, {**base, "symbol": "BBB"}])
    rets = {"AAA": 0.10, "BBB": 0.002}  # BBB closed 0.2% up: a loss once the 0.3% cost is paid
    for p in db.open_picks():
        if p["symbol"] in rets:
            db.set_outcome(p["id"], "target" if p["symbol"] == "AAA" else "timeout", rets[p["symbol"]], 5)
    t = db.performance("cost-v")["tiers"]["strong"]
    assert t["closed"] == 2 and t["win_rate"] == 0.5
    assert t["avg_ret"] == round((0.10 + 0.002) / 2 - 0.003, 4)


def test_buy_alert_is_honest_about_the_odds():
    from app import alerts

    pick = {"symbol": "TCS", "tier": "strong", "score": 99.0, "hold_days": "15-20", "top_pick": True,
            "levels": {"entry": 100, "target": 110, "stop_loss": 95}, "research": {"summary": "No red flags."},
            "odds": {"win": 0.52, "target": 0.38, "avg_days": 12.1}}
    text = alerts.fmt_pick(pick)
    assert "ML top 1%" in text and "top pick of the day" in text and "hold up to 20 trading days" in text
    assert "52% closed in profit" in text and "38% reached the target" in text
    assert "HIGH PROBABILITY" not in text
    assert "Similar past signals" not in alerts.fmt_pick({**pick, "odds": None})  # older picks have no odds


def test_chat_picks_carry_the_same_odds_as_the_cards(monkeypatch):
    from app import chat

    pick = {"symbol": "TCS", "name": "TCS", "sector": "IT", "tier": "strong", "score": 99.0, "top_pick": True,
            "levels": {"entry": 100, "target": 110, "stop_loss": 95}, "research": {"summary": "ok"},
            "gates": {"ml_top": True, "market": True, "agent": True},
            "odds": {"win": 0.52, "target": 0.38, "avg_days": 12.1, "signals": 13084}}
    scan = {"created_at": "2026-11-14T21:49:27", "universe": "NIFTY500", "picks": [pick],
            "market": {"above_ema200": True, "breadth200": 0.505}}
    monkeypatch.setattr(chat.db, "latest_scan", lambda u=None: scan)
    monkeypatch.setattr(chat.db, "get_settings", lambda: {"capital": "500000", "risk_pct": "1"})
    buy = chat.get_latest_picks()["buy_calls"][0]
    assert buy["odds"] == {"closed_in_profit": 0.52, "reached_target": 0.38, "avg_days_held": 12.1, "past_signals": 13084}
    assert buy["top_pick_of_the_day"] is True


class _Sched:
    running = True

    def __init__(self):
        self.jobs = []

    def add_job(self, *a, **k):
        self.jobs.append(k)


def _daily_env(monkeypatch, bar, scanned, expected="2026-11-13"):
    for name in ("check_data_freshness",):
        monkeypatch.setattr(main, name, lambda: None)
    monkeypatch.setattr(main.nse, "update_recent", lambda n: None)
    monkeypatch.setattr(main.nse, "fii_dii", lambda: {})
    monkeypatch.setattr(main.paper, "update_all", lambda: None)
    monkeypatch.setattr(main, "update_outcomes", lambda: None)
    monkeypatch.setattr(main.sector_data, "warm", lambda: None)
    monkeypatch.setattr(main, "latest_bar", lambda: bar)
    monkeypatch.setattr(main, "last_scanned_bar", lambda: scanned)
    monkeypatch.setattr(main, "expected_last_bar", lambda: dt.date.fromisoformat(expected))
    sched, started = _Sched(), []
    monkeypatch.setattr(main, "scheduler", sched)
    monkeypatch.setattr(main, "start_scan", lambda *a, **k: started.append(a))
    return sched, started


def test_daily_job_waits_for_todays_bars(monkeypatch):
    sched, started = _daily_env(monkeypatch, bar="2026-11-12", scanned="2026-11-12")
    main.daily_job()
    assert started == [] and sched.jobs[0]["args"] == [1]  # retry later, do not rescan yesterday


def test_daily_job_scans_new_session(monkeypatch):
    sched, started = _daily_env(monkeypatch, bar="2026-11-13", scanned="2026-11-12")
    main.daily_job()
    assert started == [("NIFTY500", None)] and sched.jobs == []


def test_daily_job_skips_after_retries_on_a_holiday(monkeypatch):
    sched, started = _daily_env(monkeypatch, bar="2026-11-12", scanned="2026-11-12")
    main.daily_job(attempt=main.DATA_RETRIES)
    assert started == [] and sched.jobs == []
