import pandas as pd

from app.agents import orchestrator
from app.paper import check_bar


def test_check_bar_long():
    t = {"side": "long", "target": 110, "stop_loss": 95}
    assert check_bar(t, 111, 100) == ("target", 110)
    assert check_bar(t, 111, 94) == ("stop", 95)  # both touched: stop wins
    assert check_bar(t, 105, 99) is None
    assert check_bar(t, 92, 88, open_=90) == ("stop", 90)  # gap below the stop fills at the open
    assert check_bar(t, 115, 111, open_=112) == ("target", 112)


def _bars(rows, start="2026-10-01"):
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


PICK = {"as_of": "2026-09-30", "entry": 100.0, "target": 110.0, "stop_loss": 95.0}


def test_evaluate_target_stop_timeout_open():
    hit = _bars([[100, 104, 99, 103, 1], [103, 111, 102, 110, 1]])
    assert orchestrator.evaluate(PICK, hit, None)[:1] == ("target",)
    stop = _bars([[100, 101, 94, 96, 1]])
    out, ret, days = orchestrator.evaluate(PICK, stop, None)
    assert out == "stop" and round(ret, 3) == -0.05 and days == 1
    flat = _bars([[100, 102, 98, 101, 1]] * 20)
    out, ret, days = orchestrator.evaluate(PICK, flat, None)
    assert out == "timeout" and round(ret, 3) == 0.01 and days == 20
    out, ret, _ = orchestrator.evaluate(PICK, _bars([[100, 102, 98, 101, 1]] * 3), None)
    assert out == "open" and round(ret, 3) == 0.01


def test_evaluate_ignores_bars_before_recommendation():
    # a bar on the recommendation day itself must not count
    bars = _bars([[100, 120, 99, 100, 1], [100, 101, 99, 100, 1]], start="2026-09-30")
    assert orchestrator.evaluate(PICK, bars, None)[0] == "open"


def test_evaluate_entry_at_next_open_with_gaps():
    # bought at the 102 open, then gapped down to 90 through the 95 stop: loss is open-to-open
    bars = _bars([[102, 103, 100, 101, 1], [90, 91, 89, 90, 1]])
    out, ret, days = orchestrator.evaluate(PICK, bars, None)
    assert out == "stop" and round(ret, 4) == round(90 / 102 - 1, 4) and days == 2
    # opened far above the published target: still bought, with target and stop measured from the fill
    assert orchestrator.evaluate(PICK, _bars([[112, 113, 111, 112, 1]]), None)[0] == "open"


def test_levels_move_with_the_fill_like_the_backtest():
    # published: ref 100, target 110 (+10), stop 95 (-5). Bought at 104 -> target 114, stop 99.
    assert orchestrator.fill_levels(100.0, 110.0, 95.0, 104.0) == (99.0, 114.0)
    up = _bars([[104, 111, 103, 110, 1], [110, 115, 109, 114, 1]])  # 111 is above the old target, below the new one
    out, ret, days = orchestrator.evaluate(PICK, up, None)
    assert out == "target" and days == 2 and round(ret, 4) == round(114 / 104 - 1, 4)
    down = _bars([[104, 105, 98.5, 100, 1]])  # 98.5 never reached the old 95 stop, but breaks the new 99 one
    out, ret, days = orchestrator.evaluate(PICK, down, None)
    assert out == "stop" and round(ret, 4) == round(99 / 104 - 1, 4)


def test_evaluate_uses_live_bar_for_today():
    bars = _bars([[100, 102, 98, 101, 1]])
    quote = {"date": "2026-10-02", "high": 112, "low": 100}
    assert orchestrator.evaluate(PICK, bars, quote)[0] == "target"


def test_judge_gates_and_veto(monkeypatch):
    monkeypatch.setattr(orchestrator, "window_event", lambda s: None)
    row = type("R", (), {"rubric": 80.0, "symbol": "TCS"})()
    bull = {"regime": "bullish", "above_ema200": True, "breadth200": 0.62}
    clear = {"news_sentiment": "positive", "red_flags": [], "event_in_window": False, "parse_error": False}
    j = orchestrator.judge(row, bull, dict(clear), 99)
    assert j["tier"] == "strong" and all(j["gates"].values())
    # negative news or a failed agent run never passes the agent gate
    assert orchestrator.judge(row, bull, {**clear, "news_sentiment": "negative"}, 99)["tier"] == "watch"
    assert orchestrator.judge(row, bull, {**clear, "parse_error": True}, 99)["tier"] == "watch"
    assert orchestrator.judge(row, bull, {**clear, "red_flags": ["SEBI order"]}, 99)["tier"] == "avoid"
    # official results inside the window override the agent
    monkeypatch.setattr(orchestrator, "window_event", lambda s: {"purpose": "Financial Results", "date": "2026-11-12"})
    j = orchestrator.judge(row, bull, dict(clear), 99)
    assert j["gates"]["agent"] is False and j["tier"] == "watch"


def test_judge_needs_top_rank_and_breadth(monkeypatch):
    monkeypatch.setattr(orchestrator, "window_event", lambda s: None)
    row = type("R", (), {"rubric": 80.0, "symbol": "TCS"})()
    clear = {"news_sentiment": "neutral", "red_flags": [], "event_in_window": False, "parse_error": False}
    good = {"above_ema200": True, "breadth200": 0.62}
    assert orchestrator.judge(row, good, dict(clear), 97)["tier"] == "watch"  # not top 2%
    assert orchestrator.judge(row, {**good, "breadth200": 0.41}, dict(clear), 99)["tier"] == "watch"  # weak breadth
    assert orchestrator.judge(row, {**good, "above_ema200": False}, dict(clear), 99)["tier"] == "watch"  # Nifty weak
    assert orchestrator.judge(row, good, dict(clear), 99)["tier"] == "strong"

