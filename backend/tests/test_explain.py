"""Why-this-pick, risk badge, expiry and the Yahoo / NSE price cross-check."""
import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd

from app import explain
from app.agents import orchestrator


def test_why_returns_the_models_own_reasons_with_readable_values():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"ret_20": rng.normal(0, 0.1, 2000), "rsi14": rng.uniform(20, 80, 2000),
                      "xr_mom_12_1": rng.uniform(0, 1, 2000)})
    y = (X.ret_20 + 0.02 * rng.normal(size=2000) > 0).astype(int)  # only ret_20 matters
    m = lgb.LGBMClassifier(n_estimators=50, verbose=-1).fit(X, y)
    rows = pd.DataFrame({"ret_20": [0.25], "rsi14": [55.0], "xr_mom_12_1": [0.97]})
    reasons = explain.why(m, list(X.columns), rows)[0]
    assert reasons[0]["feature"] == "ret_20" and reasons[0]["effect"] > 0
    assert reasons[0]["label"] == "20-day return" and reasons[0]["value"] == "+25.0%"
    assert explain._fmt("xr_mom_12_1", 0.97) == "above 97% of stocks"
    assert explain._fmt("xr_atr_pct", 0.01) == "above 1% of stocks"
    assert explain._fmt("atr_pct", 0.018) == "1.8%"
    # contributions add up to the model's own score (log-odds), so the explanation is exact, not a guess
    full = m.booster_.predict(rows, pred_contrib=True)[0]
    assert abs(full.sum() - m.booster_.predict(rows, raw_score=True)[0]) < 1e-6


def test_risk_badge():
    calm = {"stop_pct": 3.8}
    assert explain.risk(calm, None, {})["level"] == "low"
    assert explain.risk({"stop_pct": 6.5}, None, {})["level"] == "medium"
    assert explain.risk({"stop_pct": 11.0}, None, {})["level"] == "high"
    ev = explain.risk(calm, {"purpose": "Financial Results", "date": "2026-11-20"}, {})
    assert ev["level"] == "high" and "Financial Results on 2026-11-20" in ev["reasons"]
    assert explain.risk(calm, None, {"news_sentiment": "negative"})["level"] == "high"


def test_expiry_after_the_entry_session_opens():
    assert explain.next_session("2026-11-13") == "2026-11-16"  # Friday -> Monday
    assert explain.next_session("2026-11-10") == "2026-11-11"
    assert not explain.is_expired("2026-11-16", dt.datetime(2026, 11, 15, 20, 0))
    assert not explain.is_expired("2026-11-16", dt.datetime(2026, 11, 16, 9, 14))
    assert explain.is_expired("2026-11-16", dt.datetime(2026, 11, 16, 9, 15))
    assert not explain.is_expired(None, dt.datetime(2030, 1, 1))  # older picks without the field


def test_sector_cap_flags_the_third_buy_in_a_sector():
    picks = [{"symbol": s, "sector": sec, "tier": t, "score": 99, "ml_prob": pr, "gates": {"agent": True}}
             for s, sec, t, pr in [("A", "IT", "strong", .5), ("B", "IT", "strong", .4), ("C", "IT", "strong", .3),
                                   ("D", "Banks", "strong", .2), ("E", "IT", "watch", .6)]]
    orchestrator.mark_top_pick(picks)
    flags = {p["symbol"]: p["sector_cap"] for p in picks}
    assert flags == {"A": False, "B": False, "C": True, "D": False, "E": False}  # Watch picks are never flagged


def test_price_cross_check_blocks_a_buy_on_mismatch(monkeypatch):
    bhav = pd.DataFrame({"close": [100.0, 250.0]}, index=["TCS", "INFY"])
    assert explain.cross_check("TCS", "2026-11-13", 100.4, bhav)["status"] == "ok"
    bad = explain.cross_check("INFY", "2026-11-13", 256.0, bhav)
    assert bad["status"] == "mismatch" and bad["diff_pct"] == 2.4
    assert explain.cross_check("WIPRO", "2026-11-13", 10.0, bhav)["status"] == "unavailable"
    assert explain.cross_check("TCS", "2026-11-13", 100.0, None)["status"] == "unavailable"

    monkeypatch.setattr(orchestrator, "window_event", lambda s: None)
    row = type("R", (), {"rubric": 80.0, "symbol": "INFY"})()
    good = {"above_ema200": True, "breadth200": 0.62, "adx": 30.0}
    clear = {"news_sentiment": "neutral", "red_flags": [], "event_in_window": False, "parse_error": False}
    assert orchestrator.judge(row, good, dict(clear), 99)["tier"] == "strong"
    j = orchestrator.judge(row, good, dict(clear), 99, data_ok=False)
    assert j["tier"] == "watch" and j["gates"]["data"] is False
    assert "data" not in orchestrator.judge(row, good, dict(clear), 99)["gates"]  # only shown when it fails
