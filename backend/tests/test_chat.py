from app import chat, db

LV = {"entry": 100.0, "target": 120.0, "stop_loss": 90.0}


def _scan(universe, market, picks):
    db.init()
    db.init_extra()
    db.set_settings({"risk_pct": "1"})
    sid = db.create_scan(universe, None, "news")
    db.save_picks(sid, picks)
    db.update_scan(sid, status="done", market=market)


def _pick(symbol, tier, gates):
    return {"symbol": symbol, "name": symbol, "sector": "IT", "levels": LV, "as_of": "2026-03-02", "score": 99,
            "tier": tier, "gates": gates, "research": {"summary": "x"}}


def test_watch_picks_are_never_buy_calls():
    closed = {"above_ema200": True, "breadth200": 0.48}
    _scan("CHATA", closed, [_pick("AAA", "watch", {"ml_top": True, "market": False, "agent": True})])
    r = chat.get_latest_picks("CHATA")
    assert r["market_gate_open"] is False and r["buy_calls"] == [] and r["note"]
    assert r["watch_list"][0]["failed_checks"] == ["market gate closed"]
    assert "Nifty is above" in r["market_gate_reason"] and "48.0%" in r["market_gate_reason"]


def test_buy_call_size_follows_risk_rule():
    _scan("CHATB", {"above_ema200": True, "breadth200": 0.6},
          [_pick("BBB", "strong", {"ml_top": True, "market": True, "agent": True})])
    r = chat.get_latest_picks("CHATB", capital=50000)
    size = r["buy_calls"][0]["size"]
    # 1% of 50,000 = 500 risk; 10 per share at the stop -> 50 shares, far below all-in
    assert r["market_gate_open"] and size["qty"] == 50 and size["max_loss"] == 500 and size["invest"] == 5000


def test_size_is_capped_by_capital_and_rejects_bad_levels():
    assert chat.position_size(LV, 1000, 50)["qty"] == 10  # risk allows 50, capital only 10
    assert chat.position_size({"entry": 100, "target": 120, "stop_loss": 100}, 1000, 1) is None
    assert chat.position_size(LV, 500, 1)["qty"] == 0
