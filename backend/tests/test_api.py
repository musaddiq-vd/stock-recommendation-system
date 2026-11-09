import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import auth, db, main, paper
from app.agents import orchestrator

UNI = pd.DataFrame({"symbol": ["RELIANCE", "TCS"], "name": ["Reliance", "TCS"], "sector": ["Energy", "IT"],
                    "indices": [["NIFTY500"], ["NIFTY500"]]})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "get_universe", lambda: UNI)
    monkeypatch.setattr(main, "load", lambda symbol: None)  # no network in tests
    monkeypatch.setattr(auth, "APP_PASSWORD", "")
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def locked(client, monkeypatch):
    monkeypatch.setattr(auth, "APP_PASSWORD", "s3cret-pass")
    monkeypatch.setattr(auth, "APP_SECRET", "x" * 64)
    auth._attempts.clear()
    return client


def test_health_is_public(locked):
    r = locked.get("/api/health")
    assert r.status_code == 200 and "model_ready" in r.json()


def test_api_requires_login(locked):
    assert locked.get("/api/settings").status_code == 401
    assert locked.get("/api/settings", headers={"Authorization": "Bearer forged"}).status_code == 401
    tok = locked.post("/api/auth/login", json={"password": "s3cret-pass"}).json()["token"]
    assert locked.get("/api/settings", headers={"Authorization": f"Bearer {tok}"}).status_code == 200


def test_login_rate_limit(locked):
    for _ in range(auth.MAX_ATTEMPTS):
        assert locked.post("/api/auth/login", json={"password": "wrong"}).status_code == 401
    assert locked.post("/api/auth/login", json={"password": "s3cret-pass"}).status_code == 429


def test_expired_token_rejected(monkeypatch):
    monkeypatch.setattr(auth, "APP_SECRET", "y" * 64)
    monkeypatch.setattr(auth, "TOKEN_TTL", -10)
    assert not auth.verify_token(auth.make_token())


def test_unknown_or_unsafe_symbol_is_404(client):
    for s in ("ZZZZ", "..%5C..%5Cetc", "REL$"):
        assert client.get(f"/api/stock/{s}").status_code == 404


def test_chat_must_end_with_user(client):
    r = client.post("/api/chat", json={"messages": [{"role": "assistant", "content": "hi"}]})
    assert r.status_code == 400
    r = client.post("/api/chat", json={"messages": [{"role": "system", "content": "x"}]})
    assert r.status_code == 422


def test_settings_validation(client):
    assert client.put("/api/settings", json={"risk_pct": 50}).status_code == 422
    assert client.put("/api/settings", json={"capital": -1}).status_code == 422
    assert client.put("/api/settings", json={"telegram_bot_token": "not a token"}).status_code == 422
    r = client.put("/api/settings", json={"capital": 250000, "risk_pct": 0.5})
    assert r.status_code == 200 and r.json()["capital"] == "250000.0"
    assert "telegram_bot_token" not in r.json()  # the token is never sent back


def test_scan_busy_returns_409(client, monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", type("P", (), {"exists": lambda self: True})())

    def busy(*a, **k):
        raise orchestrator.ScanBusy()

    monkeypatch.setattr(main, "start_scan", busy)
    assert client.post("/api/scan", json={"universe": "nifty50"}).status_code == 409
    assert client.post("/api/scan", json={"universe": "NIFTY999"}).status_code == 422
    assert client.post("/api/scan", json={"sector": "Moon Mining"}).status_code == 400


def test_restart_closes_running_scans():
    sid = db.create_scan("NIFTY50", None, "news")
    db.init()
    assert db.get_scan(sid)["status"] == "error"


def test_paper_trade_opens_at_current_price(monkeypatch):
    db.init_extra()
    bars = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1.0]},
                        index=[pd.Timestamp("2026-11-06")])
    monkeypatch.setattr(paper, "load_many", lambda syms: {"TCS": bars})
    monkeypatch.setattr(paper, "live_quotes", lambda syms: {"TCS": {"last": 104.0, "date": "2026-11-09"}})
    t = paper.open_trade("TCS", 10, target=110, stop_loss=95)
    assert t["entry"] == 104.0 and t["opened_on"] == "2026-11-09"
    with pytest.raises(paper.TradeRejected):  # price already above the target
        paper.open_trade("TCS", 10, target=103, stop_loss=95)


def test_tier_rules():
    ok = {"ml_top": True, "market": True, "agent": True}
    assert orchestrator.tier("buy", ok) == "strong"
    for gate in ok:  # every gate is required: the Buy tier maps to exactly one backtested variant
        assert orchestrator.tier("buy", {**ok, gate: False}) == "watch"
    assert orchestrator.tier("watch", ok) == "watch"
    assert orchestrator.tier("avoid", ok) == "avoid"


def test_settings_have_no_short_selling_fields(client):
    assert "margin_pct" not in client.get("/api/settings").json()


def test_track_record_counts_each_trade_once():
    db.init()
    pick = {"symbol": "TCS", "levels": {"entry": 100, "target": 110, "stop_loss": 95}, "as_of": "2026-01-05",
            "score": 90, "tier": "strong", "scoring_version": "test-v"}
    old_rules = {**pick, "symbol": "INFY", "scoring_version": "old"}
    for _ in range(2):  # the same recommendation from two scans
        sid = db.create_scan("NIFTY50", None, "news")
        db.save_picks(sid, [pick, old_rules])
    for p in db.open_picks():
        db.set_outcome(p["id"], "target", 0.1, 5)
    perf = db.performance("test-v")
    assert perf["tiers"]["strong"]["closed"] == 1 and perf["tiers"]["strong"]["win_rate"] == 1.0
    assert len([r for r in perf["recent"] if r["symbol"] == "TCS"]) == 1
    assert not any(r["symbol"] == "INFY" for r in perf["recent"])  # other rule sets never mix in
    assert perf["earlier_version_picks"] >= 1


def test_agent_veto_check_splits_cleared_and_vetoed():
    db.init()
    base = {"levels": {"entry": 100, "target": 110, "stop_loss": 95}, "as_of": "2026-02-02", "score": 80,
            "tier": "strong", "scoring_version": "veto-v"}
    picks = [{**base, "symbol": "AAA", "gates": {"ml_top": True, "market": True, "agent": True}},
             {**base, "symbol": "BBB", "gates": {"ml_top": True, "market": True, "agent": False}, "tier": "watch"},
             {**base, "symbol": "CCC", "gates": {"ml_top": True, "market": False, "agent": False}, "tier": "watch"}]
    sid = db.create_scan("NIFTY50", None, "news")
    db.save_picks(sid, picks)
    for p in db.open_picks():
        if p["symbol"] in ("AAA", "BBB", "CCC"):
            db.set_outcome(p["id"], "target" if p["symbol"] == "AAA" else "stop", 0.1 if p["symbol"] == "AAA" else -0.05, 3)
    chk = db.performance("veto-v")["agent_check"]
    assert chk["cleared"]["closed"] == 1 and chk["cleared"]["win_rate"] == 1.0
    assert chk["vetoed"]["closed"] == 1  # CCC failed the market gates, so it is not part of the comparison


def test_outcome_method_migration_resets_old_outcomes():
    db.init()
    db.init_extra()
    sid = db.create_scan("NIFTY50", None, "news")
    db.save_picks(sid, [{"symbol": "ZZM", "levels": {"entry": 1, "target": 2, "stop_loss": 0.5}, "as_of": "2026-01-01",
                         "score": 1, "tier": "watch"}])
    pid = [p for p in db.open_picks() if p["symbol"] == "ZZM"][0]["id"]
    db.set_outcome(pid, "target", 0.5, 2)
    db.set_settings({})  # no-op
    with db.conn() as c:
        c.execute("UPDATE settings SET value='old' WHERE key='outcome_method'")
    db.init_extra()
    assert any(p["id"] == pid for p in db.open_picks())  # recomputed under the new method
    assert "outcome_method" not in db.get_settings()


def test_weekly_summary_text():
    from app import alerts

    scan = {"id": 7, "market": {"above_ema200": True, "breadth200": 0.62},
            "picks": [{"symbol": "TCS", "tier": "strong"}, {"symbol": "INFY", "tier": "watch"}]}
    perf = {"version": "7", "tiers": {"strong": {"closed": 4, "open": 2, "win_rate": 0.5, "avg_ret": 0.012}}}
    text = alerts.weekly_summary(scan, perf, {"trades": [], "equity": 0, "return_pct": 0, "open": 0})
    assert "Market gate: <b>open</b>" in text and "TCS" in text and "INFY" not in text
    assert "50% profitable" in text and "+1.2%" in text


def test_stale_data_raises_one_ops_alert(monkeypatch):
    import datetime as dt

    sent = []
    monkeypatch.setattr(main.alerts, "ops_alert", lambda key, text: sent.append(key))
    bars = pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp("2026-10-01")])
    monkeypatch.setattr(main, "load", lambda s: bars)
    monkeypatch.setattr(main, "expected_last_bar", lambda: dt.date(2026, 11, 6))
    main.check_data_freshness()
    assert sent == ["stale:2026-11-06"]
    monkeypatch.setattr(main, "expected_last_bar", lambda: dt.date(2026, 10, 2))
    sent.clear()
    main.check_data_freshness()
    assert sent == []  # one session behind is normal
