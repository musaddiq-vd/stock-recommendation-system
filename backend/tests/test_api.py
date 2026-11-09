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
    ok = {"regime": True, "rubric": True, "agent": True}
    assert orchestrator.tier(90, "buy", ok) == "strong"
    assert orchestrator.tier(90, "buy", {**ok, "rubric": False}) == "good"
    assert orchestrator.tier(90, "buy", {**ok, "regime": False}) == "watch"  # no regime, no tier backed by data
    assert orchestrator.tier(80, "buy", ok) == "good"
    assert orchestrator.tier(90, "buy", {"regime": False, "rubric": False, "agent": True}) == "watch"
    assert orchestrator.tier(95, "watch", ok) == "watch"
    assert orchestrator.tier(95, "avoid", ok) == "avoid"


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
