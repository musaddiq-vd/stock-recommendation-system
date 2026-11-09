import json
import logging
import re
import threading
import time
import warnings
from contextlib import asynccontextmanager
from typing import Literal

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .config import (ALLOWED_ORIGINS, BENCHMARK, LOG_LEVEL, MODEL_PATH, RUN_SCHEDULER,  # noqa: I001 (loads .env first)
                     SCORING_VERSION, STORAGE)
from . import alerts, auth, db, paper
from .agents.orchestrator import ScanBusy, market_regime, scan_running, start_scan, update_outcomes
from .agents.tools import get_fundamentals, get_technicals
from .chat import chat_stream
from .data import nse
from .data.prices import expected_last_bar, load
from .model.train import load_model, train_all
from .universe import get_universe

warnings.filterwarnings("ignore")
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)
scheduler = BackgroundScheduler(timezone="Asia/Kolkata", job_defaults={"coalesce": True, "max_instances": 1})
SYMBOL_RE = re.compile(r"^[A-Z0-9&\-]{1,20}$")
UNIVERSES = ("NIFTY50", "NIFTY100", "NIFTY500")


# ---------------------------------------------------------------- scheduled jobs
def daily_job():
    for step in (lambda: nse.update_recent(7), nse.fii_dii, paper.update_all, update_outcomes):
        try:
            step()
        except Exception:
            log.exception("daily step failed")
    check_data_freshness()
    try:
        start_scan("NIFTY500", None)
    except ScanBusy:
        log.warning("daily scan skipped: another scan is running")


def check_data_freshness() -> None:
    """Alert when market data stops updating (Yahoo / NSE outage), since every pick would then be stale."""
    try:
        bars = load(BENCHMARK)
        last = bars.index[-1].date() if bars is not None else None
        expected = expected_last_bar()
        if last is None or (expected - last).days > 3:  # allows for weekends and single holidays
            alerts.ops_alert(f"stale:{expected}", f"market data is stale (last bar {last}, expected {expected}). "
                                                  "Picks are not reliable until it updates.")
    except Exception:
        log.exception("freshness check failed")


def weekly_job():
    try:
        alerts.send(alerts.weekly_summary(db.latest_scan(), db.performance(SCORING_VERSION),
                                          paper.portfolio()))
    except Exception:
        log.exception("weekly summary failed")


def intraday_job():
    """Every 15 min in market hours: check paper trades and open picks against live prices."""
    try:
        paper.update_all(use_live=True)
        update_outcomes(use_live=True)
    except Exception:
        log.exception("intraday check failed")


@asynccontextmanager
async def lifespan(app):
    db.init()
    db.init_extra()
    if RUN_SCHEDULER:
        # Mon-Fri after NSE close (data on Yahoo settles by ~18:00 IST)
        scheduler.add_job(daily_job, "cron", day_of_week="mon-fri", hour=18, minute=30, id="daily",
                          replace_existing=True)
        scheduler.add_job(intraday_job, "cron", day_of_week="mon-fri", hour="9-15", minute="*/15", id="intraday",
                          replace_existing=True)
        # Markets change: retrain both models on fresh data on the 1st Saturday of every month
        scheduler.add_job(weekly_job, "cron", day_of_week="sat", hour=9, id="weekly", replace_existing=True)
        scheduler.add_job(train_all, "cron", day="1-7", day_of_week="sat", hour=10, id="retrain",
                          replace_existing=True)
        scheduler.start()
        log.info("scheduler started")
    yield
    if scheduler.running:
        scheduler.shutdown(wait=False)


app = FastAPI(title="AI Swing Trader (NSE)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"],
                   allow_headers=["Authorization", "Content-Type"])

PUBLIC_PATHS = {"/api/health", "/api/auth/status", "/api/auth/login", "/docs", "/openapi.json"}


@app.middleware("http")
async def require_login(request: Request, call_next):
    path = request.url.path
    if auth.enabled() and request.method != "OPTIONS" and path.startswith("/api") and path not in PUBLIC_PATHS:
        header = request.headers.get("authorization", "")
        if not (header.startswith("Bearer ") and auth.verify_token(header[7:])):
            return JSONResponse({"detail": "Login required"}, status_code=401)
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    # log the details server-side; never leak tracebacks to the client
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


def _symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    if not SYMBOL_RE.match(s) or s not in set(get_universe().symbol):
        raise HTTPException(404, "Unknown NSE symbol")
    return s


# ---------------------------------------------------------------- auth + health
class LoginReq(BaseModel):
    password: str = Field(min_length=1, max_length=200)


@app.get("/api/auth/status")
def auth_status():
    return {"auth_required": auth.enabled()}


@app.post("/api/auth/login")
def login(req: LoginReq, request: Request):
    client = request.client.host if request.client else "unknown"
    if not auth.enabled():
        return {"token": "", "auth_required": False}
    if auth.rate_limited(client):
        raise HTTPException(429, "Too many attempts. Try again in 15 minutes.")
    if not auth.check_password(client, req.password):
        raise HTTPException(401, "Wrong password")
    return {"token": auth.make_token(), "auth_required": True}


@app.get("/api/health")
def health():
    """Liveness plus the data the app depends on, so a monitor can tell 'up' from 'useful'."""
    out = {"ok": True, "model_ready": MODEL_PATH.exists(), "scan_running": scan_running(),
           "scheduler": scheduler.running}
    try:
        bench = load(BENCHMARK)
        out["prices_as_of"] = str(bench.index[-1].date()) if bench is not None else None
        last = db.latest_scan()
        out["last_scan"] = {"id": last["id"], "at": last["created_at"]} if last else None
    except Exception:
        out["ok"] = False
    return out


# ---------------------------------------------------------------- market + scans
class ScanReq(BaseModel):
    universe: Literal["NIFTY50", "NIFTY100", "NIFTY500"] = "NIFTY100"
    sector: str | None = Field(default=None, max_length=80)
    top_n: int = Field(default=15, ge=3, le=40)

    @field_validator("universe", mode="before")
    @classmethod
    def upper(cls, v):
        return str(v).upper()


@app.get("/api/universe")
def universe():
    u = get_universe()
    return {"stocks": u.to_dict("records"), "sectors": sorted(u.sector.dropna().unique().tolist())}


@app.get("/api/market")
def market():
    m = market_regime()
    try:
        m["flows"] = nse.fii_dii()
    except Exception:
        m["flows"] = {}
    return m


@app.post("/api/scan")
def scan(req: ScanReq):
    if req.sector and req.sector not in set(get_universe().sector.dropna()):
        raise HTTPException(400, "Unknown sector")
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not trained yet. Run: python -m app.model.train")
    try:
        return {"scan_id": start_scan(req.universe, req.sector or None, req.top_n)}
    except ScanBusy:
        raise HTTPException(409, "A scan is already running. Wait for it to finish.")


@app.get("/api/scans")
def scans():
    return db.list_scans()


@app.get("/api/scans/latest")
def latest(universe: str | None = None, include_running: bool = False):
    if universe and universe.upper() not in UNIVERSES:
        raise HTTPException(400, "universe must be NIFTY50, NIFTY100 or NIFTY500")
    return db.latest_scan(universe.upper() if universe else None, done_only=not include_running) or {}


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: int):
    s = db.get_scan(scan_id)
    if not s:
        raise HTTPException(404)
    return s


@app.get("/api/stock/{symbol}")
def stock(symbol: str):
    symbol = _symbol(symbol)
    df = load(symbol)
    if df is None:
        raise HTTPException(404, "No data")
    tail = df.tail(250)
    candles = [{"time": d.strftime("%Y-%m-%d"), "open": round(r.open, 2), "high": round(r.high, 2),
                "low": round(r.low, 2), "close": round(r.close, 2), "volume": int(r.volume)}
               for d, r in tail.iterrows()]
    try:
        fund = get_fundamentals(symbol)
    except Exception:
        fund = {}
    return {"symbol": symbol, "candles": candles, "technicals": get_technicals(symbol), "fundamentals": fund}


@app.get("/api/model")
def model_report():
    try:
        r = dict(load_model()["report"])
    except FileNotFoundError:
        raise HTTPException(404, "Model not trained yet")
    r.pop("prob_percentiles", None)
    return r


_outcomes = {"ts": 0.0, "lock": threading.Lock()}
OUTCOME_REFRESH_SECONDS = 30 * 60


def _refresh_outcomes_in_background() -> None:
    """Outcomes need price data for every open pick (slow), so a request never waits for them:
    it gets the stored numbers and, if they are older than 30 minutes, a refresh starts in the background."""
    if time.time() - _outcomes["ts"] < OUTCOME_REFRESH_SECONDS or not _outcomes["lock"].acquire(blocking=False):
        return

    def run():
        try:
            update_outcomes()
            _outcomes["ts"] = time.time()
        except Exception:
            log.exception("outcome refresh failed")
        finally:
            _outcomes["lock"].release()

    threading.Thread(target=run, daemon=True, name="outcomes").start()


@app.get("/api/performance")
def performance():
    _refresh_outcomes_in_background()
    return db.performance(SCORING_VERSION)


# ---------------------------------------------------------------- chat
class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatReq(BaseModel):
    messages: list[ChatMsg] = Field(min_length=1, max_length=60)


@app.post("/api/chat")
def chat(req: ChatReq):
    if req.messages[-1].role != "user" or not req.messages[-1].content.strip():
        raise HTTPException(400, "Type a question first")
    msgs = [m.model_dump() for m in req.messages if m.content.strip()][-20:]
    while msgs and msgs[0]["role"] != "user":  # Bedrock needs the conversation to start with the user
        msgs.pop(0)

    def gen():
        try:
            for ev in chat_stream(msgs):
                yield f"data: {ev}\n\n"
        except Exception:
            log.exception("chat failed")
            yield "data: " + json.dumps({"type": "error", "text": "The research assistant failed. Try again."}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------- settings + alerts
class SettingsReq(BaseModel):
    capital: float | None = Field(default=None, gt=0, le=1e11)
    risk_pct: float | None = Field(default=None, ge=0.05, le=10)
    alerts_enabled: bool | None = None
    telegram_bot_token: str | None = Field(default=None, max_length=100, pattern=r"^(\d{5,15}:[\w-]{20,80})?$")
    telegram_chat_id: str | None = Field(default=None, max_length=30, pattern=r"^-?\d*$")


def _public_settings():
    s = db.get_settings()
    tok = s.pop("telegram_bot_token", "")
    s["telegram_configured"] = bool(tok and s.get("telegram_chat_id"))
    s["telegram_token_hint"] = f"...{tok[-4:]}" if tok else ""
    return s


@app.get("/api/settings")
def get_settings():
    return _public_settings()


@app.put("/api/settings")
def put_settings(req: SettingsReq):
    vals = req.model_dump(exclude_none=True)
    if "alerts_enabled" in vals:
        vals["alerts_enabled"] = "1" if vals["alerts_enabled"] else "0"
    if vals.get("telegram_bot_token") and not vals.get("telegram_chat_id"):
        try:  # a new bot means a new chat: detect it from the bot's /start message
            cid = alerts.find_chat_id(vals["telegram_bot_token"].strip())
            vals["telegram_chat_id"] = cid or ""
        except Exception:
            log.warning("could not detect the Telegram chat id")
    db.set_settings(vals)
    return _public_settings()


@app.post("/api/alerts/test")
def test_alert():
    s = db.get_settings()
    if not s["telegram_bot_token"]:
        raise HTTPException(400, "Save a Telegram bot token first.")
    if not s["telegram_chat_id"]:
        cid = alerts.find_chat_id(s["telegram_bot_token"])
        if cid:
            db.set_settings({"telegram_chat_id": cid})
    ok = alerts.send("✅ AI Swing Trader alerts connected. You will receive new picks and target/stop alerts here.",
                     force=True)
    if not ok:
        raise HTTPException(400, "Telegram message failed. Check the bot token and send /start to your bot first.")
    return {"ok": True}


# ---------------------------------------------------------------- paper trading
class PaperReq(BaseModel):
    symbol: str = Field(max_length=20)
    qty: int = Field(gt=0, le=10_000_000)
    target: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    scan_id: int | None = None


@app.get("/api/paper")
def paper_list(refresh: bool = True):
    if refresh:
        try:
            paper.update_all(use_live=True)
        except Exception:
            log.exception("paper refresh failed")
    return paper.portfolio()


@app.post("/api/paper")
def paper_open(req: PaperReq):
    if req.target <= req.stop_loss:
        raise HTTPException(400, "Target must be above the stop-loss for a buy trade")
    try:
        return paper.open_trade(_symbol(req.symbol), req.qty, req.target, req.stop_loss, req.scan_id)
    except paper.TradeRejected as e:
        raise HTTPException(400, str(e))


@app.post("/api/paper/{trade_id}/close")
def paper_close(trade_id: int):
    trade = db.get_paper(trade_id)
    if not trade:
        raise HTTPException(404, "Paper trade not found")
    if trade["status"] != "open":
        raise HTTPException(409, "This paper trade is already closed")
    paper.close(trade_id)
    return paper.portfolio()


@app.delete("/api/paper/{trade_id}")
def paper_delete(trade_id: int):
    if not db.get_paper(trade_id):
        raise HTTPException(404, "Paper trade not found")
    db.delete_paper(trade_id)
    return {"ok": True}


@app.get("/api/flows")
def flows():
    return nse.fii_dii()


# ---------------------------------------------------------------- sectors / backtest
@app.get("/api/sectors")
def sectors(index: str = "NIFTY500"):
    from .sectors import sector_heatmap

    if index.upper() not in UNIVERSES:
        raise HTTPException(400, "index must be NIFTY50, NIFTY100 or NIFTY500")
    return sector_heatmap(index)


@app.get("/api/backtest")
def backtest_report():
    try:
        r = load_model()["report"]
    except FileNotFoundError:
        raise HTTPException(404, "Model not trained yet")
    out = {k: r.get(k) for k in ("auc", "test_from", "test_to", "trained_at", "portfolio", "backtest")}
    out["top_features"] = (r.get("feature_importance") or [])[:10]
    return out


@app.get("/api/research")
def research_summary():
    """22-year walk-forward results of the live setting (built by `python -m app.research.analyze export`)."""
    path = STORAGE / "research" / "summary.json"
    if not path.exists():
        raise HTTPException(404, "Run: python -m app.research.analyze export")
    out = json.loads(path.read_text())
    llm = STORAGE / "research" / "llm_summary.json"
    if llm.exists():
        out["llm_test"] = json.loads(llm.read_text())
    return out
