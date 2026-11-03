import logging
import warnings
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import alerts, db, paper
from .agents.orchestrator import market_regime, start_scan, update_outcomes
from .agents.tools import get_fundamentals, get_technicals
from .chat import chat_stream
from .data import nse
from .data.prices import load
from .model.train import load_model, train_all
from .universe import get_universe

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")


def daily_job():
    for step in (lambda: nse.update_recent(7), nse.fii_dii, paper.update_all, update_outcomes):
        try:
            step()
        except Exception:
            log.exception("daily step failed")
    start_scan("NIFTY500", None, "news")


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
    # Mon-Fri after NSE close (data on Yahoo settles by ~18:00 IST)
    scheduler.add_job(daily_job, "cron", day_of_week="mon-fri", hour=18, minute=30, id="daily", replace_existing=True)
    scheduler.add_job(intraday_job, "cron", day_of_week="mon-fri", hour="9-15", minute="*/15", id="intraday",
                      replace_existing=True)
    # Markets change: retrain both models on fresh data on the 1st Saturday of every month
    scheduler.add_job(train_all, "cron", day="1-7", day_of_week="sat", hour=10, id="retrain", replace_existing=True)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="AI Swing Trader (NSE)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ScanReq(BaseModel):
    universe: str = "NIFTY100"
    sector: str | None = None
    mode: str = "news"  # "news" = technical + web/news research, "technical" = technical only
    top_n: int = 15


class ChatReq(BaseModel):
    messages: list[dict]


@app.get("/api/health")
def health():
    return {"ok": True}


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
    if req.universe.upper() not in ("NIFTY50", "NIFTY100", "NIFTY500"):
        raise HTTPException(400, "universe must be NIFTY50/NIFTY100/NIFTY500")
    if req.mode not in ("news", "technical"):
        raise HTTPException(400, "mode must be news/technical")
    return {"scan_id": start_scan(req.universe.upper(), req.sector or None, req.mode, max(3, min(req.top_n, 40)))}


@app.get("/api/scans")
def scans():
    return db.list_scans()


@app.get("/api/scans/latest")
def latest(universe: str | None = None):
    return db.latest_scan(universe.upper() if universe else None) or {}


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: int):
    s = db.get_scan(scan_id)
    if not s:
        raise HTTPException(404)
    return s


@app.get("/api/stock/{symbol}")
def stock(symbol: str):
    symbol = symbol.upper()
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
def model_report(side: str = "long"):
    r = dict(load_model("short" if side == "short" else "long")["report"])
    r.pop("prob_percentiles", None)
    return r


@app.get("/api/performance")
def performance():
    update_outcomes()
    return db.performance()


@app.post("/api/chat")
def chat(req: ChatReq):
    def gen():
        try:
            for ev in chat_stream(req.messages[-20:]):
                yield f"data: {ev}\n\n"
        except Exception as e:
            yield f'data: {{"type": "error", "text": "{str(e)[:200]}"}}\n\n'
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- settings ----------
class SettingsReq(BaseModel):
    capital: float | None = None
    risk_pct: float | None = None
    margin_pct: float | None = None
    alerts_enabled: bool | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None


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
    if vals.get("telegram_bot_token") and not vals.get("telegram_chat_id") and not db.get_settings()["telegram_chat_id"]:
        try:
            cid = alerts.find_chat_id(vals["telegram_bot_token"].strip())
            if cid:
                vals["telegram_chat_id"] = cid
        except Exception:
            pass
    db.set_settings(vals)
    return _public_settings()


@app.post("/api/alerts/test")
def test_alert():
    s = db.get_settings()
    if not s["telegram_chat_id"] and s["telegram_bot_token"]:
        cid = alerts.find_chat_id(s["telegram_bot_token"])
        if cid:
            db.set_settings({"telegram_chat_id": cid})
    ok = alerts.send("✅ AI Swing Trader alerts connected. You will receive new picks and target/stop alerts here.", force=True)
    if not ok:
        raise HTTPException(400, "Telegram message failed. Check bot token and send /start to your bot first.")
    return {"ok": True}


# ---------- paper trading ----------
class PaperReq(BaseModel):
    symbol: str
    side: str = "long"
    qty: int
    lots: int | None = None
    entry: float
    target: float
    stop_loss: float
    opened_on: str
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
    if req.qty <= 0:
        raise HTTPException(400, "qty must be > 0")
    tid = db.add_paper(req.model_dump())
    return {"id": tid}


@app.post("/api/paper/{trade_id}/close")
def paper_close(trade_id: int):
    paper.close(trade_id)
    return paper.portfolio()


@app.delete("/api/paper/{trade_id}")
def paper_delete(trade_id: int):
    db.delete_paper(trade_id)
    return {"ok": True}


@app.get("/api/flows")
def flows():
    return nse.fii_dii()


# ---------- sectors / backtest ----------
@app.get("/api/sectors")
def sectors(index: str = "NIFTY500"):
    from .sectors import sector_heatmap

    if index.upper() not in ("NIFTY50", "NIFTY100", "NIFTY500"):
        raise HTTPException(400, "bad index")
    return sector_heatmap(index)


@app.get("/api/backtest")
def backtest_report():
    out = {}
    for side in ("long", "short"):
        try:
            r = load_model(side)["report"]
        except FileNotFoundError:
            continue
        out[side] = {k: r.get(k) for k in ("auc", "test_from", "test_to", "trained_at", "feature_set",
                                           "comparison", "portfolio", "backtest")}
        out[side]["top_features"] = (r.get("feature_importance") or [])[:10]
    return out


@app.get("/api/research")
def research_summary():
    """22-year walk-forward results of the live setting (built by `python -m app.research.analyze export`)."""
    import json as _json

    from .config import STORAGE

    path = STORAGE / "research" / "summary.json"
    if not path.exists():
        raise HTTPException(404, "Run: python -m app.research.analyze export")
    out = _json.loads(path.read_text())
    llm = STORAGE / "research" / "llm_summary.json"
    if llm.exists():
        out["llm_test"] = _json.loads(llm.read_text())
    return out
