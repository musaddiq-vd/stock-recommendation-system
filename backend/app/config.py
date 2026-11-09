import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

STORAGE = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))
PRICE_DIR = STORAGE / "prices"
UNIVERSE_DIR = STORAGE / "universe"
MODEL_PATH = STORAGE / "model.joblib"
DB_PATH = STORAGE / "app.db"
for d in (STORAGE, PRICE_DIR, UNIVERSE_DIR):
    d.mkdir(parents=True, exist_ok=True)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "moonshotai.kimi-k2.5")

# Swing trade setup: hold 15-20 trading days
HOLD_MIN_DAYS = int(os.getenv("HOLD_MIN_DAYS", 15))
HOLD_MAX_DAYS = int(os.getenv("HOLD_MAX_DAYS", 20))
TARGET_ATR = float(os.getenv("TARGET_ATR", 3.5))  # target = entry + 3.5 x ATR
STOP_ATR = float(os.getenv("STOP_ATR", 2.5))       # stop = entry - 2.5 x ATR (chosen by the 22-year walk-forward)

HISTORY_PERIOD = os.getenv("HISTORY_PERIOD", "6y")
MIN_AVG_TRADED_VALUE = float(os.getenv("MIN_AVG_TRADED_VALUE", 5e7))  # Rs 5 cr/day
AGENT_TOP_N = int(os.getenv("AGENT_TOP_N", 15))

# Rules chosen by the survivorship-free 22-year walk-forward (app/research/v2.py, delisted stocks included)
ML_TOP_PCT = float(os.getenv("ML_TOP_PCT", 98))         # only the top 2% of the daily ML ranking can be a Buy
# Market gate, chosen on pre-2016 data and confirmed on 2016+: Nifty above its 200-day average AND
# more than half of the liquid stocks above their own 200-day averages
BREADTH_MIN = float(os.getenv("BREADTH_MIN", 0.5))

BENCHMARK = "^NSEI"

# Server
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
# The scheduler runs inside the API process: keep exactly one API worker, or set this to 0 on extra workers
RUN_SCHEDULER = os.getenv("RUN_SCHEDULER", "1") == "1"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Bump when tiers or scoring rules change, so the live track record never mixes different rule sets
SCORING_VERSION = "7"
