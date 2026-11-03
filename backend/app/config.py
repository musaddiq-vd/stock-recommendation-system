import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

STORAGE = BASE_DIR / "storage"
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

# Gates validated on 22 years of walk-forward data (see app/research)
REGIME_FILTER = os.getenv("REGIME_FILTER", "1") == "1"   # long "strong" picks only when Nifty > 200 EMA
RUBRIC_MIN = float(os.getenv("RUBRIC_MIN", 70))            # rule-based rubric score needed for "strong"
# Short setups lost money in every tested configuration over 22 years, so they are off by default
SHORTS_ENABLED = os.getenv("SHORTS_ENABLED", "0") == "1"

BENCHMARK = "^NSEI"
