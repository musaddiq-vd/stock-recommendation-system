"""Single-user password login with signed, expiring bearer tokens.

Set APP_PASSWORD (and APP_SECRET) to protect the API. When APP_PASSWORD is empty, auth is disabled,
which is only meant for local development.
"""
import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import defaultdict, deque

log = logging.getLogger(__name__)

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
APP_SECRET = os.getenv("APP_SECRET", "")
TOKEN_TTL = int(os.getenv("TOKEN_TTL_HOURS", 24 * 7)) * 3600
MAX_ATTEMPTS, WINDOW = 10, 15 * 60  # failed logins allowed per client per window

if APP_PASSWORD and not APP_SECRET:
    # tokens still work, but every restart logs everyone out
    APP_SECRET = secrets.token_hex(32)
    log.warning("APP_SECRET is not set; a random one is used and sessions end on restart")
if not APP_PASSWORD:
    log.warning("APP_PASSWORD is not set: the API is open to anyone who can reach it (local development only)")

_attempts: dict[str, deque] = defaultdict(deque)


def enabled() -> bool:
    return bool(APP_PASSWORD)


def _sign(payload: str) -> str:
    return hmac.new(APP_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_token() -> str:
    exp = str(int(time.time()) + TOKEN_TTL)
    return base64.urlsafe_b64encode(f"{exp}.{_sign(exp)}".encode()).decode()


def verify_token(token: str) -> bool:
    try:
        exp, sig = base64.urlsafe_b64decode(token.encode()).decode().split(".", 1)
        return hmac.compare_digest(sig, _sign(exp)) and int(exp) > time.time()
    except Exception:
        return False


def rate_limited(client: str) -> bool:
    q = _attempts[client]
    while q and q[0] < time.time() - WINDOW:
        q.popleft()
    return len(q) >= MAX_ATTEMPTS


def check_password(client: str, password: str) -> bool:
    ok = hmac.compare_digest(password.encode(), APP_PASSWORD.encode())
    if not ok:
        _attempts[client].append(time.time())
    else:
        _attempts.pop(client, None)
    return ok
