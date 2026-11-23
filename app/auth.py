"""Đăng nhập trang Admin phía server.

* Tài khoản nhân viên lưu ở `data/admin_users.json` (git-ignore), mật khẩu băm PBKDF2-SHA256 + salt riêng.
  Lần chạy đầu tạo sẵn 3 tài khoản demo (admin/admin123, lan/123456, linh/123456); nhớ đổi khi dùng thật:
  `python -m app.auth set <user> <mật khẩu mới> [tên hiển thị]`.
* Đăng nhập trả về token ngẫu nhiên (hết hạn sau `SESSION_HOURS`), trình duyệt gửi kèm header
  `Authorization: Bearer <token>` cho các API quản trị. Token lưu trong bộ nhớ: khởi động lại server là phải đăng nhập lại.
* Tắt kiểm tra (chỉ để chạy thử nội bộ): đặt `ADMIN_AUTH=0`.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .config import BASE_DIR

USERS_FILE = Path(os.getenv("ADMIN_USERS_FILE", str(BASE_DIR / "data" / "admin_users.json")))
ENABLED = os.getenv("ADMIN_AUTH", "1") != "0"
SESSION_HOURS = float(os.getenv("ADMIN_SESSION_HOURS", "12"))
_ITER = 200_000
_DEMO = [("admin", "admin123", "Quản trị viên"), ("lan", "123456", "Lan (CSKH Tier-2)"), ("linh", "123456", "Linh (Bảo hành)")]

_tokens: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_fails: dict[str, list[float]] = {}

# API chỉ dành cho nhân viên (method, regex đường dẫn). Các API khách dùng (chat, đặt hàng, webhook...) để mở.
PROTECTED = [
    ("GET", r"^/api/(omnichannel/)?conversations/?$"),
    ("POST", r"^/api/conversations/[^/]+/(reply|takeover|qa|close)$"),
    ("GET", r"^/api/customers/?$"),
    ("POST", r"^/api/llm/config$"),
    ("POST", r"^/api/kb/(upload|reload)$"),
    ("*", r"^/api/kb/gaps(/.*)?$"),
    ("POST", r"^/api/copilot/(suggest|rewrite|summary)$"),
    ("*", r"^/api/reports(/.*)?$"),
    ("*", r"^/api/integrations/config$"),
    ("POST", r"^/api/integrations/telegram/poll$"),
    ("GET", r"^/api/(traces|metrics|state)(/.*)?$"),
    ("GET", r"^/api/agent/overview$"),
    ("POST", r"^/api/evals/run$"),
    ("POST", r"^/api/shops/active$"),
]


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _ITER).hex()


def _load() -> dict[str, dict[str, str]]:
    if not USERS_FILE.exists():
        users = {}
        for u, p, n in _DEMO:
            salt = secrets.token_hex(16)
            users[u] = {"name": n, "salt": salt, "hash": _hash(p, salt)}
        _save(users)
        return users
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def _save(users: dict[str, dict[str, str]]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def set_user(user: str, password: str, name: str | None = None) -> None:
    users = _load()
    salt = secrets.token_hex(16)
    users[user] = {"name": name or users.get(user, {}).get("name") or user, "salt": salt, "hash": _hash(password, salt)}
    _save(users)


def login(user: str, password: str, client: str = "") -> dict[str, Any] | None:
    user = (user or "").strip().lower()
    now = time.time()
    key = f"{client}|{user}"
    recent = [t for t in _fails.get(key, []) if now - t < 300]
    if len(recent) >= 8:  # chống dò mật khẩu: tối đa 8 lần sai / 5 phút
        raise PermissionError("Sai quá nhiều lần, thử lại sau 5 phút.")
    acc = _load().get(user)
    if not acc or not hmac.compare_digest(acc["hash"], _hash(password or "", acc["salt"])):
        _fails[key] = recent + [now]
        return None
    _fails.pop(key, None)
    token = secrets.token_urlsafe(32)
    sess = {"user": user, "name": acc["name"], "login_at": now, "exp": now + SESSION_HOURS * 3600}
    with _lock:
        _tokens[token] = sess
    return {"token": token, **sess}


def logout(token: str) -> None:
    with _lock:
        _tokens.pop(token, None)


def session(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    s = _tokens.get(token)
    if s and s["exp"] < time.time():
        logout(token)
        return None
    return s


def token_from_header(value: str | None) -> str | None:
    if value and value.lower().startswith("bearer "):
        return value[7:].strip()
    return None


def is_protected(method: str, path: str) -> bool:
    return ENABLED and any((m == "*" or m == method) and re.match(rx, path) for m, rx in PROTECTED)


if __name__ == "__main__":  # python -m app.auth set <user> <password> [name]
    if len(sys.argv) >= 4 and sys.argv[1] == "set":
        set_user(sys.argv[2].lower(), sys.argv[3], " ".join(sys.argv[4:]) or None)
        print(f"Đã cập nhật tài khoản {sys.argv[2].lower()} trong {USERS_FILE}")
    else:
        print("Cách dùng: python -m app.auth set <user> <mật khẩu> [tên hiển thị]")
