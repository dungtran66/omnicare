"""Observability: tracing theo span, log JSON có cấu trúc, lưu trace vào SQLite và tính metrics."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger("omnicare")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


def log_event(event: str, **fields: Any) -> None:
    """Ghi log dạng JSON một dòng (dễ đẩy sang Loki/ELK/Datadog)."""
    logger.info(json.dumps({"ts": time.time(), "event": event, **fields}, ensure_ascii=False, default=str))


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(((self.end or time.perf_counter()) - self.start) * 1000, 2),
            "status": self.status,
            "attributes": self.attributes,
        }


class Trace:
    """Một trace = một lượt hỏi-đáp của agent, gồm nhiều span (guardrail, retrieve, tool, llm...)."""

    def __init__(self, session_id: str, user_input: str):
        self.trace_id = uuid.uuid4().hex[:16]
        self.session_id = session_id
        self.user_input = user_input
        self.spans: list[Span] = []
        self.started = time.perf_counter()
        self.created_at = time.time()
        self.attributes: dict[str, Any] = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @contextmanager
    def span(self, name: str, **attributes: Any):
        s = Span(name=name, start=time.perf_counter(), attributes=dict(attributes))
        self.spans.append(s)
        cb = getattr(self, "on_span", None)  # báo tiến trình cho SSE (/api/chat/stream)
        if cb:
            try:
                cb(name)
            except Exception:  # noqa: BLE001 - không để lỗi hiển thị làm hỏng xử lý
                pass
        try:
            yield s
        except Exception as exc:
            s.status = "error"
            s.attributes["error"] = repr(exc)
            raise
        finally:
            s.end = time.perf_counter()
            log_event("span", trace_id=self.trace_id, **s.to_dict())

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    @property
    def latency_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 2)


def _shop_trace_db():
    from .shop import active  # import muộn để tránh vòng import

    return active().trace_db


def _shop_name() -> str:
    from .shop import active

    return active().name


# Giá USD / 1 triệu token (input, output) để ước tính chi phí; build_llm() cập nhật model đang dùng.
_ACTIVE_MODEL = {"name": ""}
_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def set_active_model(name: str) -> None:
    _ACTIVE_MODEL["name"] = name or ""


def _price_per_mtok(name: str | None = None) -> tuple[float, float, str]:
    """Giá (input, output) USD / 1 triệu token + nguồn giá.
    Ưu tiên LLM_PRICE_INPUT / LLM_PRICE_OUTPUT trong .env (proxy/model không có trong bảng, vd GPT qua codex.fosta.app)."""
    import os

    pin, pout = os.getenv("LLM_PRICE_INPUT"), os.getenv("LLM_PRICE_OUTPUT")
    if pin and pout:
        try:
            return float(pin), float(pout), "env"
        except ValueError:
            pass
    name = name or _ACTIVE_MODEL["name"]
    for model_id, price in _PRICES.items():
        if model_id in name:
            return price[0], price[1], "bảng giá"
    return 0.15, 0.60, "mặc định"  # chưa biết giá model này: ước tính, nên đặt LLM_PRICE_* cho đúng


class TraceStore:
    def __init__(self, path=None):
        self.path = str(path or _shop_trace_db())
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY, session_id TEXT, created_at REAL,
                    user_input TEXT, output TEXT, intent TEXT, status TEXT,
                    latency_ms REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
                    guardrail_flags TEXT, spans TEXT, attributes TEXT, feedback INTEGER)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY, data TEXT, updated_at REAL)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY, data TEXT, updated_at REAL)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY, data TEXT, updated_at REAL)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS customer_memory (
                    member_id TEXT PRIMARY KEY, data TEXT, updated_at REAL)"""
            )
            c.execute(  # khách chấm sao (CSAT) khi nhân viên kết thúc hội thoại
                """CREATE TABLE IF NOT EXISTS csat (
                    session_id TEXT PRIMARY KEY, stars INTEGER, created_at REAL)"""
            )
            c.execute(  # hội thoại nhân viên đã đóng: trang khách dựa vào mốc này để mở hội thoại mới
                """CREATE TABLE IF NOT EXISTS closed_conversations (
                    session_id TEXT PRIMARY KEY, closed_at REAL)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS customers (
                    member_id TEXT PRIMARY KEY, data TEXT, updated_at REAL)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS kb_gaps_resolved (
                    gap_id TEXT PRIMARY KEY, resolved_at REAL)"""
            )
            c.execute(  # Báo cáo do AI viết (báo cáo tuần...)
                """CREATE TABLE IF NOT EXISTS ai_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, data TEXT, created_at REAL)"""
            )
            c.execute(  # Auto-FAQ: kết quả model đánh giá/soạn nháp, để khởi động lại không phải gọi model lần nữa
                """CREATE TABLE IF NOT EXISTS autofaq_cache (
                    qkey TEXT PRIMARY KEY, data TEXT, model TEXT, created_at REAL)"""
            )

    def _conn(self):
        return sqlite3.connect(self.path)

    def save(self, trace: Trace, output: str, status: str, intent: str, flags: list[str]) -> None:
        row = (
            trace.trace_id, trace.session_id, trace.created_at, trace.user_input, output, intent,
            status, trace.latency_ms, trace.prompt_tokens, trace.completion_tokens,
            json.dumps(flags, ensure_ascii=False),
            json.dumps([s.to_dict() for s in trace.spans], ensure_ascii=False, default=str),
            json.dumps(trace.attributes, ensure_ascii=False, default=str), None,
        )
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        log_event("trace", trace_id=trace.trace_id, status=status, intent=intent,
                  latency_ms=trace.latency_ms, flags=flags)

    def save_csat(self, session_id: str, stars: int) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO csat VALUES (?, ?, ?)", (session_id, stars, time.time()))

    def set_feedback(self, trace_id: str, score: int) -> bool:
        with self._lock, self._conn() as c:
            cur = c.execute("UPDATE traces SET feedback=? WHERE trace_id=?", (score, trace_id))
            return cur.rowcount > 0

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = c.execute("SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("guardrail_flags", "spans", "attributes"):
                d[k] = json.loads(d[k]) if d[k] else None
            out.append(d)
        return out

    def get(self, trace_id: str) -> dict[str, Any] | None:
        return next((t for t in self.list(10_000) if t["trace_id"] == trace_id), None)

    def _csat_summary(self) -> dict[str, Any]:
        with self._conn() as c:
            stars = [r[0] for r in c.execute("SELECT stars FROM csat").fetchall()]
        return {"count": len(stars), "avg_stars": round(sum(stars) / len(stars), 2) if stars else None,
                "satisfied_rate": round(sum(1 for s in stars if s >= 4) / len(stars), 3) if stars else None}

    def metrics(self) -> dict[str, Any]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT status, intent, latency_ms, prompt_tokens, completion_tokens, guardrail_flags, feedback, attributes FROM traces"
            ).fetchall()
        n = len(rows)
        if n == 0:
            return {"total_requests": 0}
        lat = sorted(r[2] for r in rows)
        by_status: dict[str, int] = {}
        by_intent: dict[str, int] = {}
        flag_counts: dict[str, int] = {}
        for r in rows:
            by_status[r[0]] = by_status.get(r[0], 0) + 1
            by_intent[r[1]] = by_intent.get(r[1], 0) + 1
            for f in json.loads(r[5] or "[]"):
                flag_counts[f] = flag_counts.get(f, 0) + 1
        fb = [r[6] for r in rows if r[6] is not None]
        prompt_t = sum(r[3] for r in rows)
        comp_t = sum(r[4] for r in rows)
        deflected = sum(1 for r in rows if r[0] in ("ok", "clarify", "confirm"))
        # Chỉ tính tiền lượt có model thật trả lời (attributes.llm_used); MockLLM chỉ đếm từ, không tốn tiền.
        # Giá lấy theo đúng model của từng lượt (model chính hay model dự phòng).
        est_cost, real_calls, real_p, real_c, sources = 0.0, 0, 0, 0, set()
        for r in rows:
            try:
                used = (json.loads(r[7] or "{}") or {}).get("llm_used") or []
            except ValueError:
                used = []
            if not used:
                continue
            in_rate, out_rate, src = _price_per_mtok(str(used[0].get("model", "")))
            sources.add(src)
            real_calls += 1
            real_p += r[3] or 0
            real_c += r[4] or 0
            est_cost += ((r[3] or 0) * in_rate + (r[4] or 0) * out_rate) / 1_000_000
        est_cost = round(est_cost, 6)
        return {
            "total_requests": n,
            "latency_ms": {"p50": lat[n // 2], "p95": lat[min(n - 1, int(n * 0.95))], "max": lat[-1]},
            "tokens": {"prompt": prompt_t, "completion": comp_t},
            "estimated_cost_usd": est_cost,
            "model_usage": {"calls": real_calls, "prompt_tokens": real_p, "completion_tokens": real_c,
                            "price_source": sorted(sources)},
            "by_status": by_status,
            "by_intent": by_intent,
            "guardrail_flags": flag_counts,
            "deflection_rate": round(deflected / n, 3),
            "handoff_rate": round(by_status.get("handoff", 0) / n, 3),
            "blocked_rate": round(by_status.get("blocked", 0) / n, 3),
            "negative_sentiment_count": sum(1 for r in rows if {"negative_sentiment", "abusive_language"} & set(json.loads(r[5] or "[]"))),
            "feedback": {"count": len(fb), "positive_rate": round(sum(1 for x in fb if x > 0) / len(fb), 3) if fb else None},
            "csat": self._csat_summary(),
        }

    # ---------------- Persistence cho Orders, Tickets, Conversations, Customers, KB Gaps ----------------
    def save_order(self, order_id: str, data: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO orders VALUES (?, ?, ?)",
                (order_id, json.dumps(data, ensure_ascii=False, default=str), time.time()),
            )

    def load_orders(self) -> dict[str, dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT order_id, data FROM orders").fetchall()
        return {r[0]: json.loads(r[1]) for r in rows if r[1]}

    def save_ticket(self, ticket_id: str, data: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO tickets VALUES (?, ?, ?)",
                (ticket_id, json.dumps(data, ensure_ascii=False, default=str), time.time()),
            )

    def load_tickets(self) -> dict[str, dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT ticket_id, data FROM tickets ORDER BY updated_at DESC").fetchall()
        return {r[0]: json.loads(r[1]) for r in rows if r[1]}

    def save_conversation(self, session_id: str, data: dict[str, Any]) -> None:
        # Lưu bản sao gọn của hội thoại vào SQLite
        payload = {
            "id": session_id,
            "name": data.get("name") or data.get("label") or f"Khách ({session_id})",
            "channel": data.get("channel") or f"Website {_shop_name()}",
            "type": data.get("type") or (data.get("customer") or {}).get("type", "guest"),
            "customer": data.get("customer"),
            "takeover": bool(data.get("takeover", False)),
            "last_order_id": data.get("last_order_id"),
            "pending": data.get("pending"),
            "pending_action": data.get("pending_action"),
            "handoff_summary": data.get("handoff_summary"),
            "history": (data.get("history") or [])[-20:],
            "messages": (data.get("messages") or [])[-60:],
            "updated_at": data.get("updated_at") or time.time(),
        }
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO conversations VALUES (?, ?, ?)",
                (session_id, json.dumps(payload, ensure_ascii=False, default=str), payload["updated_at"]),
            )

    def delete_conversation(self, session_id: str) -> float:
        """Xóa hội thoại và ghi mốc đóng (giây). Trả về mốc đóng."""
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            c.execute("INSERT OR REPLACE INTO closed_conversations VALUES (?, ?)", (session_id, now))
        return now

    def closed_at(self, session_id: str) -> float | None:
        with self._conn() as c:
            row = c.execute("SELECT closed_at FROM closed_conversations WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row else None

    def load_conversations(self) -> dict[str, dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT session_id, data FROM conversations ORDER BY updated_at DESC").fetchall()
        return {r[0]: json.loads(r[1]) for r in rows if r[1]}

    def save_memory(self, member_id: str, data: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO customer_memory VALUES (?, ?, ?)",
                      (member_id, json.dumps(data, ensure_ascii=False, default=str), time.time()))

    def load_memory(self, member_id: str) -> dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT data FROM customer_memory WHERE member_id = ?", (member_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def save_customer(self, member_id: str, data: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO customers VALUES (?, ?, ?)",
                (member_id, json.dumps(data, ensure_ascii=False, default=str), time.time()),
            )

    def load_customers(self) -> dict[str, dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT member_id, data FROM customers").fetchall()
        return {r[0]: json.loads(r[1]) for r in rows if r[1]}

    def mark_gap_resolved(self, gap_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO kb_gaps_resolved VALUES (?, ?)", (gap_id, time.time()))

    def save_report(self, kind: str, data: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO ai_reports (kind, data, created_at) VALUES (?, ?, ?)",
                      (kind, json.dumps(data, ensure_ascii=False), time.time()))

    def latest_report(self, kind: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT data FROM ai_reports WHERE kind = ? ORDER BY id DESC LIMIT 1", (kind,)).fetchone()
        return json.loads(row[0]) if row else None

    def load_autofaq_cache(self) -> dict[str, dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT qkey, data FROM autofaq_cache").fetchall()
        return {k: json.loads(d) for k, d in rows}

    def save_autofaq(self, qkey: str, data: dict[str, Any], model: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO autofaq_cache VALUES (?, ?, ?, ?)",
                      (qkey, json.dumps(data, ensure_ascii=False), model, time.time()))

    def list_resolved_gaps(self) -> set[str]:
        with self._conn() as c:
            rows = c.execute("SELECT gap_id FROM kb_gaps_resolved").fetchall()
        return {r[0] for r in rows}
