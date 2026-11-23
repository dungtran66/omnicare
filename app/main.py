"""OmniCare API (FastAPI)."""
from __future__ import annotations

import asyncio
import os
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import ai_features, auth, autofaq, config
from .agent import OmniCareAgent, shipping_fee
from .llm import build_llm
from .rag import parse_document_to_markdown
from .shop import activate as activate_shop, active as active_shop, list_shops
from .tools import ToolError

NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}  # trình duyệt luôn lấy bản trang mới nhất sau khi sửa
app = FastAPI(title="OmniCare", version="0.2.0", description="AI Agent chăm sóc khách hàng - Production Ready")
agent = OmniCareAgent()
STATIC = Path(__file__).parent / "static"


# Widget nhúng trên website khác gọi các API khách này (không cookie; API Admin vẫn cần token Bearer)
_WIDGET_API = re.compile(r"^/api/(chat|chat/stream|shop|conversations/[A-Za-z0-9_.:-]{1,120})$")
_WIDGET_ORIGINS = [o.strip() for o in os.getenv("WIDGET_ORIGINS", "*").split(",") if o.strip()]


def _cors_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    return "*" if "*" in _WIDGET_ORIGINS else (origin if origin in _WIDGET_ORIGINS else None)


@app.middleware("http")
async def widget_cors(request: Request, call_next):
    """CORS cho widget.js: chỉ các API khách (chat, thông tin shop, đọc hội thoại theo mã phiên)."""
    allow = _cors_origin(request.headers.get("origin")) if _WIDGET_API.match(request.url.path) else None
    if allow and request.method == "OPTIONS":
        return Response(status_code=204, headers={
            "Access-Control-Allow-Origin": allow, "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "600"})
    resp = await call_next(request)
    if allow:
        resp.headers["Access-Control-Allow-Origin"] = allow
    return resp


@app.middleware("http")
async def admin_guard(request: Request, call_next):
    """API quản trị bắt buộc token đăng nhập Admin (xem app/auth.py)."""
    if auth.is_protected(request.method, request.url.path):
        if not auth.session(auth.token_from_header(request.headers.get("authorization"))):
            return JSONResponse({"detail": "Cần đăng nhập Admin"}, status_code=401)
    return await call_next(request)


class AdminLoginRequest(BaseModel):
    user: str
    password: str


@app.post("/api/admin/login")
def admin_login(req: AdminLoginRequest, request: Request) -> dict[str, Any]:
    try:
        res = auth.login(req.user, req.password, request.client.host if request.client else "")
    except PermissionError as exc:
        raise HTTPException(429, str(exc)) from exc
    if not res:
        raise HTTPException(401, "Sai tên đăng nhập hoặc mật khẩu.")
    return res


@app.get("/api/admin/me")
def admin_me(request: Request) -> dict[str, Any]:
    s = auth.session(auth.token_from_header(request.headers.get("authorization")))
    if not s:
        raise HTTPException(401, "Chưa đăng nhập")
    return {k: s[k] for k in ("user", "name", "login_at")}


@app.post("/api/admin/logout")
def admin_logout(request: Request) -> dict[str, Any]:
    tok = auth.token_from_header(request.headers.get("authorization"))
    if tok:
        auth.logout(tok)
    return {"ok": True}

_RATE_BUCKET: dict[str, list[float]] = {}
RATE_LIMIT_MAX = 30
RATE_LIMIT_WINDOW = 60.0


def _check_rate_limit(key: str) -> None:
    now = time.time()
    hits = [t for t in _RATE_BUCKET.get(key, []) if now - t < RATE_LIMIT_WINDOW]
    if len(hits) >= RATE_LIMIT_MAX:
        raise HTTPException(429, "Quá nhiều yêu cầu trong 1 phút. Vui lòng thử lại sau ít giây.")
    hits.append(now)
    _RATE_BUCKET[key] = hits


SID_PATTERN = r"^[A-Za-z0-9_.:-]{1,120}$"  # mã phiên chỉ gồm chữ/số/_.:- (được chèn vào HTML/onclick ở Admin)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=4000)
    session_id: str | None = Field(default=None, pattern=SID_PATTERN)
    image_info: dict[str, Any] | None = None
    # {"type": "guest"|"member", "guest_code"?, "member_id"?, "name"?, "phone"?, "order_ids"?: [...]}
    customer: dict[str, Any] | None = None
    # Ngữ cảnh trang bán hàng: {"cart": [{id, size, qty}], "product_id": "p2", "gender": "nam"|"nu"}
    page: dict[str, Any] | None = None


class LLMConfigRequest(BaseModel):
    provider: str = Field(default="mock", max_length=40)
    base_url: str = Field(default="https://api.openai.com/v1", max_length=300)
    api_key: str = Field(default="", max_length=300)
    model: str = Field(default="gpt-4o-mini", max_length=100)


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., max_length=200)
    score: int = Field(..., ge=-1, le=1)
    stars: int | None = Field(default=None, ge=1, le=5)  # CSAT 1-5 sao khi kết thúc hội thoại


class OrderItem(BaseModel):
    id: str = Field(..., max_length=40)
    size: str = Field(default="", max_length=20)
    qty: int = Field(default=1, ge=1, le=50)


class CreateOrderRequest(BaseModel):
    # items: server tự tính tiền theo danh mục (không tin số tiền trình duyệt gửi lên).
    # product/subtotal/total chỉ dùng cho kênh cũ không gửi items.
    items: list[OrderItem] | None = Field(default=None, max_length=30)
    product: str = Field(default="", max_length=500)
    total: int = Field(default=500000, ge=0)
    subtotal: int | None = Field(default=None, ge=0)  # tiền hàng; có thì server tự cộng phí ship
    customer_name: str = Field(default="Khách hàng NORIVA", max_length=200)
    phone: str = Field(default="0918888999", max_length=30)


class KBUploadRequest(BaseModel):
    filename: str = Field(..., max_length=120)
    content: str = Field(default="", max_length=100000)
    base64_data: str | None = None


class CopilotRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)


class TicketUpdateRequest(BaseModel):
    status: str | None = Field(default=None, pattern="^(open|in_progress|resolved)$")
    assignee: str | None = Field(default=None, max_length=100)


class WebhookPayload(BaseModel):
    sender_id: str = Field(..., max_length=100, pattern=r"^[A-Za-z0-9_.:-]{1,100}$")
    sender_name: str | None = Field(default=None, max_length=100)
    text: str = Field(..., max_length=4000)


class HumanReplyRequest(BaseModel):
    text: str = Field(..., max_length=4000)
    agent_name: str = Field(default="Lan (CSKH Tier-2)", max_length=100)
    takeover: bool | None = None


class TakeoverRequest(BaseModel):
    takeover: bool = Field(...)


class ApproveGapRequest(BaseModel):
    gap_id: str = Field(..., max_length=200)
    filename: str = Field(..., max_length=120)
    content: str = Field(..., max_length=50000)


class IntegrationsConfigRequest(BaseModel):
    telegram_bot_token: str | None = None


_WS_CLIENTS: list[WebSocket] = []
# WebSocket -> session_id của khách đang nghe (None = Admin đã đăng nhập, nhận mọi hội thoại).
# Khách chỉ nhận sự kiện của chính phiên mình, không bao giờ nhận danh sách hội thoại của người khác.
_WS_SID: dict[int, str | None] = {}
_OMNI_SESSIONS: dict[str, dict[str, Any]] = {}
_LLM_CFG: dict[str, str] = {"provider": config.LLM_PROVIDER, "base_url": config.LLM_BASE_URL, "model": config.LLM_MODEL}
_INTEGRATIONS_CFG: dict[str, Any] = {
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "telegram_last_update_id": 0,
}


_INTERNAL_KEYS = ("handoff_summary", "qa", "memory")  # chỉ nhân viên xem (qua API có token), không phát cho khách


def _public_conv(conv: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(conv, dict):
        return conv
    return {k: v for k, v in conv.items() if k not in _INTERNAL_KEYS}


async def _broadcast_ws(payload: dict[str, Any]) -> None:
    if "conversation" in payload:
        payload = {**payload, "conversation": _public_conv(payload["conversation"])}
    if "conversations" in payload:
        payload = {**payload, "conversations": [_public_conv(c) for c in payload["conversations"]]}
    stale: list[WebSocket] = []
    target = payload.get("session_id") or (payload.get("conversation") or {}).get("id")
    for ws in list(_WS_CLIENTS):
        sid = _WS_SID.get(id(ws))
        if sid is not None and (sid != target or "conversations" in payload):
            continue  # kết nối của khách: chỉ gửi sự kiện thuộc phiên của họ
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    for ws in stale:
        if ws in _WS_CLIENTS:
            _WS_CLIENTS.remove(ws)
        _WS_SID.pop(id(ws), None)


def _send_telegram_message(chat_id: str | int, text: str) -> bool:
    token = (_INTEGRATIONS_CFG.get("telegram_bot_token") or "").strip()
    if not token:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            return resp.status == 200
    except Exception:
        return False


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": agent.llm.name, "kb_chunks": len(agent.tools.retriever.chunks),
            "tools": list(agent.tools.tools)}


@app.get("/api/llm/config")
def get_llm_config() -> dict[str, Any]:
    return {"active_model": agent.llm.name, **_LLM_CFG}


@app.post("/api/llm/config")
def set_llm_config(req: LLMConfigRequest) -> dict[str, Any]:
    agent.llm = build_llm(req.provider, req.base_url, req.api_key, req.model)
    _LLM_CFG.update(provider=req.provider, base_url=req.base_url, model=req.model)
    return {"ok": True, "active_model": agent.llm.name, **_LLM_CFG}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    sid = req.session_id or uuid.uuid4().hex[:12]
    _check_rate_limit(sid)
    res = agent.handle(sid, req.message, image_info=req.image_info, customer=req.customer, page=req.page)
    sess = agent._session(sid)
    await _broadcast_ws({"event": "conversation_updated", "session_id": sid, "conversation": sess})
    _schedule_handoff_summary(sid, res)
    return {"session_id": sid, "takeover": bool(sess.get("takeover", False)), **res.to_dict()}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE: báo từng bước xử lý (event `step`) rồi chạy chữ câu trả lời (`token`) và kết quả đầy đủ (`done`).

    Câu trả lời chỉ được gửi sau khi đã qua guardrail đầu ra, nên không có chuyện hiện chữ rồi mới bị chặn.
    """
    sid = req.session_id or uuid.uuid4().hex[:12]
    _check_rate_limit(sid)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress(step: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, step)

    async def gen():
        task = asyncio.create_task(asyncio.to_thread(
            agent.handle, sid, req.message, image_info=req.image_info, customer=req.customer, progress=progress, page=req.page))
        while not task.done() or not queue.empty():
            try:
                step = await asyncio.wait_for(queue.get(), 0.25)
                yield f"data: {json.dumps({'type': 'step', 'step': step}, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                pass
        try:
            res = task.result()
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"
            return
        sess = agent._session(sid)
        await _broadcast_ws({"event": "conversation_updated", "session_id": sid, "conversation": sess})
        _schedule_handoff_summary(sid, res)
        words = res.answer.split(" ")
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i + 3]) + (" " if i + 3 < len(words) else "")
            yield f"data: {json.dumps({'type': 'token', 'token': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.03)
        done = {"type": "done", "session_id": sid, "takeover": bool(sess.get("takeover", False)), **res.to_dict()}
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/kb/upload")
def kb_upload(req: KBUploadRequest) -> dict[str, Any]:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(req.filename).stem).strip("_") or "uploaded_doc"
    md_text = parse_document_to_markdown(req.filename, req.content, req.base64_data)
    target = agent.tools.retriever.kb_dir / f"{safe_name}.md"
    target.write_text(md_text, encoding="utf-8")
    agent.tools.retriever.reload()
    return {"ok": True, "filename": target.name, "total_chunks": len(agent.tools.retriever.chunks)}


@app.post("/api/copilot/suggest")
def copilot_suggest(req: CopilotRequest) -> dict[str, Any]:
    return agent.copilot_suggest(req.messages)


class RewriteRequest(BaseModel):
    text: str = Field(..., max_length=2000)
    style: str = "polite"


@app.post("/api/copilot/rewrite")
def copilot_rewrite(req: RewriteRequest) -> dict[str, Any]:
    """Viết lại câu nhân viên đang soạn: polite | short | apology | friendly."""
    return ai_features.rewrite(agent.llm, req.text, req.style)


class SummaryRequest(BaseModel):
    messages: list[dict[str, Any]]
    customer: dict[str, Any] | None = None
    session_id: str | None = None


class QARequest(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)


@app.post("/api/conversations/{sid}/qa")
def conversation_qa(sid: str, req: QARequest) -> dict[str, Any]:
    """Model chấm chất lượng hội thoại vừa đóng (điểm, đã giải quyết chưa, vấn đề, gợi ý cải thiện). Lưu vào hội thoại."""
    sess = agent.sessions.get(sid)
    msgs = req.messages or (sess or {}).get("messages", [])
    # Tài liệu để giám khảo đối chiếu: tra theo cả câu khách hỏi lẫn câu AI đã trả lời (vd bảng size, phí ship)
    docs: list[dict[str, Any]] = []
    for x in msgs[-12:]:
        t = str(x.get("text", "")).strip()
        if t and x.get("role") in ("visitor", "user", "agent"):
            for d in agent.tools.call("search_knowledge_base", {"query": t[:300]}).get("results", [])[:2]:
                if d.get("id") not in {e.get("id") for e in docs}:
                    docs.append(d)
    docs = docs[:6]
    catalog = [{"name": p["name"], "price": p["price"], "old_price": p.get("old_price"), "sizes": p.get("sizes"),
                "colors": p.get("colors")} for p in agent.tools.products]
    res = ai_features.score_conversation(agent.llm, msgs, docs, catalog)
    if sess is not None and res:
        sess["qa"] = res
        agent.store.save_conversation(sid, sess)
    return res


@app.post("/api/copilot/summary")
def copilot_summary(req: SummaryRequest) -> dict[str, Any]:
    """Tóm tắt bàn giao cho nhân viên (bấm tay trên Admin). Có session_id thì lưu vào hội thoại."""
    s = ai_features.summarize_handoff(agent.llm, req.messages, req.customer)
    if req.session_id and req.session_id in agent.sessions:
        agent.sessions[req.session_id]["handoff_summary"] = s
        agent.store.save_conversation(req.session_id, agent.sessions[req.session_id])
    return s


def _schedule_handoff_summary(sid: str, res) -> None:
    """Khách vừa được chuyển nhân viên: tóm tắt hội thoại ở luồng nền (không làm khách chờ) rồi đẩy lên Admin."""
    if res.status != "handoff":
        return
    loop = asyncio.get_running_loop()

    def work() -> None:
        sess = agent._session(sid)
        s = ai_features.summarize_handoff(agent.llm, sess.get("messages", []), sess.get("customer"))
        if not s or agent.sessions.get(sid) is not sess:
            return  # nhân viên đã đóng (xóa) hội thoại trong lúc model đang tóm tắt: không ghi lại
        sess["handoff_summary"] = s
        agent.store.save_conversation(sid, sess)
        # Chỉ báo "có tóm tắt mới"; Admin tự tải lại qua /api/conversations (cần token)
        asyncio.run_coroutine_threadsafe(_broadcast_ws({"event": "summary_ready", "session_id": sid}), loop)

    threading.Thread(target=work, daemon=True).start()


@app.get("/api/reports/weekly")
def get_weekly_report() -> dict[str, Any]:
    return agent.store.latest_report("weekly") or {}


@app.post("/api/reports/weekly")
def make_weekly_report() -> dict[str, Any]:
    """Model đọc trace 7 ngày qua và viết báo cáo cho chủ shop (lưu SQLite)."""
    stats = ai_features.collect_week(agent.store.list(2000), agent.tools.products)
    rep = ai_features.weekly_report(agent.llm, stats)
    rep["model"] = agent.llm.name
    agent.store.save_report("weekly", rep)
    return rep


@app.patch("/api/tickets/{ticket_id}")
def update_ticket(ticket_id: str, req: TicketUpdateRequest) -> dict[str, Any]:
    try:
        tk = agent.tools.update_ticket_status(ticket_id, status=req.status, assignee=req.assignee)
    except ToolError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not tk:
        raise HTTPException(404, "ticket không tồn tại")
    return {"ok": True, "ticket": tk}


@app.post("/api/webhook/{channel}")
async def omnichannel_webhook(channel: str, payload: WebhookPayload) -> dict[str, Any]:
    ch = channel.lower()
    if ch not in ("facebook", "zalo", "telegram", "web"):
        raise HTTPException(400, "channel phải thuộc: facebook, zalo, telegram, web")
    sid = f"{ch}-{payload.sender_id}"
    _check_rate_limit(sid)
    ch_label = {"facebook": "Facebook Messenger", "zalo": "Zalo OA", "telegram": "Telegram", "web": "Web Widget"}[ch]
    cname = payload.sender_name or f"Khách {ch_label} ({payload.sender_id})"
    res = agent.handle(
        sid,
        payload.text,
        customer={"type": "guest", "guest_code": sid, "name": cname, "channel": ch_label},
    )
    sess = agent._session(sid)
    sess["name"] = cname
    sess["channel"] = ch_label
    agent.store.save_conversation(sid, sess)
    _OMNI_SESSIONS[sid] = sess
    await _broadcast_ws({"event": "conversation_updated", "session_id": sid, "conversation": sess})
    return {"channel": ch, "channel_label": ch_label, "session_id": sid, "sender_id": payload.sender_id,
            "reply": res.answer, "status": res.status, "intent": res.intent, "trace_id": res.trace_id,
            "conversation": sess}


def _all_conversations() -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for sid, s in agent.store.load_conversations().items():
        if s.get("messages"):
            merged[sid] = s
    for sid, s in agent.sessions.items():
        if s.get("messages"):
            merged[sid] = s
    for sid, s in _OMNI_SESSIONS.items():
        if s.get("messages"):
            merged[sid] = s
    return sorted(merged.values(), key=lambda x: x.get("updated_at") or 0, reverse=True)


@app.get("/api/omnichannel/conversations")
@app.get("/api/conversations")
def list_omnichannel_conversations() -> list[dict[str, Any]]:
    return _all_conversations()


@app.post("/api/conversations/{session_id}/close")
async def close_conversation(session_id: str) -> dict[str, Any]:
    """Nhân viên đóng hội thoại: XÓA hội thoại (bộ nhớ + SQLite) và báo cho trang khách qua WebSocket.
    Trace (Observability) và trí nhớ thành viên vẫn giữ. Khách nhắn tiếp sẽ là một hội thoại mới."""
    existed = bool(agent.sessions.pop(session_id, None)) | bool(_OMNI_SESSIONS.pop(session_id, None))
    existed = existed or session_id in agent.store.load_conversations()
    closed_at = agent.store.delete_conversation(session_id)  # mốc đóng lưu trong SQLite, khởi động lại không mất
    await _broadcast_ws({"event": "conversation_closed", "session_id": session_id, "closed_at": closed_at})
    return {"ok": True, "session_id": session_id, "deleted": existed, "closed_at": closed_at}


@app.get("/api/conversations/{session_id}")
def get_conversation_detail(session_id: str) -> dict[str, Any]:
    sess = agent.sessions.get(session_id) or agent.store.load_conversations().get(session_id)
    closed_at = agent.store.closed_at(session_id)
    if not sess:
        return {"id": session_id, "messages": [], "takeover": False, "closed_at": closed_at}
    return {**_public_conv(sess), "closed_at": closed_at}


@app.post("/api/conversations/{session_id}/reply")
async def human_reply_conversation(session_id: str, req: HumanReplyRequest) -> dict[str, Any]:
    sess = agent._session(session_id)
    if req.takeover is not None:
        sess["takeover"] = req.takeover
    else:
        sess["takeover"] = True
    now = time.time()
    sess["updated_at"] = now
    msg = {
        "role": "human",
        "sender_name": req.agent_name,
        "text": req.text.strip(),
        "status": "human_reply",
        "at": int(now * 1000),
    }
    sess["messages"].append(msg)
    sess["history"].append({"role": "assistant", "content": f"[{req.agent_name}]: {req.text.strip()}"})
    agent.store.save_conversation(session_id, sess)
    if session_id.startswith("telegram-"):
        _send_telegram_message(session_id.replace("telegram-", "", 1), f"👩‍💼 {req.agent_name}: {req.text.strip()}")
    await _broadcast_ws({"event": "human_reply", "session_id": session_id, "message": msg, "conversation": sess})
    return {"ok": True, "session_id": session_id, "takeover": sess["takeover"], "message": msg, "conversation": sess}


@app.post("/api/conversations/{session_id}/takeover")
async def toggle_conversation_takeover(session_id: str, req: TakeoverRequest) -> dict[str, Any]:
    sess = agent._session(session_id)
    sess["takeover"] = req.takeover
    sess["updated_at"] = time.time()
    agent.store.save_conversation(session_id, sess)
    await _broadcast_ws({"event": "takeover_changed", "session_id": session_id, "takeover": req.takeover, "conversation": sess})
    return {"ok": True, "session_id": session_id, "takeover": req.takeover}


@app.get("/api/customers")
def list_customers() -> list[dict[str, Any]]:
    return list(agent.store.load_customers().values())


@app.post("/api/customers")
def upsert_customer(cust: dict[str, Any]) -> dict[str, Any]:
    mid = str(cust.get("memberId") or cust.get("member_id") or "").strip()
    if not mid:
        raise HTTPException(400, "Thiếu memberId")
    cust = {k: v for k, v in cust.items() if k.lower() not in ("password", "pass", "matkhau")}  # không lưu mật khẩu
    agent.store.save_customer(mid, cust)
    return {"ok": True, "customer": cust}


# ---------------- Feature 3: Auto-FAQ / Knowledge Gaps (Captain AI style) ----------------
_SEED_KB_GAPS: list[dict[str, Any]] = [
    {
        "gap_id": "gap-alteration-pants",
        "question": "Shop có hỗ trợ cắt/sửa gấu quần dài Smart Ankle miễn phí không?",
        "count": 5,
        "source_type": "fallback_trace",
        "suggested_filename": "dich_vu_sua_gau_quan.md",
        "suggested_title": "Dịch vụ sửa gấu quần và chăm sóc trang phục NORIVA",
        "suggested_markdown": (
            "# Dịch vụ sửa gấu quần và chăm sóc trang phục NORIVA\n\n"
            "## Chính sách cắt/sửa gấu quần miễn phí\n"
            "- NORIVA hỗ trợ cắt và lên gấu quần miễn phí cho tất cả các dòng Quần Dài Smart Ankle, quần tây và quần jeans.\n"
            "- Khách đặt online chỉ cần ghi chú số đo chiều dài ống quần mong muốn (cm); thời gian xử lý cộng thêm 24 giờ trước khi giao."
        ),
    },
    {
        "gap_id": "gap-b2b-uniform",
        "question": "Mua số lượng lớn làm đồng phục công ty từ 50 áo có chiết khấu và xuất VAT không?",
        "count": 4,
        "source_type": "fallback_trace",
        "suggested_filename": "don_hang_doanh_nghiep_b2b.md",
        "suggested_title": "Chính sách đơn hàng doanh nghiệp B2B & Đồng phục",
        "suggested_markdown": (
            "# Chính sách đơn hàng doanh nghiệp B2B & Đồng phục\n\n"
            "## Chiết khấu số lượng lớn và hóa đơn VAT\n"
            "- Đơn hàng doanh nghiệp từ 30 đến 99 sản phẩm được chiết khấu 10% và từ 100 sản phẩm trở lên được chiết khấu 15%.\n"
            "- Hỗ trợ xuất hóa đơn đỏ VAT 8% và dịch vụ thêu/in logo công ty trên Áo Polo Dry Pique, Áo Thun AIRism và Áo Khoác Parka."
        ),
    },
    {
        "gap_id": "gap-click-collect",
        "question": "Có thể đặt online rồi qua cửa hàng nhận (Click & Collect) để khỏi tốn phí ship không?",
        "count": 3,
        "source_type": "resolved_ticket",
        "suggested_filename": "nhan_tai_cua_hang.md",
        "suggested_title": "Dịch vụ Đặt Online Nhận Tại Cửa Hàng (Click & Collect)",
        "suggested_markdown": (
            "# Dịch vụ Đặt Online Nhận Tại Cửa Hàng (Click & Collect)\n\n"
            "## Quy trình nhận hàng tại showroom NORIVA\n"
            "- Khách chọn hình thức Nhận tại cửa hàng (Click & Collect) được miễn phí vận chuyển 100% cho mọi giá trị đơn hàng.\n"
            "- Đơn hàng sẵn sàng tại quầy sau 2 giờ kể từ lúc đặt và được giữ tại cửa hàng tối đa 7 ngày."
        ),
    },
]


@app.get("/api/kb/gaps")
def get_kb_gaps() -> dict[str, Any]:
    resolved = agent.store.list_resolved_gaps()
    # 1) Câu khách hỏi mà AI chưa trả lời được: lọc rác + model (Gemini/Claude...) đánh giá và soạn nháp
    # Câu khách hỏi mà AI chưa trả lời được (+ vài câu mẫu cho shop NORIVA): lọc rác, gộp trùng,
    # model (Gemini/Claude...) đánh giá và soạn nháp, không bịa số liệu.
    seeds = [(g["question"], g["count"]) for g in _SEED_KB_GAPS] if active_shop().is_default else []
    gaps = autofaq.build_gaps(agent.store.list(300), resolved, agent.llm,
                              agent.tools.retriever, agent.tools.products, extra=seeds, store=agent.store)

    return {"gaps": gaps, "resolved_count": len(resolved), "model": agent.llm.name,
            "hidden_by_model": autofaq.dismissed_by_model()}


class DismissGapRequest(BaseModel):
    gap_id: str


@app.post("/api/kb/gaps/dismiss")
def dismiss_kb_gap(req: DismissGapRequest) -> dict[str, Any]:
    """Bỏ qua câu hỏi không đáng thêm vào KB (không ghi file)."""
    agent.store.mark_gap_resolved(req.gap_id)
    return {"ok": True, "gap_id": req.gap_id}


@app.post("/api/kb/gaps/approve")
def approve_kb_gap(req: ApproveGapRequest) -> dict[str, Any]:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(req.filename).stem).strip("_") or "auto_faq"
    target = agent.tools.retriever.kb_dir / f"{safe_name}.md"
    target.write_text(req.content.strip() + "\n", encoding="utf-8")
    agent.tools.retriever.reload()
    agent.store.mark_gap_resolved(req.gap_id)
    return {
        "ok": True,
        "gap_id": req.gap_id,
        "filename": target.name,
        "total_chunks": len(agent.tools.retriever.chunks),
    }


# ---------------- Feature 5: Telegram Bot ----------------
@app.get("/api/integrations/config")
def get_integrations_config() -> dict[str, Any]:
    tok = _INTEGRATIONS_CFG.get("telegram_bot_token") or ""
    return {
        "telegram_configured": bool(tok),
        "telegram_token_masked": (tok[:8] + "..." + tok[-4:]) if len(tok) > 12 else ("Đã cấu hình" if tok else ""),
        "webhooks": {
            "telegram_webhook": "/api/integrations/telegram/webhook",
            "facebook_webhook": "/api/webhook/facebook",
            "zalo_webhook": "/api/webhook/zalo",
        },
    }


@app.post("/api/integrations/config")
def save_integrations_config(req: IntegrationsConfigRequest) -> dict[str, Any]:
    if req.telegram_bot_token is not None:
        _INTEGRATIONS_CFG["telegram_bot_token"] = req.telegram_bot_token.strip()
    return {"ok": True, **get_integrations_config()}


@app.post("/api/integrations/telegram/webhook")
async def telegram_bot_webhook(update: dict[str, Any]) -> dict[str, Any]:
    """Webhook chuẩn Telegram Bot API (Update object)."""
    msg = update.get("message") or update.get("edited_message") or {}
    chat_obj = msg.get("chat") or {}
    chat_id = chat_obj.get("id")
    if not chat_id:
        return {"ok": True, "ignored": True}
    from_user = msg.get("from") or {}
    full_name = " ".join(filter(None, [from_user.get("first_name"), from_user.get("last_name")])) or f"Telegram #{chat_id}"
    text = (msg.get("text") or msg.get("caption") or "").strip()
    img_info = {"name": f"telegram_photo_{chat_id}.jpg"} if msg.get("photo") else None

    sid = f"telegram-{chat_id}"
    _check_rate_limit(sid)
    res = agent.handle(
        sid,
        text,
        image_info=img_info,
        customer={"type": "guest", "guest_code": str(chat_id), "name": full_name, "channel": "Telegram Bot"},
    )
    sess = agent._session(sid)
    sess["name"] = full_name
    sess["channel"] = "Telegram Bot"
    agent.store.save_conversation(sid, sess)
    _OMNI_SESSIONS[sid] = sess
    sent = _send_telegram_message(chat_id, res.answer)
    await _broadcast_ws({"event": "conversation_updated", "session_id": sid, "conversation": sess})
    return {
        "ok": True,
        "session_id": sid,
        "chat_id": chat_id,
        "sender_name": full_name,
        "reply": res.answer,
        "status": res.status,
        "intent": res.intent,
        "sent_to_telegram": sent,
    }


@app.post("/api/integrations/telegram/poll")
async def telegram_poll_updates() -> dict[str, Any]:
    """Kéo tin nhắn mới từ Telegram getUpdates (giúp test Telegram Bot thật ngay trên localhost không cần ngrok)."""
    token = (_INTEGRATIONS_CFG.get("telegram_bot_token") or "").strip()
    if not token:
        raise HTTPException(400, "Chưa cấu hình Telegram Bot Token.")
    offset = int(_INTEGRATIONS_CFG.get("telegram_last_update_id") or 0) + 1
    url = f"https://api.telegram.org/bot{token}/getUpdates?offset={offset}&timeout=1"
    try:
        with urllib.request.urlopen(url, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(502, f"Không kết nối được Telegram API: {exc}") from exc

    processed = []
    for upd in data.get("result") or []:
        uid = upd.get("update_id", 0)
        if uid > int(_INTEGRATIONS_CFG.get("telegram_last_update_id") or 0):
            _INTEGRATIONS_CFG["telegram_last_update_id"] = uid
        r = await telegram_bot_webhook(upd)
        if not r.get("ignored"):
            processed.append(r)
    return {"ok": True, "processed_count": len(processed), "results": processed}


@app.websocket("/ws/crm")
async def ws_crm(ws: WebSocket, token: str | None = None, sid: str | None = None) -> None:
    """Admin: ?token=<token đăng nhập> → nhận mọi hội thoại. Trang bán hàng: ?sid=<session_id> → chỉ phiên đó."""
    is_admin = (not auth.ENABLED) or auth.session(token) is not None
    if not is_admin and not sid:
        await ws.close(code=4401)
        return
    await ws.accept()
    _WS_CLIENTS.append(ws)
    _WS_SID[id(ws)] = None if is_admin else sid
    try:
        if is_admin:
            await ws.send_json({"event": "connected", "active_model": agent.llm.name,
                                "conversations": [_public_conv(c) for c in _all_conversations()]})
        else:
            own = agent.sessions.get(sid) or agent.store.load_conversations().get(sid)
            await ws.send_json({"event": "connected", "session_id": sid, "conversation": _public_conv(own) if own else None})
        while True:
            data = await ws.receive_json()
            if data.get("type") == "ping":
                await ws.send_json({"event": "pong", "ts": time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _WS_CLIENTS:
            _WS_CLIENTS.remove(ws)
        _WS_SID.pop(id(ws), None)


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    if req.trace_id.startswith("csat-"):  # khách chấm sao cuối hội thoại (không gắn với 1 trace)
        agent.store.save_csat(req.trace_id[5:], req.stars or (5 if req.score > 0 else 2))
        return {"ok": True}
    if not agent.store.set_feedback(req.trace_id, req.score):
        raise HTTPException(404, "trace không tồn tại")
    return {"ok": True}


@app.get("/api/traces")
def traces(limit: int = 50) -> list[dict[str, Any]]:
    return agent.store.list(limit)


@app.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str) -> dict[str, Any]:
    t = agent.store.get(trace_id)
    if not t:
        raise HTTPException(404, "trace không tồn tại")
    return t


@app.get("/api/agent/overview")
def agent_overview() -> dict[str, Any]:
    """Dữ liệu thật cho trang "Cách Agent xử lý": model, công cụ, tài liệu, các nhánh và số lượt dùng."""
    from .agent import BRANCHES
    shop = active_shop()
    llm = agent.llm
    tool_use: dict[str, int] = {}
    for t in agent.store.list(500):
        for sp in t.get("spans") or []:
            n = sp.get("name", "")
            if n.startswith("tool:"):
                tool_use[n[5:]] = tool_use.get(n[5:], 0) + 1
    by_intent = agent.store.metrics().get("by_intent", {}) if agent.store.list(1) else {}
    docs = sorted({c.source for c in agent.tools.retriever.chunks})
    branches = []
    for b in BRANCHES:
        if b["id"] == "product" and not shop.size_advice:
            b = {**b, "intents": [i for i in b["intents"] if i != "size_advice"],
                 "tools": [t for t in b["tools"] if t != "recommend_size"], "kb": []}
        kb = docs if b["kb"] == ["*"] else [d for d in b["kb"] if d in docs]
        branches.append({**b, "kb": kb, "tools": [t for t in b["tools"] if t in agent.tools.tools],
                         "count": sum(by_intent.get(i, 0) for i in b["intents"] + b.get("intent_aliases", []))})
    return {
        "shop": shop.name,
        "model": {"name": llm.name, "fallbacks": getattr(llm, "fallback_models", []),
                  "embedding": agent.tools.retriever.embedding_mode},
        "tools": [{**t.spec(), "uses": tool_use.get(n, 0)} for n, t in agent.tools.tools.items()],
        "kb_docs": docs, "kb_chunks": len(agent.tools.retriever.chunks),
        "branches": branches,
        "total_requests": sum(by_intent.values()),
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return agent.store.metrics()


@app.post("/api/kb/reload")
def kb_reload() -> dict[str, Any]:
    agent.tools.retriever.reload()
    return {"chunks": len(agent.tools.retriever.chunks)}


@app.get("/api/kb")
def kb_list() -> dict[str, Any]:
    docs: dict[str, dict[str, Any]] = {}
    for ch in agent.tools.retriever.chunks:
        d = docs.setdefault(ch.source, {
            "source": ch.source,
            "title": ch.title.split(" - ")[0],
            "chunks": [],
        })
        d["chunks"].append({"id": ch.id, "title": ch.title, "text": ch.text})
    return {"documents": list(docs.values()), "total_chunks": len(agent.tools.retriever.chunks)}


@app.get("/api/state")
def system_state() -> dict[str, Any]:
    eval_path = Path(__file__).resolve().parent.parent / "evals" / "report.json"
    eval_report = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None
    return {
        "model": agent.llm.name,
        "orders": [
            {"order_id": k, **{ kk: vv for kk, vv in v.items() if kk not in {"phone", "address"} }}
            for k, v in agent.tools.orders.items()
        ],
        "tickets": list(agent.tools.tickets.values()),
        "tools": agent.tools.list_specs(),
        "eval_report": eval_report,
    }


@app.post("/api/evals/run")
def run_evals_api() -> dict[str, Any]:
    from evals.run_evals import evaluate_case, summarize
    from .observability import TraceStore
    import tempfile, time
    root = Path(__file__).resolve().parent.parent
    cases = [json.loads(l) for l in (root / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tmp_db = Path(tempfile.gettempdir()) / f"omnicare_eval_{time.time_ns()}.db"
    eval_agent = OmniCareAgent(store=TraceStore(tmp_db))
    results = [evaluate_case(eval_agent, c) for c in cases]
    summary = summarize(results)
    payload = {"model": eval_agent.llm.name, "summary": summary, "results": results}
    (root / "evals" / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


@app.post("/mcp")
def mcp(rpc: dict[str, Any]) -> dict[str, Any]:
    """Endpoint MCP tối giản (JSON-RPC 2.0) cho các tool: initialize, tools/list, tools/call."""
    rid, method, params = rpc.get("id"), rpc.get("method"), rpc.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "omnicare-tools", "version": "0.1.0"},
                  "capabilities": {"tools": {}}}
    elif method == "tools/list":
        result = {"tools": agent.tools.list_specs()}
    elif method == "tools/call":
        try:
            out = agent.tools.call(params.get("name", ""), params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}], "isError": False}
        except ToolError as exc:
            result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


@app.post("/api/orders/create")
def create_order(req: CreateOrderRequest) -> dict[str, Any]:
    next_num = 10001 + len(agent.tools.orders)
    while f"OC{next_num}" in agent.tools.orders:
        next_num += 1
    oid = f"OC{next_num}"
    product_text = req.product
    if req.items:
        by_id = {p["id"]: p for p in agent.tools.products}
        parts, subtotal = [], 0
        for it in req.items:
            p = by_id.get(it.id)
            if not p:
                raise HTTPException(400, f"Sản phẩm {it.id} không có trong danh mục")
            subtotal += int(p["price"]) * it.qty
            size = "" if agent.tools.is_free_size(p) or not it.size else f" ({active_shop().variant_label} {it.size})"
            parts.append(f"{p['name']}{size}" + (f" x{it.qty}" if it.qty > 1 else ""))
        product_text = ", ".join(parts)
    else:
        if not product_text:
            raise HTTPException(400, "Thiếu sản phẩm")
        subtotal = req.subtotal if req.subtotal is not None else req.total
    fee = shipping_fee(subtotal)
    now = time.time()
    order_data = {
        "product": product_text,
        "status": "Đang chuẩn bị hàng",
        "carrier": "GHN",
        "tracking": f"GHN{next_num}99",
        "created_at": time.strftime("%d/%m/%Y %H:%M", time.localtime(now)),
        # Chính sách: nội thành 1-2 ngày, tỉnh 3-5 ngày: báo ngày dự kiến theo thời điểm đặt thực tế
        "eta": time.strftime("%d/%m/%Y", time.localtime(now + 3 * 86400)),
        "subtotal": subtotal,
        "shipping_fee": fee,
        "total": subtotal + fee,
        "phone": req.phone,
        "address": req.customer_name,
    }
    agent.tools.orders[oid] = order_data
    agent.store.save_order(oid, order_data)
    return {"order_id": oid, **order_data}


@app.get("/")
@app.get("/index.html")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


@app.get("/widget.js")
def widget_js() -> FileResponse:
    """Mã nhúng chat cho website bất kỳ: <script src="https://<server>/widget.js" async></script>"""
    return FileResponse(STATIC / "widget.js", media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=300", "Access-Control-Allow-Origin": "*"})


@app.get("/widget-demo")
def widget_demo_page() -> FileResponse:
    """Website mẫu của "một shop khác" có dán widget.js, để xem thử mã nhúng."""
    return FileResponse(STATIC / "widget-demo.html", headers=NO_CACHE)


@app.get("/store")
@app.get("/store.html")
def store_page() -> FileResponse:
    return FileResponse(STATIC / "store.html", headers=NO_CACHE)


# ---------------- Đa cửa hàng (shop profiles) ----------------
class ActivateShopRequest(BaseModel):
    shop_id: str = Field(..., pattern=r"^[a-z0-9_-]{1,40}$")


@app.get("/api/shops")
def get_shops() -> dict[str, Any]:
    return {"active": active_shop().id, "shops": list_shops()}


@app.post("/api/shops/active")
def set_active_shop(req: ActivateShopRequest) -> dict[str, Any]:
    """Đổi shop đang demo: nạp hồ sơ mới rồi tạo lại agent (giữ model LLM đang dùng)."""
    global agent
    try:
        prof = activate_shop(req.shop_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    agent = OmniCareAgent(llm=agent.llm)
    _OMNI_SESSIONS.clear()
    autofaq._CACHE.clear()  # cache Auto-FAQ theo shop (mỗi shop có DB riêng)
    return {"ok": True, "shop": prof.public(), "kb_chunks": len(agent.tools.retriever.chunks)}


@app.get("/api/shop")
def get_shop_info() -> dict[str, Any]:
    """Thông tin shop đang bật + danh mục sản phẩm, cho trang bán hàng dùng chung /shop."""
    return {**active_shop().public(), "products": agent.tools.products}


@app.get("/api/shop/orders")
def get_shop_orders(phone: str = "") -> dict[str, Any]:
    """Đơn của một tài khoản demo (khớp theo số điện thoại). Chỉ dùng cho demo, không có xác thực thật."""
    if not re.fullmatch(r"0\d{9}", phone):
        raise HTTPException(400, "Số điện thoại không hợp lệ")
    orders = [{"order_id": oid, **o} for oid, o in agent.tools.orders.items() if o.get("phone") == phone]
    return {"orders": orders}


@app.get("/shop")
@app.get("/shop.html")
def generic_store_page() -> FileResponse:
    return FileResponse(STATIC / "shop.html", headers=NO_CACHE)


if (STATIC / "img").exists():
    app.mount("/img", StaticFiles(directory=STATIC / "img"), name="img")
