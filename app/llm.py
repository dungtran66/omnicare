"""Tầng model: một interface chung `LLM` với 2 implementation.

* `OpenAICompatibleLLM`: gọi bất kỳ API tương thích OpenAI Chat Completions (OpenAI, Groq, OpenRouter, Ollama, vLLM...).
* `MockLLM`: chạy offline, sinh câu trả lời có căn cứ từ ngữ cảnh — để demo, test và chạy evals không cần API key.
"""
from __future__ import annotations

import contextvars
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from . import config
from .rag import normalize, tokenize
from .shop import active as active_shop


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class LLMError(Exception):
    pass


# ---- Giới hạn lượt gọi model cho mỗi khách / ngày (link demo công khai dùng chung một API key) ----
# agent.handle() đặt QUOTA_KEY = mã phiên của khách; các lệnh gọi từ Admin (Copilot, báo cáo...) không đặt nên không bị giới hạn.
QUOTA_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar("omnicare_quota_key", default=None)
_QUOTA: dict[tuple[str, str], int] = {}
_QUOTA_LOCK = threading.Lock()


class QuotaExceeded(LLMError):
    pass


def quota_take() -> None:
    """Trừ 1 lượt gọi model của khách hiện tại; hết lượt thì báo lỗi để agent rơi về luật / câu dự phòng."""
    key, limit = QUOTA_KEY.get(), config.MODEL_CALLS_PER_DAY
    if not key or limit <= 0:
        return
    day = time.strftime("%Y-%m-%d")
    with _QUOTA_LOCK:
        n = _QUOTA.get((key, day), 0)
        if n >= limit:
            raise QuotaExceeded(f"Khách {key} đã dùng hết {limit} lượt gọi model hôm nay")
        _QUOTA[(key, day)] = n + 1


def quota_used(key: str) -> int:
    return _QUOTA.get((key, time.strftime("%Y-%m-%d")), 0)


class LLM:
    name = "base"
    supports_tools = False   # gọi công cụ kiểu function calling
    supports_vision = False  # đọc ảnh

    def complete(self, system: str, user: str, context: dict[str, Any], json_mode: bool = False) -> LLMResult:
        raise NotImplementedError


def render_context(context: dict[str, Any]) -> str:
    parts = []
    # Tài liệu RAG bọc trong thẻ XML: Claude và các model khác tách nguồn rõ ràng, dễ trích dẫn hơn
    docs = context.get("documents", [])
    if docs:
        body = "\n".join(
            f'<document index="{i}" source="{doc.get("source", "")}" title="{doc["title"]}">\n{doc["text"]}\n</document>'
            for i, doc in enumerate(docs, 1)
        )
        parts.append(f"<documents>\n{body}\n</documents>\nChỉ trả lời dựa trên các tài liệu trên; nếu tài liệu không nói tới thì nói là chưa có thông tin.")
    for tr in context.get("tool_results", []):
        parts.append(f"[Kết quả công cụ {tr['tool']}]\n{json.dumps(tr['result'], ensure_ascii=False)}")
    if context.get("customer_memory"):
        parts.append("[Điều đã biết về khách từ các lần trước - dùng để cá nhân hóa, không cần nhắc lại nguyên văn]\n"
                     + json.dumps(context["customer_memory"], ensure_ascii=False))
    if context.get("history"):
        hist = "\n".join(f"{m['role']}: {m['content']}" for m in context["history"])
        parts.append(f"[Lịch sử hội thoại gần nhất]\n{hist}")
    return "\n\n".join(parts) or "(không có ngữ cảnh)"


# Số yêu cầu tới model chạy cùng lúc trong 1 tiến trình (proxy codex.fosta.app cho tối đa 4/tài khoản).
_INFLIGHT = threading.BoundedSemaphore(max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "3"))))
_RETRY_429 = (1.5, 4.0)  # giây chờ trước lần thử lại thứ 1, 2 khi bị 429


class OpenAICompatibleLLM(LLM):
    name = "openai"
    supports_tools = True
    supports_vision = True

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30,
                 fallback_models: list[str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # Model dự phòng (LLM_FALLBACK_MODELS): thử lần lượt khi model chính lỗi/quá tải/quá chậm.
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model]

    def _post(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Gọi /chat/completions. Giới hạn số yêu cầu chạy cùng lúc (LLM_MAX_CONCURRENCY) để không tự gây lỗi 429;
        gặp 429 thì chờ ngắn rồi thử lại cùng model (model dự phòng thường dùng chung tài khoản nên chuyển cũng vô ích)."""
        quota_take()
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        for attempt in range(len(_RETRY_429) + 1):
            if not _INFLIGHT.acquire(timeout=timeout):
                raise LLMError("Lỗi gọi model: hàng chờ gọi model quá lâu")
            try:
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == len(_RETRY_429):
                    raise LLMError(f"Lỗi gọi model: {exc}") from exc
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                raise LLMError(f"Lỗi gọi model: {exc}") from exc
            finally:
                _INFLIGHT.release()
            time.sleep(_RETRY_429[attempt])
        raise LLMError("Lỗi gọi model: quá nhiều yêu cầu")

    def _models(self):
        for i, mdl in enumerate([self.model, *self.fallback_models]):
            yield mdl, (config.LLM_PRIMARY_TIMEOUT if i == 0 and self.fallback_models and config.LLM_PRIMARY_TIMEOUT
                        else self.timeout)

    def run_tools(self, system: str, user: str, tools: list[dict[str, Any]],
                  execute: Callable[[str, dict[str, Any]], dict[str, Any]], max_steps: int = 4) -> LLMResult:
        """Vòng lặp agent: model tự chọn công cụ -> code chạy công cụ -> trả kết quả cho model -> ... -> câu trả lời.

        `execute(name, args)` do agent cung cấp (kiểm tra quyền, ghi trace). Model lỗi thì thử model dự phòng từ đầu
        (các công cụ cho model dùng ở đây chỉ đọc dữ liệu nên chạy lại không có tác dụng phụ)."""
        errors: list[str] = []
        for mdl, timeout in self._models():
            msgs: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            pt = ct = 0
            try:
                for step in range(max_steps + 1):
                    payload: dict[str, Any] = {"model": mdl, "temperature": 0.2, "messages": msgs}
                    if step < max_steps:
                        payload["tools"] = tools
                    data = self._post(payload, timeout if step == 0 else self.timeout)
                    u = data.get("usage") or {}
                    pt += u.get("prompt_tokens", 0)
                    ct += u.get("completion_tokens", 0)
                    msg = data["choices"][0]["message"]
                    calls = msg.get("tool_calls") or []
                    if not calls:
                        return LLMResult(text=msg.get("content") or "", prompt_tokens=pt, completion_tokens=ct,
                                         model=data.get("model", mdl))
                    msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                    for c in calls:
                        fn = c.get("function") or {}
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        result = execute(fn.get("name", ""), args if isinstance(args, dict) else {})
                        msgs.append({"role": "tool", "tool_call_id": c.get("id", ""),
                                     "content": json.dumps(result, ensure_ascii=False)[:6000]})
                raise LLMError("Model gọi công cụ quá nhiều bước")
            except QuotaExceeded:
                raise
            except LLMError as exc:
                errors.append(f"{mdl}: {exc}")
            except (KeyError, IndexError, TypeError) as exc:
                errors.append(f"{mdl}: phản hồi không hợp lệ ({exc})")
        raise LLMError("; ".join(errors) or "Không gọi được model")

    def see(self, system: str, prompt: str, image_url: str, json_mode: bool = True) -> LLMResult:
        """Gửi ảnh (data URL jpeg/png/webp/gif hoặc http) cho model đọc."""
        errors: list[str] = []
        for mdl, timeout in self._models():
            payload: dict[str, Any] = {"model": mdl, "temperature": 0.1, "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": prompt},
                                             {"type": "image_url", "image_url": {"url": image_url}}]}]}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                data = self._post(payload, max(timeout, 20))
                u = data.get("usage") or {}
                return LLMResult(text=data["choices"][0]["message"]["content"] or "",
                                 prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                                 model=data.get("model", mdl))
            except QuotaExceeded:
                raise
            except LLMError as exc:
                errors.append(f"{mdl}: {exc}")
            except (KeyError, IndexError, TypeError) as exc:
                errors.append(f"{mdl}: {exc}")
        raise LLMError("; ".join(errors))

    def complete(self, system: str, user: str, context: dict[str, Any], json_mode: bool = False) -> LLMResult:
        errors = []
        for i, mdl in enumerate([self.model, *self.fallback_models]):
            # Có model dự phòng thì chỉ chờ model chính tối đa LLM_PRIMARY_TIMEOUT giây rồi chuyển.
            timeout = config.LLM_PRIMARY_TIMEOUT if i == 0 and self.fallback_models and config.LLM_PRIMARY_TIMEOUT else self.timeout
            try:
                return self._complete_one(mdl, system, user, context, json_mode, timeout)
            except QuotaExceeded:
                raise
            except LLMError as exc:
                errors.append(f"{mdl}: {exc}")
        raise LLMError("; ".join(errors))

    def _complete_one(self, model: str, system: str, user: str, context: dict[str, Any], json_mode: bool,
                      timeout: float | None = None) -> LLMResult:
        payload: dict[str, Any] = {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"NGỮ CẢNH:\n{render_context(context)}\n\nCÂU HỎI CỦA KHÁCH:\n{user}"},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(payload, timeout or self.timeout)
        usage = data.get("usage", {})
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Phản hồi không hợp lệ: {exc}") from exc
        return LLMResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=data.get("model", model),
        )


class AnthropicLLM(LLM):
    """Claude qua SDK chính thức `anthropic` (Messages API).

    * Khóa API: tham số `api_key`, hoặc để trống để SDK tự đọc `ANTHROPIC_API_KEY` / hồ sơ `ant auth login`.
    * Bật `fallbacks: "default"`: nếu Claude từ chối vì bộ lọc an toàn, API tự chạy lại trên model dự phòng.
    * `json_mode` (phân loại ý định) dùng structured outputs nên luôn trả JSON hợp lệ.
    * System prompt được cache (prompt caching) để giảm chi phí khi hội thoại lặp lại.
    """

    name = "anthropic"
    _FALLBACK_BETA = "server-side-fallback-2026-07-01"
    _INTENT_SCHEMA = {
        "type": "object",
        "properties": {"intent": {"type": "string"}},
        "required": ["intent"],
        "additionalProperties": False,
    }

    def __init__(self, model: str, api_key: str = "", timeout: float = 30, effort: str = "medium",
                 max_tokens: int = 2048):
        import anthropic  # import muộn: chỉ cần khi dùng Claude

        self._anthropic = anthropic
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": 2}
        if api_key:
            kwargs["api_key"] = api_key
        try:
            self.client = anthropic.Anthropic(**kwargs)
        except Exception as exc:  # thiếu khóa khi khởi tạo -> báo lỗi lúc gọi, agent sẽ tự fallback
            self.client = None
            self._init_error = str(exc)

    def complete(self, system: str, user: str, context: dict[str, Any], json_mode: bool = False) -> LLMResult:
        if self.client is None:
            raise LLMError(f"Chưa cấu hình Claude: {self._init_error}")
        quota_take()
        if json_mode:
            content = user
            output_config: dict[str, Any] = {
                "effort": "low",  # phân loại ý định là việc đơn giản
                "format": {"type": "json_schema", "schema": self._INTENT_SCHEMA},
            }
            max_tokens = 1024
        else:
            content = f"NGỮ CẢNH:\n{render_context(context)}\n\nCÂU HỎI CỦA KHÁCH:\n{user}"
            output_config = {"effort": self.effort}
            max_tokens = self.max_tokens
        a = self._anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                output_config=output_config,
                betas=[self._FALLBACK_BETA],
                fallbacks="default",
            )
        except a.AuthenticationError as exc:
            raise LLMError("Khóa API Claude không hợp lệ") from exc
        except a.RateLimitError as exc:
            raise LLMError("Claude đang giới hạn tốc độ (429)") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"Lỗi API Claude {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise LLMError(f"Không kết nối được Claude: {exc}") from exc
        except TypeError as exc:  # SDK báo thiếu khóa (không có ANTHROPIC_API_KEY / hồ sơ đăng nhập)
            if "authentication" in str(exc).lower():
                raise LLMError("Chưa có khóa API Claude (đặt ANTHROPIC_API_KEY)") from exc
            raise
        if resp.stop_reason == "refusal":
            raise LLMError("Claude từ chối trả lời yêu cầu này")
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        prompt_tokens = (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
            + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        return LLMResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=u.output_tokens or 0,
                         model=resp.model)


class MockLLM(LLM):
    """Model giả lập có tính quyết định (deterministic). Trả lời bằng cách trích câu liên quan nhất từ ngữ cảnh."""

    name = "mock"

    def complete(self, system: str, user: str, context: dict[str, Any], json_mode: bool = False) -> LLMResult:
        if json_mode:
            text = json.dumps({"intent": context.get("rule_intent", "knowledge")})
        else:
            text = self._answer(user, context)
        return LLMResult(text=text, prompt_tokens=len((system + user + render_context(context)).split()),
                         completion_tokens=len(text.split()), model="mock-1")

    @staticmethod
    def _answer(user: str, context: dict[str, Any]) -> str:
        for tr in context.get("tool_results", []):
            r = tr["result"]
            if tr["tool"] == "get_order_status":
                if not r.get("found"):
                    return (f"Mình chưa tìm thấy đơn hàng {r['order_id']}. Bạn vui lòng kiểm tra lại mã đơn "
                            "(bắt đầu bằng OC) trong email xác nhận nhé.")
                msg = f"Đơn hàng {r['order_id']} ({r.get('product', '')}) hiện đang ở trạng thái: {r['status']}."
                if r.get("carrier"):
                    msg += f" Đơn vị vận chuyển: {r['carrier']}, mã vận đơn {r.get('tracking', '')}."
                if r.get("eta"):
                    msg += f" Dự kiến giao: {r['eta']}."
                return msg
            if tr["tool"] == "create_support_ticket":
                return (f"Mình đã tạo phiếu hỗ trợ {r['ticket_id']} cho bạn. Bộ phận chuyên trách sẽ liên hệ "
                        f"trong vòng {r['sla_hours']} giờ.")
            if tr["tool"] == "handoff_to_human":
                return ("Mình đã chuyển cuộc trò chuyện tới nhân viên chăm sóc khách hàng. "
                        f"Thời gian chờ dự kiến khoảng {r['eta_minutes']} phút, bạn vui lòng giữ kết nối nhé.")
            if tr["tool"] == "search_products":
                prods = r.get("products", [])
                if not prods:
                    return (f"Xin lỗi, mình chưa tìm thấy sản phẩm phù hợp trong danh mục {active_shop().name}. "
                            "Bạn có thể cho mình biết rõ tên sản phẩm bạn đang tìm không?")
                p = prods[0]
                price_str = f"{p['price']:,}".replace(",", ".") + "đ"
                sizes_ok = [k for k, v in (p.get("sizes") or {}).items() if v > 0]
                msg = f"{p['name']} hiện có giá {price_str}."
                if p.get("requested_size"):
                    st = p.get("requested_size_stock", 0)
                    msg += (f" {active_shop().variant_label} {p['requested_size']} hiện còn {st} sản phẩm."
                            if st > 0 else f" {active_shop().variant_label} {p['requested_size']} hiện tạm hết hàng.")
                elif sizes_ok:
                    msg += f" Các {active_shop().variant_label.lower()} còn hàng: {', '.join(sizes_ok)}."
                if p.get("colors"):
                    msg += f" Màu sắc: {', '.join(p['colors'][:3])}."
                return msg
            if tr["tool"] == "cancel_order":
                if r.get("cancelled"):
                    return f"Đơn hàng {r['order_id']} ({r.get('product', '')}) đã được hủy thành công theo xác nhận của bạn."
                return r.get("message") or f"Không thể hủy đơn hàng {r.get('order_id', '')}."
            if tr["tool"] == "check_inventory":
                if not r.get("found"):
                    return f"Xin lỗi, mình chưa tìm thấy sản phẩm '{r.get('product', '')}' trong kho."
                price_str = f"{r['price']:,}".replace(",", ".") + "đ"
                if r.get("requested_size"):
                    st = r.get("stock", 0)
                    return (f"Sản phẩm {r['name']} (giá {price_str}) {active_shop().variant_label} {r['requested_size']} "
                            f"{f'hiện còn {st} sản phẩm trong kho.' if st > 0 else 'hiện đang tạm hết hàng.'}")
                return f"Sản phẩm {r['name']} (giá {price_str}) hiện còn tổng cộng {r.get('stock', 0)} sản phẩm."
            if tr["tool"] == "update_shipping_address":
                if r.get("updated"):
                    return (f"Đã cập nhật địa chỉ giao hàng mới cho đơn {r['order_id']} thành: "
                            f"\"{r['new_address']}\".")
                return r.get("message") or f"Không thể đổi địa chỉ cho đơn {r.get('order_id', '')}."
            if tr["tool"] == "request_refund":
                if r.get("refund_requested"):
                    amt = f"{r.get('amount', 0):,}".replace(",", ".") + "đ"
                    return (f"Đã tạo yêu cầu hoàn tiền (mã phiếu {r['ticket_id']}) cho đơn {r['order_id']} "
                            f"(số tiền {amt}). Bộ phận tài chính sẽ xử lý trong vòng {r['sla_hours']} giờ.")
                return r.get("message") or f"Không thể gửi yêu cầu hoàn tiền cho đơn {r.get('order_id', '')}."

        docs = context.get("documents", [])
        if not docs:
            return ("Xin lỗi, mình chưa có thông tin chính xác cho câu hỏi này. "
                    "Bạn có muốn mình chuyển tới nhân viên hỗ trợ không?")
        q = set(tokenize(user))
        sentences = []
        for d_rank, doc in enumerate(docs):
            for s in re.split(r"(?<=[.!?])\s+|\n+", doc["text"]):
                s = s.strip(" -*")
                if len(s) < 15:
                    continue
                overlap = len(q & set(tokenize(s)))
                sentences.append((overlap - d_rank * 0.5, s))
        sentences.sort(key=lambda x: -x[0])
        best = [s for score, s in sentences[:2] if score > 0] or [sentences[0][1]] if sentences else []
        return " ".join(best)


def build_llm(provider: str | None = None, base_url: str | None = None,
              api_key: str | None = None, model: str | None = None) -> LLM:
    from .observability import set_active_model

    llm = _build_llm(provider, base_url, api_key, model)
    set_active_model(llm.name)
    return llm


def _build_llm(provider: str | None = None, base_url: str | None = None,
               api_key: str | None = None, model: str | None = None) -> LLM:
    prov = (provider or config.LLM_PROVIDER).lower()
    burl = base_url or config.LLM_BASE_URL
    key = api_key if api_key is not None else config.LLM_API_KEY
    mdl = model or config.LLM_MODEL
    if prov in ("anthropic", "claude"):
        mdl = mdl if mdl.startswith("claude") else config.CLAUDE_DEFAULT_MODEL
        llm = AnthropicLLM(mdl, key or config.ANTHROPIC_API_KEY, config.LLM_TIMEOUT, config.LLM_EFFORT)
        llm.name = f"anthropic:{mdl}"
        return llm
    if prov in ("openai", "gemini", "groq", "openrouter") and key:
        llm = OpenAICompatibleLLM(burl, key, mdl, config.LLM_TIMEOUT, config.LLM_FALLBACK_MODELS)
        llm.name = f"{prov}:{mdl}"
        return llm
    if prov == "ollama":
        base = burl if "11434" in burl else os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        llm = OpenAICompatibleLLM(base, "ollama", mdl, config.LLM_TIMEOUT)
        llm.name = f"ollama:{mdl}"
        return llm
    return MockLLM()


__all__ = ["LLM", "LLMResult", "LLMError", "MockLLM", "OpenAICompatibleLLM", "AnthropicLLM", "build_llm", "normalize"]
