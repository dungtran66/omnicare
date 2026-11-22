"""Captain Auto-FAQ: gom câu hỏi AI chưa trả lời được, lọc rác và nhờ model (Gemini/Claude...) soạn FAQ nháp.

Luồng:
1. Lấy trace bị `fallback` / `no_grounding`.
2. Lọc bằng luật: bỏ câu bị guardrail chặn, câu xúc phạm, câu ngoài lề, câu không dính tới mua sắm.
3. Gộp câu trùng (không phân biệt hoa thường, dấu câu) và đếm số lượt hỏi thật.
4. Model đánh giá câu có đáng thành FAQ không và soạn bản nháp Markdown. Model KHÔNG được bịa chính sách:
   thông tin shop chưa có thì để chỗ trống `[Cần bổ sung: ...]` cho nhân viên điền trước khi duyệt.
5. Kết quả model được cache theo câu hỏi để mở lại trang không gọi API lần nữa.
Model lỗi hoặc đang chạy Mock thì dùng bản nháp mẫu có chỗ trống (không bịa nội dung).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import guardrails
from .agent import _SHOP_TOPIC_RE
from .llm import LLMError, MockLLM
from .rag import normalize
from .shop import active as active_shop

MAX_AI_CALLS = 8  # mỗi lần mở trang chỉ gọi model cho tối đa N câu mới (các câu còn lại lần sau)
_CACHE: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()

_PROMPT = """Bạn là trưởng nhóm CSKH của {shop}. Dưới đây là câu hỏi khách đã hỏi mà trợ lý AI chưa trả lời được vì kho kiến thức (KB) thiếu tài liệu.
Nhiệm vụ:
1. Đánh giá câu hỏi có NÊN thêm vào KB không. KHÔNG nên nếu: chửi bới, trêu đùa, câu vô nghĩa, câu hỏi ngoài phạm vi cửa hàng (thời tiết, chính trị, kiến thức chung...), hoặc câu chỉ hỏi về một đơn hàng cụ thể.
   NÊN nếu: khách thật sự cần biết về sản phẩm, dịch vụ, chính sách, mua hàng của shop (kể cả hỏi sản phẩm shop không bán: FAQ sẽ nói rõ shop bán gì).
2. Nếu nên, soạn tài liệu Markdown ngắn để nạp vào KB:
   - Dòng đầu "# <tiêu đề>", sau đó "## <mục>" và 2-4 gạch đầu dòng, viết tự nhiên như chính sách của shop.
   - CHỈ dùng sự thật có trong NGỮ CẢNH (danh mục sản phẩm, danh sách tài liệu). Không bịa giá, phần trăm, thời hạn, địa chỉ, số điện thoại.
   - Chỗ nào shop cần tự quyết định thì ghi "[Cần bổ sung: ...]" để nhân viên điền.
Trả về DUY NHẤT một JSON:
{{"relevant": true/false, "reason": "lý do ngắn", "title": "tiêu đề", "filename": "ten_file_khong_dau", "markdown": "nội dung"}}

CÂU HỎI CỦA KHÁCH: "{question}\""""


def _gap_id(q: str) -> str:
    return "gap-" + hashlib.md5(normalize(q).encode()).hexdigest()[:12]


def _key(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize(q)).strip()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize(text)).strip("_")[:40] or "faq_moi"


def _is_candidate(q: str) -> bool:
    """Luật lọc nhanh trước khi tốn lượt gọi model."""
    if len(q) < 6:
        return False
    g = guardrails.check_input(q)
    if g.action != "allow" or "abusive_language" in g.flags:
        return False
    return bool(re.search(_SHOP_TOPIC_RE, normalize(q)))


def _template(q: str) -> dict[str, Any]:
    shop = active_shop().name
    return {
        "relevant": True, "ai": False, "reason": "Bản nháp mẫu (model chưa sẵn sàng)",
        "title": f"Giải đáp: {q[:60]}", "filename": f"faq_{_slug(q)}",
        "markdown": (f"# Giải đáp: {q[:80]}\n\n## Trả lời\n"
                     f"- [Cần bổ sung: câu trả lời chính thức của {shop} cho câu hỏi này]\n"
                     f"- [Cần bổ sung: điều kiện áp dụng, nếu có]"),
    }


def _context(retriever, products) -> dict[str, Any]:
    titles = sorted({c.title.split(" - ")[0] for c in retriever.chunks})
    catalog = [{"name": p["name"], "category": p.get("category", ""), "price": p.get("price")} for p in products]
    return {"tool_results": [
        {"tool": "danh_muc_san_pham", "result": catalog},
        {"tool": "tai_lieu_da_co_trong_kb", "result": titles},
    ]}


def _ask_model(llm, q: str, ctx: dict[str, Any]) -> dict[str, Any]:
    if isinstance(llm, MockLLM):
        return _template(q)
    try:
        res = llm.complete(_PROMPT.format(shop=active_shop().intro, question=q), q, ctx)
        m = re.search(r"\{.*\}", res.text, re.S)
        data = json.loads(m.group(0)) if m else {}
    except (LLMError, ValueError, AttributeError):
        return _template(q)
    if not isinstance(data, dict) or "relevant" not in data:
        return _template(q)
    md = str(data.get("markdown") or "").strip()
    if data.get("relevant") and not md.startswith("#"):
        return _template(q)
    return {
        "relevant": bool(data.get("relevant")), "ai": True, "reason": str(data.get("reason") or "")[:200],
        "title": str(data.get("title") or q)[:120], "filename": _slug(str(data.get("filename") or q)),
        "markdown": md,
    }


def build_gaps(traces: list[dict[str, Any]], resolved: set[str], llm, retriever, products,
               extra: list[tuple[str, int]] | None = None, store=None) -> list[dict[str, Any]]:
    """`extra`: câu hỏi mẫu (câu, số lượt) đưa qua cùng quy trình để model soạn nháp, không dùng nội dung soạn sẵn."""
    groups: dict[str, dict[str, Any]] = {_key(q): {"question": q, "count": n} for q, n in (extra or [])}
    for t in traces:
        flags = t.get("guardrail_flags") or []
        if t.get("status") != "fallback" and "no_grounding" not in flags:
            continue
        q = (t.get("user_input") or "").strip()
        if not _is_candidate(q):
            continue
        k = _key(q)
        if k in groups:
            groups[k]["count"] += 1
        else:
            groups[k] = {"question": q, "count": 1}

    if store is not None and not _CACHE:
        _CACHE.update(store.load_autofaq_cache())  # nạp kết quả đã lưu trong SQLite (sau khi khởi động lại)
    todo = [g for g in groups.values() if _gap_id(g["question"]) not in resolved]
    pending = [g["question"] for g in todo if _key(g["question"]) not in _CACHE][:MAX_AI_CALLS]
    if pending:
        ctx = _context(retriever, products)
        with ThreadPoolExecutor(max_workers=4) as ex:
            for q, r in zip(pending, ex.map(lambda q: _ask_model(llm, q, ctx), pending)):
                if r.get("ai"):  # chỉ cache kết quả thật của model; bản mẫu sẽ thử lại lần sau
                    with _LOCK:
                        _CACHE[_key(q)] = r
                    if store is not None:
                        store.save_autofaq(_key(q), r, getattr(llm, "name", ""))

    gaps = []
    for g in sorted(todo, key=lambda g: -g["count"]):
        q = g["question"]
        r = _CACHE.get(_key(q)) or _template(q)
        if not r["relevant"]:
            continue
        gaps.append({
            "gap_id": _gap_id(q), "question": q, "count": g["count"], "source_type": "fallback_trace",
            "ai_drafted": r["ai"], "reason": r["reason"],
            "suggested_filename": f"{r['filename']}.md", "suggested_title": r["title"],
            "suggested_markdown": r["markdown"],
        })
    return gaps


def dismissed_by_model() -> int:
    return sum(1 for r in _CACHE.values() if not r["relevant"])
