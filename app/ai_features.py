"""Các tính năng tận dụng model (Gemini/Claude/OpenAI...) cho nhân viên và quản lý.

* `summarize_handoff`: tóm tắt hội thoại khi chuyển nhân viên (khách là ai, vấn đề, đã làm gì, cảm xúc, việc nên làm).
* `copilot`: soạn 3 câu trả lời gợi ý cho nhân viên dựa trên hội thoại + tài liệu + đơn hàng; `rewrite` viết lại câu.
* `pick_outfit`: Stylist AI phối đồ từ danh mục thật (chỉ chọn sản phẩm còn hàng, không bịa giá).
* `weekly_report`: báo cáo tuần do model viết từ trace thật.
Mọi hàm đều có đường lui khi model lỗi hoặc đang chạy Mock: dùng luật đơn giản, không bịa nội dung.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

from . import guardrails
from .llm import LLMError, MockLLM
from .rag import normalize
from .shop import active as active_shop


# ---------------------------------------------------------------- tiện ích
def _ask_json(llm, system: str, user: str, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Gọi model, lấy JSON đầu tiên trong câu trả lời. Trả None nếu Mock/lỗi/không phải JSON."""
    if llm is None or isinstance(llm, MockLLM):
        return None
    try:
        res = llm.complete(system, user, context or {})
        m = re.search(r"\{.*\}", res.text, re.S)
        data = json.loads(m.group(0)) if m else None
        return data if isinstance(data, dict) else None
    except (LLMError, ValueError, AttributeError):
        return None


def _transcript(messages: list[dict[str, Any]], limit: int = 20) -> str:
    role = {"visitor": "Khách", "user": "Khách", "agent": "AI", "assistant": "AI", "human": "Nhân viên", "staff": "Nhân viên"}
    lines = []
    for m in messages[-limit:]:
        r = role.get(m.get("role", ""), None)
        t = (m.get("text") or m.get("content") or "").strip()
        if r and t:
            lines.append(f"{r}: {t[:400]}")
    return "\n".join(lines)


def _visitor_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [(m.get("text") or m.get("content") or "") for m in messages if m.get("role") in ("visitor", "user")]


# ---------------------------------------------------------------- 1) Tóm tắt khi chuyển nhân viên
_SUMMARY_PROMPT = """Bạn là trưởng ca CSKH của {shop}. Đọc hội thoại giữa khách và trợ lý AI, viết bản bàn giao cho nhân viên sắp tiếp nhận.
Trả về DUY NHẤT JSON:
{{"customer": "khách là ai, đang hỏi về đơn/sản phẩm nào (1 câu)",
  "issue": "vấn đề chính của khách (1 câu)",
  "done": "AI đã làm/trả lời gì (1 câu)",
  "mood": "binh_thuong | buc_xuc | rat_buc_xuc",
  "urgency": "thap | trung_binh | cao",
  "next_step": "việc nhân viên nên làm ngay (1 câu)"}}
Chỉ dùng thông tin có trong hội thoại, không đoán thêm."""


def summarize_handoff(llm, messages: list[dict[str, Any]], customer: dict[str, Any] | None = None) -> dict[str, Any]:
    texts = _visitor_texts(messages)
    if not texts:
        return {}
    cust = customer or {}
    who = cust.get("name") or (f"Khách #{cust.get('guest_code')}" if cust.get("guest_code") else "Khách")
    data = _ask_json(llm, _SUMMARY_PROMPT.format(shop=active_shop().name), _transcript(messages),
                     {"tool_results": [{"tool": "thong_tin_khach", "result": cust}]} if cust else None)
    if data and data.get("issue"):
        out = {k: str(data.get(k) or "").strip()[:300] for k in ("customer", "issue", "done", "mood", "urgency", "next_step")}
        out["ai"] = True
    else:
        senti = guardrails.analyze_sentiment(" ".join(texts[-3:]))
        orders = sorted(set(re.findall(r"\bOC\d{5}\b", " ".join(texts), re.I)))
        out = {
            "customer": who + (f", nhắc tới đơn {', '.join(orders)}" if orders else ""),
            "issue": texts[-1][:200],
            "done": f"Khách đã nhắn {len(texts)} tin, AI đã trả lời tự động.",
            "mood": "buc_xuc" if senti["label"] == "negative" else "binh_thuong",
            "urgency": "cao" if senti["label"] == "negative" else "trung_binh",
            "next_step": "Đọc lại tin cuối của khách và phản hồi trong 5 phút.",
            "ai": False,
        }
    out["at"] = time.time()
    return out


# ---------------------------------------------------------------- 1b) Chấm chất lượng hội thoại (khi nhân viên bấm Đóng)
_QA_PROMPT = """Bạn là trưởng nhóm CSKH của {shop}, chấm chất lượng một cuộc hội thoại đã kết thúc giữa khách với trợ lý AI và nhân viên.
NGỮ CẢNH là tài liệu chính sách và danh mục sản phẩm (giá, size, tồn kho, màu) để đối chiếu xem câu trả lời có đúng không.
Trả về DUY NHẤT JSON:
{{"score": 1-5, "resolved": true|false, "accuracy": 1-5, "tone": 1-5,
 "issues": ["tối đa 3 vấn đề cụ thể (vd: AI nói sai phí ship, bỏ sót câu hỏi về size, chuyển nhân viên chậm); không có thì []"],
 "suggestion": "1 câu: nên cải thiện gì (bổ sung tài liệu, sửa câu trả lời mẫu, hướng dẫn nhân viên...)",
 "summary": "1 câu tóm tắt khách cần gì và kết quả"}}
Chấm công bằng theo nội dung thật; khách chỉ hỏi và được trả lời đúng là 5 điểm."""


def score_conversation(llm, messages: list[dict[str, Any]], documents: list[dict[str, Any]] | None = None,
                       catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    texts = _visitor_texts(messages)
    if not texts:
        return {}
    data = _ask_json(llm, _QA_PROMPT.format(shop=active_shop().name), _transcript(messages, 30),
                     {"documents": documents or [],
                      "tool_results": [{"tool": "danh_muc_san_pham_va_ton_kho", "result": catalog}] if catalog else []})
    if data and data.get("score") is not None:
        def num(k):
            try:
                return max(1, min(5, int(round(float(data.get(k) or 0)))))
            except (TypeError, ValueError):
                return None
        out = {"score": num("score"), "resolved": bool(data.get("resolved")), "accuracy": num("accuracy"),
               "tone": num("tone"), "issues": [str(x)[:160] for x in (data.get("issues") or [])][:3],
               "suggestion": str(data.get("suggestion") or "")[:240], "summary": str(data.get("summary") or "")[:240],
               "ai": True}
    else:  # không có model: chấm theo trạng thái các câu trả lời của AI
        bad = [m for m in messages if m.get("role") == "agent" and m.get("status") in ("fallback", "blocked", "denied")]
        handed = any(m.get("status") == "handoff" for m in messages)
        score = max(1, 5 - 2 * len(bad) - (1 if handed else 0))
        out = {"score": score, "resolved": not bad, "accuracy": None, "tone": None,
               "issues": ([f"{len(bad)} câu AI không trả lời được"] if bad else []) + (["Khách phải chuyển nhân viên"] if handed else []),
               "suggestion": "Bổ sung tài liệu cho các câu AI chưa trả lời được." if bad else "",
               "summary": texts[-1][:160], "ai": False}
    out["at"] = time.time()
    return out


# ---------------------------------------------------------------- 2) Copilot cho nhân viên
_COPILOT_PROMPT = """Bạn hỗ trợ nhân viên CSKH của {shop} trả lời khách. Dựa vào hội thoại và NGỮ CẢNH (tài liệu chính sách, đơn hàng),
soạn 3 câu trả lời khác nhau để nhân viên chọn và gửi: (1) trả lời thẳng vấn đề, (2) đồng cảm/xin lỗi nếu khách không vui, hoặc hỏi thêm thông tin còn thiếu, (3) đề xuất bước tiếp theo.
Quy tắc: xưng "em", gọi khách "anh/chị"; tự nhiên như người thật nhắn tin; mỗi câu tối đa 3 câu ngắn; chỉ dùng thông tin trong NGỮ CẢNH,
không hứa hoàn tiền/bồi thường vượt chính sách; không ghi nguồn tài liệu.
Trả về DUY NHẤT JSON: {{"summary": "tóm tắt tình huống 1 câu", "suggestions": ["...", "...", "..."]}}"""


def copilot(llm, messages: list[dict[str, Any]], kb_results: list[dict[str, Any]], orders: list[dict[str, Any]],
            fallback: dict[str, Any]) -> dict[str, Any]:
    ctx = {"documents": kb_results[:3], "tool_results": [{"tool": "don_hang", "result": o} for o in orders[:2]]}
    data = _ask_json(llm, _COPILOT_PROMPT.format(shop=active_shop().name), _transcript(messages), ctx)
    sugg = [str(s).strip() for s in (data or {}).get("suggestions", []) if str(s).strip()]
    if len(sugg) < 2:
        return {**fallback, "ai": False}
    safe = [s for s in sugg if guardrails.check_output(s).allowed][:3]
    return {**fallback, "summary": str(data.get("summary") or fallback.get("summary", ""))[:300],
            "suggestions": safe or fallback.get("suggestions", []), "ai": True}


_REWRITE_STYLES = {
    "polite": "lịch sự, nhã nhặn hơn",
    "short": "ngắn gọn hơn, giữ đủ ý",
    "apology": "xin lỗi khéo léo, thể hiện đồng cảm với khách",
    "friendly": "thân thiện, gần gũi hơn",
}


def rewrite(llm, text: str, style: str) -> dict[str, Any]:
    goal = _REWRITE_STYLES.get(style, _REWRITE_STYLES["polite"])
    prompt = (f"Viết lại tin nhắn CSKH sau cho {goal}. Giữ nguyên mọi con số, mã đơn, chính sách; không thêm cam kết mới. "
              'Trả về DUY NHẤT JSON: {"text": "..."}')
    data = _ask_json(llm, prompt, text)
    out = str((data or {}).get("text") or "").strip()
    if not out or not guardrails.check_output(out).allowed:
        return {"text": text, "ai": False}
    return {"text": out, "ai": True}


# ---------------------------------------------------------------- 3) Stylist AI phối đồ
OUTFIT_RE = re.compile(
    r"\b(phoi (do|ao|quan|sao|the nao|cho)|mac gi|nen mac|mac sao|mac the nao|outfit|set do|bo do|"
    r"di (bien|bien|du lich|choi|lam|tiec|phong van|hen ho|cong so|leo nui|da lat|sapa|picnic)|"
    r"chon do|tu van (do|trang phuc|cach mac))\b")

_OUTFIT_PROMPT = """Bạn là stylist của {shop}. Khách cần tư vấn trang phục. Chọn từ DANH MỤC (chỉ dùng id có trong danh mục, chỉ chọn sản phẩm còn hàng)
một bộ 2-4 món hợp với nhu cầu của khách (dịp, thời tiết, phong cách). Không chọn 2 món cùng vai trò (2 quần, 2 áo khoác).
Trả về DUY NHẤT JSON:
{{"items": [{{"id": "p2", "why": "lý do ngắn"}}],
  "answer": "lời tư vấn 2-4 câu, xưng mình gọi bạn, gọi đúng tên sản phẩm và giá trong danh mục, gợi ý màu phối"}}
Nếu danh mục không có món phù hợp thì items rỗng và answer nói thật là shop chưa có."""

# Đường lui khi không có model: vai trò từng món + vài dịp quen thuộc
_ROLE = {"Quần": "bottom", "Áo Khoác": "outer", "Sơ Mi": "top", "AIRism": "top", "UV Protection": "outer"}
_RULES = [
    (r"bien|nang|du lich|picnic|he|leo nui", ["AIRism", "UV Protection", "Quần"]),
    (r"phong van|cong so|di lam|lich su|tiec|hen ho", ["Sơ Mi", "Quần"]),
    (r"lanh|dong|da lat|sapa|ret", ["Áo Khoác", "AIRism", "Quần"]),
]


def _catalog(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": p["id"], "name": p["name"], "category": p.get("category", ""), "price": p["price"],
             "colors": p.get("colors", []), "in_stock_sizes": [s for s, n in (p.get("sizes") or {}).items() if n > 0]}
            for p in products]


def _rule_outfit(text: str, products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    t = normalize(text)
    cats = next((c for rx, c in _RULES if re.search(rx, t)), ["AIRism", "Quần"])
    picked, used_roles = [], set()
    for cat in cats:
        for p in sorted(products, key=lambda p: p["price"]):
            role = "acc" if "kinh" in normalize(p["name"]) else _ROLE.get(p.get("category", ""), "top")
            if p.get("category") == cat and role not in used_roles and any(n > 0 for n in (p.get("sizes") or {}).values()):
                picked.append(p)
                used_roles.add(role)
                break
    if re.search(r"bien|nang|du lich|picnic|he", t):
        acc = next((p for p in products if "kinh" in normalize(p["name"])), None)
        if acc and acc not in picked:
            picked.append(acc)
    return picked


def pick_outfit(llm, text: str, products: list[dict[str, Any]], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    by_id = {p["id"]: p for p in products}
    ctx = {"tool_results": [{"tool": "danh_muc_san_pham", "result": _catalog(products)}], "history": history or []}
    data = _ask_json(llm, _OUTFIT_PROMPT.format(shop=active_shop().name), text, ctx)
    items, answer, ai = [], "", False
    if data:
        seen = set()
        for it in data.get("items") or []:
            pid = str((it or {}).get("id", "")).strip()
            p = by_id.get(pid)
            if p and pid not in seen and any(n > 0 for n in (p.get("sizes") or {}).values()):
                items.append(p)
                seen.add(pid)
        answer = str(data.get("answer") or "").strip()
        ai = bool(items and answer)
        if ai and len(items) < 2:  # model chỉ chọn 1 món: bổ sung món khác vai trò để thành một bộ
            vnd = lambda n: f"{int(n):,}".replace(",", ".") + "đ"  # noqa: E731
            extra = [p for p in _rule_outfit(text, products) if p["id"] not in seen][: 3 - len(items)]
            if extra:
                items += extra
                answer += " Bạn có thể phối thêm " + ", ".join(f"{p['name']} ({vnd(p['price'])})" for p in extra) + "."
    if not ai:
        items = _rule_outfit(text, products)
        vnd = lambda n: f"{int(n):,}".replace(",", ".") + "đ"
        answer = ("Mình gợi ý bạn bộ này nè: " + ", ".join(f"{p['name']} ({vnd(p['price'])})" for p in items) + "."
                  if items else "Hiện shop chưa có món phù hợp với nhu cầu này, bạn mô tả thêm dịp mặc để mình tư vấn lại nhé.")
    total = sum(p["price"] for p in items)
    return {"items": items, "answer": answer, "total": total, "ai": ai}


# ---------------------------------------------------------------- 4) Báo cáo tuần
_REPORT_PROMPT = """Bạn là chuyên viên phân tích CSKH của {shop}. Từ SỐ LIỆU và MẪU TIN NHẮN của 7 ngày qua, viết báo cáo cho chủ shop bằng Markdown:
## Tổng quan (2-3 câu, có số liệu)
## Khách hỏi gì nhiều nhất (3-5 gạch đầu dòng)
## Vấn đề cần xử lý (khiếu nại, câu AI chưa trả lời được, sản phẩm khách hỏi mà hết hàng)
## Đề xuất (3-5 việc cụ thể: bổ sung tài liệu, nhập hàng, sửa chính sách, đào tạo nhân viên...)
Chỉ dựa trên dữ liệu được cung cấp, không bịa số, không suy diễn nguyên nhân không có trong dữ liệu.
Viết cho người không rành kỹ thuật: không dùng tên biến tiếng Anh, không dùng dấu `. Ngắn gọn, tiếng Việt.
Trả về DUY NHẤT nội dung Markdown."""


# Nhãn tiếng Việt để model viết báo cáo dễ hiểu cho chủ shop (không lộ tên kỹ thuật)
LABEL_VI = {
    "knowledge": "hỏi chính sách", "order_status": "tra đơn hàng", "handoff": "đòi gặp nhân viên",
    "product_inquiry": "hỏi sản phẩm", "complaint": "khiếu nại", "blocked": "cố lừa AI (bị chặn)",
    "off_topic": "hỏi ngoài lề", "greeting": "chào hỏi", "cancel_order": "hủy đơn", "update_address": "đổi địa chỉ",
    "update_shipping_address": "đổi địa chỉ", "request_refund": "đòi hoàn tiền", "size_advice": "tư vấn size",
    "restock_notify": "báo có hàng lại", "abusive": "xúc phạm", "outfit_advice": "nhờ phối đồ",
    "vision_defect": "gửi ảnh hàng lỗi", "pii_phone": "để lộ số điện thoại (đã tự ẩn)", "pii_email": "để lộ email (đã tự ẩn)",
    "pii_card": "để lộ số thẻ ngân hàng trong chat (đã tự ẩn)", "pii_national_id": "để lộ số CCCD (đã tự ẩn)",
    "prompt_injection": "cố lừa AI tiết lộ hướng dẫn", "no_grounding": "AI không tìm thấy tài liệu để trả lời",
    "ungrounded_numbers": "câu trả lời có số liệu cần kiểm tra", "abusive_language": "lời lẽ xúc phạm",
    "requires_confirmation": "thao tác cần khách xác nhận", "member_only": "khách chưa đăng nhập hỏi việc của thành viên",
    "llm_fallback": "model lỗi, dùng câu dự phòng", "price_mismatch": "AI nêu sai giá (đã chặn)",
    "negative_sentiment": "khách có cảm xúc tiêu cực",
}


def _vi(pairs):
    return [(LABEL_VI.get(k, k), v) for k, v in pairs]


def collect_week(traces: list[dict[str, Any]], products: list[dict[str, Any]], days: int = 7) -> dict[str, Any]:
    since = time.time() - days * 86400
    rows = [t for t in traces if (t.get("created_at") or 0) >= since]
    st, it, fl = Counter(), Counter(), Counter()
    unanswered, complaints, prod_hits, oos = [], [], Counter(), Counter()
    norm_names = [(normalize(p["name"]), p) for p in products]
    for t in rows:
        st[t.get("status")] += 1
        it[t.get("intent")] += 1
        for f in t.get("guardrail_flags") or []:
            fl[f] += 1
        q = (t.get("user_input") or "").strip()
        nq = normalize(q)
        if t.get("status") == "fallback" and len(unanswered) < 15:
            unanswered.append(q[:160])
        if t.get("intent") in ("complaint", "vision_defect") and len(complaints) < 10:
            complaints.append(q[:160])
        for n, p in norm_names:
            key = " ".join(n.split()[:3])
            if key and key in nq:
                prod_hits[p["name"]] += 1
                m = re.search(r"\bsize\s*(xs|s|m|l|xl|xxl)\b", nq)
                if m and (p.get("sizes") or {}).get(m.group(1).upper(), 1) == 0:
                    oos[f"{p['name']} size {m.group(1).upper()}"] += 1
    n = len(rows) or 1
    return {
        "days": days, "total": len(rows),
        "ai_resolved_pct": round(sum(st[s] for s in ("ok", "clarify", "confirm")) / n * 100, 1),
        "handoff": st["handoff"], "fallback": st["fallback"], "login_required": st["login_required"],
        "top_intents": _vi(it.most_common(8)), "flags": _vi(fl.most_common(8)),
        "products_asked": prod_hits.most_common(8), "out_of_stock_asked": oos.most_common(5),
        "unanswered_samples": unanswered, "complaint_samples": complaints,
    }


def weekly_report(llm, stats: dict[str, Any]) -> dict[str, Any]:
    md = ""
    if not isinstance(llm, MockLLM) and stats["total"]:
        try:
            res = llm.complete(_REPORT_PROMPT.format(shop=active_shop().name), "Viết báo cáo tuần.",
                               {"tool_results": [{"tool": "so_lieu_7_ngay", "result": stats}]})
            md = res.text.strip()
        except LLMError:
            md = ""
    ai = bool(md)
    if not md:
        lines = [f"## Tổng quan", f"- {stats['total']} tin nhắn trong {stats['days']} ngày, AI tự xử lý {stats['ai_resolved_pct']}%.",
                 f"- Chuyển nhân viên {stats['handoff']} lần, chưa trả lời được {stats['fallback']} lần.", "",
                 "## Khách hỏi gì nhiều nhất"] + [f"- {k}: {v}" for k, v in stats["top_intents"][:5]]
        if stats["unanswered_samples"]:
            lines += ["", "## Câu AI chưa trả lời được"] + [f"- {q}" for q in stats["unanswered_samples"][:5]]
        if stats["out_of_stock_asked"]:
            lines += ["", "## Hỏi mà hết hàng"] + [f"- {k}: {v} lượt" for k, v in stats["out_of_stock_asked"]]
        md = "\n".join(lines)
    return {"markdown": md, "stats": stats, "ai": ai, "created_at": time.time()}
