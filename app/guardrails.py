"""Guardrails cho đầu vào và đầu ra của agent.

Input:  giới hạn độ dài, phát hiện prompt injection / jailbreak, che PII, phát hiện ngôn từ xúc phạm, chặn chủ đề ngoài phạm vi.
Output: che PII bị lộ, chặn rò rỉ system prompt, chặn cam kết vượt chính sách, kiểm tra grounding (có căn cứ).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .rag import normalize

MAX_INPUT_CHARS = 2000

_INJECTION_PATTERNS = [
    r"ignore (all|any|the)? ?(previous|above|prior) (instructions|prompts?)",
    r"disregard (the|all|your) (rules|instructions)",
    r"(reveal|show|print|repeat) (your|the) (system )?(prompt|instructions)",
    r"you are now (dan|developer mode)",
    r"\bjailbreak\b",
    r"bo qua (tat ca |moi )?(cac )?(huong dan|chi dan|quy tac|lenh)",
    r"(tiet lo|in ra|cho (toi|minh) xem) (system prompt|prompt he thong|huong dan he thong)",
    r"gia vo (ban|minh) la",
    r"dong vai (admin|quan tri|hacker)",
    # Biến thể trộn Anh-Việt / teencode / viết tắt
    r"\b(ignore|bo qua|quen|phot lo|lo di)\b.{0,30}\b(rule|rules|luat|quy tac|huong dan|instruction|instructions|lenh)\b",
    r"\b(cho|show|in|xem|dump|leak|lo)\b.{0,25}\b(system ?prompt|prompt (he thong|goc|an)|sys prompt|huong dan an)\b",
    r"\b(prompt|huong dan) (he thong|goc)\b.{0,20}\b(la gi|ra|di|xem)\b",
]

_OFF_TOPIC_PATTERNS = [
    r"\b(viet|code|lap trinh) (code|chuong trinh|ham|script)\b",
    r"\b(chinh tri|bau cu|ton giao)\b",
    r"\b(hack|bom|vu khi|ma tuy)\b",
    r"\b(chung khoan|tien ao|crypto) nen mua\b",
    r"\b(thoi tiet|du bao thoi tiet|ke chuyen cuoi|chuyen cuoi|lam tho|xo so|so de|ket qua bong da|tong thong|thu tuong)\b",
    r"\b(giai|lam) (bai tap|toan|phuong trinh)\b",
    r"^\s*\d+\s*[-+*/x:]\s*\d+\s*(=|bang)?\s*(may|bao nhieu)?\s*\??\s*$",
]

# Câu xúc phạm / tục tĩu: so trên chữ thường CÒN DẤU để không nhầm "đồ ngủ", "đeo"... với chửi bậy.
_INSULT_RE = re.compile(
    r"(?<!\w)(ngu|đồ ngu|ngu ngốc|ngu như bò|óc chó|oc cho|óc lợn|đồ chó|thằng chó|con chó|chó chết|điên|khùng|"
    r"khốn nạn|mất dạy|láo|đéo|địt|dit me|đm|đmm|dm|dmm|dcm|đcm|dkm|đkm|cc|clm|cl|vcl|vkl|vl|cút|"
    r"mẹ mày|bố mày|má mày|thg ngu|thằng ngu|con ngu|rác rưởi|đồ rác)(?!\w)",
    re.IGNORECASE,
)
# Có kèm vấn đề cụ thể (đơn, giao hàng, lỗi...) thì vẫn chuyển nhân viên như cũ.
_COMPLAINT_HINT_RE = r"\b(lua dao|giao (cham|sai|thieu|lau)|cham|don|oc\d+|loi|hong|rach|hoan tien|bom hang|chua nhan|khong nhan|ship)\b"

_ABUSE_WORDS = ["dm", "dmm", "vcl", "vl", "ngu vai", "lua dao", "cut"]  # "do ngu" bỏ vì trùng "đồ ngủ"

_NEGATIVE_SENTIMENT_PATTERNS = [
    r"rat that vong",
    r"qua te",
    r"cho mai khong thay",
    r"lam an (chan|qua te|kieu gi)",
    r"buc minh",
    r"khong hai long",
]

_POSITIVE_SENTIMENT_PATTERNS = [
    r"cam on",
    r"rat tot",
    r"hai long",
    r"tuyet voi",
    r"nhanh chong",
    r"dep",
]


def analyze_sentiment(text: str) -> dict[str, str | float]:
    """Chấm điểm cảm xúc ngữ nghĩa từ -1.0 (tiêu cực) đến +1.0 (tích cực)."""
    norm = normalize(text)
    words = set(re.findall(r"[a-z]+", norm))
    if any((w in words) if " " not in w else (w in norm) for w in _ABUSE_WORDS):
        return {"score": -0.95, "label": "negative"}
    neg_hits = sum(1 for p in _NEGATIVE_SENTIMENT_PATTERNS if re.search(p, norm))
    pos_hits = sum(1 for p in _POSITIVE_SENTIMENT_PATTERNS if re.search(p, norm))
    if neg_hits > pos_hits:
        return {"score": round(max(-0.9, -0.45 * neg_hits), 2), "label": "negative"}
    if pos_hits > neg_hits:
        return {"score": round(min(0.95, 0.45 * pos_hits), 2), "label": "positive"}
    return {"score": 0.0, "label": "neutral"}

_PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "national_id": re.compile(r"\b\d{12}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){9}(?!\d)"),
}

_FORBIDDEN_COMMITMENTS = [
    r"cam ket hoan tien 100",
    r"chac chan (se )?(duoc )?(hoan tien|boi thuong)",
    r"mien phi tron doi",
    r"boi thuong gap (doi|muoi)",
]

_SYSTEM_LEAK_MARKERS = ["omnicare_system_prompt", "quy tac bat buoc:"]


@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    flags: list[str] = field(default_factory=list)
    reason: str | None = None
    action: str = "allow"  # allow | block | handoff | decline


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def card_sub(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("pii_card")
            return "[REDACTED_CARD]"
        return m.group()

    text = _PII_PATTERNS["card"].sub(card_sub, text)
    for kind in ("email", "national_id", "phone"):
        def sub(m: re.Match, kind=kind) -> str:
            found.append(f"pii_{kind}")
            return f"[REDACTED_{kind.upper()}]"
        text = _PII_PATTERNS[kind].sub(sub, text)
    return text, sorted(set(found))


def check_input(text: str) -> GuardrailResult:
    flags: list[str] = []
    if not text or not text.strip():
        return GuardrailResult(False, text, ["empty_input"], "Tin nhắn trống.", "block")
    if len(text) > MAX_INPUT_CHARS:
        return GuardrailResult(False, text[:MAX_INPUT_CHARS], ["input_too_long"],
                               f"Tin nhắn vượt quá {MAX_INPUT_CHARS} ký tự.", "block")

    norm = normalize(text)
    if any(re.search(p, norm) for p in _INJECTION_PATTERNS):
        return GuardrailResult(False, text, ["prompt_injection"],
                               "Yêu cầu chứa nội dung cố thay đổi chỉ dẫn hệ thống.", "block")
    if any(re.search(p, norm) for p in _OFF_TOPIC_PATTERNS):
        return GuardrailResult(False, text, ["off_topic"], "Câu hỏi nằm ngoài phạm vi hỗ trợ.", "decline")

    redacted, pii = redact_pii(text)
    flags += pii

    words = set(re.findall(r"[a-z]+", norm))
    if _INSULT_RE.search(text) or any((w in words) if " " not in w else (w in norm) for w in _ABUSE_WORDS):
        flags.append("abusive_language")
        if re.search(_COMPLAINT_HINT_RE, norm):
            return GuardrailResult(True, redacted, flags, "Khách hàng đang bức xúc.", "handoff")
        return GuardrailResult(True, redacted, flags, "Khách dùng lời lẽ xúc phạm.", "calm")

    if any(re.search(p, norm) for p in _NEGATIVE_SENTIMENT_PATTERNS):
        flags.append("negative_sentiment")

    return GuardrailResult(True, redacted, flags)


def check_output(answer: str, context_texts: list[str] | None = None) -> GuardrailResult:
    flags: list[str] = []
    norm = normalize(answer)

    if any(m in norm for m in _SYSTEM_LEAK_MARKERS):
        return GuardrailResult(False, answer, ["system_prompt_leak"], "Phát hiện rò rỉ chỉ dẫn hệ thống.", "block")

    if any(re.search(p, norm) for p in _FORBIDDEN_COMMITMENTS):
        return GuardrailResult(False, answer, ["forbidden_commitment"],
                               "Câu trả lời đưa ra cam kết vượt chính sách.", "block")

    answer, pii = redact_pii(answer)
    flags += [f"output_{p}" for p in pii]

    # Grounding: các con số (ngày, %, tiền) trong câu trả lời phải xuất hiện trong ngữ cảnh.
    if context_texts is not None:
        ctx = normalize(" ".join(context_texts))
        numbers = set(re.findall(r"\d+(?:[.,]\d+)*", normalize(answer)))
        # context gồm cả kết quả tool (mã đơn, mã phiếu, giá số nguyên 686000 vs 686.000) nên coi cả 2 dạng là có căn cứ
        ungrounded = [
            n for n in numbers
            if n not in ctx
            and n.replace(".", "").replace(",", "") not in ctx
            and n not in {"0", "1", "2", "3", "4", "5", "24"}
        ]
        if ungrounded:
            flags.append("ungrounded_numbers")

    return GuardrailResult(True, answer, flags)
