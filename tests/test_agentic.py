"""Test đợt nâng cấp 23/11/2026: agent tự gọi công cụ, đọc ảnh bằng model, giới hạn lượt gọi model, giỏ hàng, size."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from app import config, llm as llm_mod  # noqa: E402
from app.agent import OmniCareAgent, height_cm, HEIGHT_RE, plain_text  # noqa: E402
from app.llm import LLM, LLMResult  # noqa: E402
from app.observability import TraceStore  # noqa: E402


class FakeToolLLM(LLM):
    """Giả một model biết gọi công cụ: lần lượt gọi các công cụ trong `plan` rồi trả lời."""
    name = "fake:tools"
    supports_tools = True
    supports_vision = True

    def __init__(self, plan, vision_json=None):
        self.plan, self.vision_json, self.seen = plan, vision_json, []

    def complete(self, system, user, context, json_mode=False):
        return LLMResult(text="trả lời thường", model="fake-1")

    def run_tools(self, system, user, tools, execute, max_steps=4):
        self.tool_names = [t["function"]["name"] for t in tools]
        for name, args in self.plan:
            self.seen.append(execute(name, args))
        return LLMResult(text="**Đã tra xong** cho bạn", model="fake-1")

    def see(self, system, prompt, image_url, json_mode=True):
        return LLMResult(text=json.dumps(self.vision_json), model="fake-1")


@pytest.fixture()
def store(tmp_path):
    return TraceStore(tmp_path / "t.db")


def test_agent_loop_calls_tools_and_strips_markdown(store):
    fake = FakeToolLLM([("check_inventory", {"product": "Áo Polo", "size": "L"})])
    agent = OmniCareAgent(llm=fake, store=store)
    r = agent.handle("s1", "áo polo giá bao nhiêu", customer={"type": "guest", "guest_code": "AB12"})
    assert [c["tool"] for c in r.tool_calls].count("check_inventory") == 1
    assert "**" not in r.answer
    assert "get_order_status" not in fake.tool_names  # khách vãng lai không được model tra đơn
    assert not {"cancel_order", "request_refund"} & set(fake.tool_names)  # công cụ có tác dụng phụ không giao cho model


def test_agent_loop_blocks_orders_of_other_members(store):
    fake = FakeToolLLM([("get_order_status", {"order_id": "OC10001"})])
    agent = OmniCareAgent(llm=fake, store=store)
    cust = {"type": "member", "member_id": "M-1024", "phone": "0918222333", "order_ids": ["OC10005"]}
    agent.handle("s2", "đơn OC10005 của tôi đâu rồi", customer=cust)
    assert "error" in fake.seen[0]  # OC10001 không thuộc khách này


def test_vision_uses_model_and_not_product_asks_again(store):
    fake = FakeToolLLM([], vision_json={"kind": "not_product", "finding": "Ảnh bị mờ.", "severity": "nhẹ",
                                        "suggested_category": "other", "order_id": ""})
    agent = OmniCareAgent(llm=fake, store=store)
    img = {"name": "a.jpg", "image_url": "data:image/jpeg;base64,AAAA"}
    r = agent.handle("s3", "áo bị lỗi", image_info=img)
    assert r.status == "clarify" and "chụp lại" in r.answer
    fake.vision_json = {"kind": "apparel_defect", "finding": "Đường may vai trái bị bung chỉ.", "severity": "vừa",
                        "suggested_category": "return", "order_id": ""}
    r = agent.handle("s4", "áo đơn OC10005 bị lỗi", image_info=img)
    assert "bung chỉ" in r.answer and any(c["tool"] == "create_support_ticket" for c in r.tool_calls)


def test_quota_per_customer(monkeypatch):
    monkeypatch.setattr(config, "MODEL_CALLS_PER_DAY", 2)
    llm_mod._QUOTA.clear()
    llm_mod.QUOTA_KEY.set("guest-x")
    llm_mod.quota_take()
    llm_mod.quota_take()
    with pytest.raises(llm_mod.QuotaExceeded):
        llm_mod.quota_take()
    llm_mod.QUOTA_KEY.set(None)  # Admin / nội bộ không bị giới hạn
    llm_mod.quota_take()


def test_cart_answer_adds_shipping(store):
    agent = OmniCareAgent(store=store)
    r = agent.handle("s5", "tổng tiền giỏ hàng", page={"cart": [{"id": "p2", "size": "M", "qty": 1}]})
    assert r.intent == "cart_inquiry" and "30.000đ" in r.answer and "421.000đ" in r.answer


@pytest.mark.parametrize("text,cm", [("cao 1m9", 190), ("cao 1m68", 168), ("cao 175cm", 175)])
def test_height_parsing(text, cm):
    assert height_cm(HEIGHT_RE.search(text)) == cm


def test_women_size_chart_and_free_size(store):
    agent = OmniCareAgent(store=store)
    r = agent.handle("s6", "mình là nữ cao 1m60 nặng 50kg mặc size gì")
    assert "bảng size nữ" in r.answer
    r = agent.handle("s7", "Kính Mát Chống Tia UV400 còn size M không, cao 1m68 nặng 60kg")
    assert "một cỡ" in r.answer


def test_plain_text():
    assert plain_text("**Đậm** và __nghiêng__\n## Tiêu đề") == "Đậm và nghiêng\nTiêu đề"


def test_image_search_for_guest(store):
    fake = FakeToolLLM([], vision_json={"kind": "fashion_item", "finding": "Áo khoác phao màu đen.", "severity": "",
                                        "suggested_category": "other", "order_id": "",
                                        "matches": [{"id": "p6", "reason": "cùng là áo khoác phao"}, {"id": "zzz"}]})
    agent = OmniCareAgent(llm=fake, store=store)
    img = {"name": "a.jpg", "image_url": "data:image/jpeg;base64,AAAA"}
    r = agent.handle("s8", "", image_info=img, customer={"type": "guest", "guest_code": "G1"})
    assert r.intent == "image_search" and r.status == "ok"
    assert [p["id"] for p in r.products] == ["p6"]  # mã không có trong danh mục bị bỏ


def test_member_memory_across_sessions(store):
    agent = OmniCareAgent(store=store)
    cust = {"type": "member", "member_id": "M-9", "name": "An", "phone": "0900000009", "order_ids": []}
    agent.handle("m1", "mình cao 1m75 nặng 70kg mặc size gì", customer=cust)
    agent2 = OmniCareAgent(store=store)  # khởi động lại server: trí nhớ đọc từ SQLite
    r = agent2.handle("m2", "xin chào", customer=cust)
    assert "size L" in r.answer


def test_conversation_qa_rule_fallback():
    from app import ai_features
    from app.llm import MockLLM
    msgs = [{"role": "visitor", "text": "trả góp được không"}, {"role": "agent", "text": "...", "status": "fallback"}]
    q = ai_features.score_conversation(MockLLM(), msgs)
    assert q["score"] <= 3 and q["resolved"] is False and q["ai"] is False


class BusyVisionLLM(FakeToolLLM):
    """Model có khả năng đọc ảnh nhưng đang quá tải (vd lỗi 429)."""

    def see(self, system, prompt, image_url, json_mode=True):
        from app.llm import LLMError
        raise LLMError("HTTP Error 429: Too Many Requests")


def test_vision_failure_does_not_invent_findings(store):
    agent = OmniCareAgent(llm=BusyVisionLLM([]), store=store)
    img = {"name": "a.jpg", "image_url": "data:image/jpeg;base64,AAAA"}
    # Khách vãng lai gửi ảnh để tìm mẫu: không bị bắt đăng nhập, được nhờ gửi lại/mô tả
    r = agent.handle("v1", "", image_info=img, customer={"type": "guest", "guest_code": "G9"})
    assert r.status == "clarify" and r.intent == "image_search"
    # Thành viên báo lỗi: vẫn tạo phiếu theo lời khách, nhưng không tự nhận đã "phát hiện" lỗi trong ảnh
    cust = {"type": "member", "member_id": "M-1024", "phone": "0918222333", "order_ids": ["OC10005"]}
    r = agent.handle("v2", "áo đơn OC10005 bị rách", image_info=img, customer=cust)
    assert any(c["tool"] == "create_support_ticket" for c in r.tool_calls)
    assert "Phát hiện" not in r.answer and "mô tả của bạn" in r.answer


def test_cancel_intent_variants():
    from app.agent import rule_intent
    assert rule_intent("Hủy luôn đơn OC10005 cho mình, khỏi hỏi lại") == "cancel_order"
