"""Test các tính năng thêm ngày 22/11/2026: model dự phòng, thẻ sản phẩm theo câu trả lời, câu xúc phạm/ngoài lề,
Auto-FAQ, đăng nhập Admin phía server và streaming."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from app import auth, autofaq, guardrails  # noqa: E402
from app.agent import OmniCareAgent  # noqa: E402
from app.llm import LLMError, LLMResult, MockLLM, OpenAICompatibleLLM  # noqa: E402
from app.observability import TraceStore  # noqa: E402


@pytest.fixture()
def agent(tmp_path):
    return OmniCareAgent(store=TraceStore(tmp_path / "t.db"))


def test_fallback_models_tried_in_order(monkeypatch):
    llm = OpenAICompatibleLLM("http://x", "k", "main", fallback_models=["fb1", "fb2"])
    tried = []

    def fake(model, *a, **k):
        tried.append(model)
        if model != "fb2":
            raise LLMError("503")
        return LLMResult(text="ok", prompt_tokens=1, completion_tokens=1, model=model)

    monkeypatch.setattr(llm, "_complete_one", fake)
    assert llm.complete("s", "u", {}).model == "fb2" and tried == ["main", "fb1", "fb2"]


def test_cards_follow_answer(agent):
    ans = ("Bên mình có Áo Polo Vải Dry Pique Thoáng Khí giá 489.000đ. À, bạn lấy thêm "
           "Kính Mát Chống Tia UV400 Gọng Nhẹ là được miễn phí giao hàng.")
    names = [c["name"] for c in agent._cards_for_answer(ans, [{"id": "p3", "name": "Cardigan"}])]
    assert names == ["Áo Polo Vải Dry Pique Thoáng Khí", "Kính Mát Chống Tia UV400 Gọng Nhẹ"]


def test_price_guardrail_per_product(agent):
    ok = "Áo Thun AIRism Cotton Dáng Rộng Tay Lỡ giá 391.000đ và Áo Polo Vải Dry Pique Thoáng Khí giá 489.000đ."
    assert agent._price_mismatch(ok) == []
    assert agent._price_mismatch("Áo Polo Vải Dry Pique Thoáng Khí giá 999.000đ.")


@pytest.mark.parametrize("text,action", [
    ("thg ngu chủ shop", "calm"), ("shop như cc", "calm"),
    ("Shop lừa đảo, giao hàng chậm vcl", "handoff"), ("Shop có bán đồ ngủ không?", "allow"),
    ("Áo này đeo kính được không", "allow"), ("Hôm nay thời tiết thế nào?", "decline"),
])
def test_insult_and_off_topic_rules(text, action):
    assert guardrails.check_input(text).action == action


def test_guest_insult_gets_calm_reply(agent):
    r = agent.handle("g1", "thg ngu chủ shop", customer={"type": "guest", "guest_code": "AB12"})
    assert r.status == "declined" and "đăng nhập" not in r.answer.lower()


def test_autofaq_filters_junk_and_groups():
    traces = [{"status": "fallback", "user_input": q, "guardrail_flags": ["no_grounding"]} for q in
              ["Shop có sửa gấu quần không?", "shop có sửa gấu quần không", "thg ngu chủ shop",
               "hôm nay thời tiết thế nào", "bạn bao nhiêu tuổi"]]
    autofaq._CACHE.clear()
    agent = OmniCareAgent()
    gaps = autofaq.build_gaps(traces, set(), MockLLM(), agent.tools.retriever, agent.tools.products)
    assert [(g["question"], g["count"]) for g in gaps] == [("Shop có sửa gấu quần không?", 2)]
    assert "[Cần bổ sung" in gaps[0]["suggested_markdown"]


def test_admin_auth_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.setattr(auth, "ENABLED", True)
    from app.main import app
    c = TestClient(app)
    assert c.get("/api/metrics").status_code == 401
    assert c.post("/api/admin/login", json={"user": "admin", "password": "sai"}).status_code == 401
    tok = c.post("/api/admin/login", json={"user": "admin", "password": "admin123"}).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    assert c.get("/api/metrics", headers=h).status_code == 200
    assert "admin123" not in (tmp_path / "users.json").read_text(encoding="utf-8")
    c.post("/api/admin/logout", headers=h)
    assert c.get("/api/metrics", headers=h).status_code == 401
    assert c.post("/api/chat", json={"session_id": "pub", "message": "Phí ship bao nhiêu?"}).status_code == 200


def test_chat_stream_sends_steps_then_answer():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app).stream("POST", "/api/chat/stream", json={"session_id": "s1", "message": "Đơn OC10002 ở đâu?"}) as r:
        events = [json.loads(l[5:]) for l in r.iter_lines() if l.startswith("data:")]
    types = [e["type"] for e in events]
    assert types[0] == "step" and "token" in types and types[-1] == "done"
    assert types.index("token") > max(i for i, t in enumerate(types) if t == "step")  # chữ chỉ chạy sau khi xử lý xong
    assert events[-1]["intent"] == "order_status"


def test_agent_overview_matches_registry(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(auth, "ENABLED", False)
    from app.main import app
    d = TestClient(app).get("/api/agent/overview").json()
    names = {t["name"] for t in d["tools"]}
    used = {t for b in d["branches"] for t in b["tools"]}
    assert used <= names and len(names) == 11 and used == names  # mọi công cụ đều thuộc một nhánh
    assert all(set(b["kb"]) <= set(d["kb_docs"]) for b in d["branches"])


# ---- Stylist, tóm tắt bàn giao, Copilot, báo cáo tuần (chạy Mock: dùng đường lui bằng luật) ----
def test_outfit_advice_picks_in_stock_bundle(agent):
    r = agent.handle("st1", "Đi phỏng vấn nên mặc gì cho lịch sự?", customer={"type": "guest", "guest_code": "ST01"})
    assert r.intent == "outfit_advice" and r.status == "ok" and len(r.products) >= 2
    cats = [p["category"] for p in r.products]
    assert "Sơ Mi" in cats and "Quần" in cats and "Cả bộ" in r.answer


def test_outfit_ignores_ids_not_in_catalog(agent, monkeypatch):
    from app import ai_features
    monkeypatch.setattr(ai_features, "_ask_json", lambda *a, **k: {
        "items": [{"id": "p999"}, {"id": "p5"}, {"id": "p4"}], "answer": "Bạn mặc Áo Sơ Mi Vải Linen Pha Dáng Thoải Mái nhé."})
    fit = ai_features.pick_outfit(object(), "đi làm mặc gì", agent.tools.products)
    assert [p["id"] for p in fit["items"]] == ["p5", "p4"] and fit["ai"]


def test_handoff_summary_fallback_has_fields():
    from app import ai_features
    msgs = [{"role": "visitor", "text": "Đơn OC10005 giao 5 ngày chưa tới, bực quá"}, {"role": "agent", "text": "Mình kiểm tra nhé"}]
    s = ai_features.summarize_handoff(MockLLM(), msgs, {"name": "Thu Lan"})
    assert s["customer"].startswith("Thu Lan") and "OC10005" in s["customer"] and s["next_step"] and not s["ai"]


def test_summary_hidden_from_public_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    main.agent.sessions["pub-sum"] = {"id": "pub-sum", "messages": [{"role": "visitor", "text": "hi"}],
                                      "handoff_summary": {"issue": "nội bộ"}}
    assert "handoff_summary" not in TestClient(main.app).get("/api/conversations/pub-sum").json()


def test_weekly_report_fallback_uses_real_numbers():
    from app import ai_features
    traces = [{"created_at": __import__("time").time(), "status": "fallback", "intent": "knowledge",
               "user_input": "Shop có sửa gấu quần không?", "guardrail_flags": ["no_grounding"]}] * 3
    stats = ai_features.collect_week(traces, [])
    rep = ai_features.weekly_report(MockLLM(), stats)
    assert stats["total"] == 3 and "3 tin nhắn" in rep["markdown"] and "sửa gấu quần" in rep["markdown"]
    assert ("hỏi chính sách", 3) in stats["top_intents"]


def test_order_price_computed_on_server():
    """Trình duyệt chỉ gửi mã + size + số lượng; giá lấy từ danh mục, số tiền gửi kèm bị bỏ qua."""
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    o = c.post("/api/orders/create", json={"items": [{"id": "p6", "size": "L", "qty": 1}], "subtotal": 1000, "total": 1000}).json()
    assert o["subtotal"] == 1470000 and o["shipping_fee"] == 0 and o["total"] == 1470000
    assert "Áo Phao" in o["product"] and "(Size L)" in o["product"]
    assert c.post("/api/orders/create", json={"items": [{"id": "khong-co", "qty": 1}]}).status_code == 400


def test_close_conversation_deletes_and_marks_closed(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "users.json")
    monkeypatch.setattr(auth, "ENABLED", True)
    from app.main import app
    c = TestClient(app)
    c.post("/api/chat", json={"session_id": "close-1", "message": "chào shop", "customer": {"type": "guest", "guest_code": "AB12"}})
    assert c.post("/api/conversations/close-1/close").status_code == 401  # cần đăng nhập Admin
    h = {"Authorization": "Bearer " + c.post("/api/admin/login", json={"user": "admin", "password": "admin123"}).json()["token"]}
    assert c.post("/api/conversations/close-1/close", headers=h).json()["deleted"] is True
    d = c.get("/api/conversations/close-1").json()
    assert d["messages"] == [] and d["closed_at"]
    assert all(x["id"] != "close-1" for x in c.get("/api/conversations", headers=h).json())


def test_csat_stars_recorded():
    from fastapi.testclient import TestClient
    from app.main import app, agent
    c = TestClient(app)
    c.post("/api/chat", json={"session_id": "csat-s", "message": "phí ship bao nhiêu"})
    assert c.post("/api/feedback", json={"trace_id": "csat-sess-g-x", "score": 1, "stars": 5}).status_code == 200
    assert agent.store.metrics()["csat"]["count"] >= 1


def test_chat_rejects_unsafe_session_id():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    r = c.post("/api/chat", json={"message": "xin chào", "session_id": "x');alert(1);//"})
    assert r.status_code == 422
    r = c.post("/api/chat", json={"message": "xin chào", "session_id": "sess-g-ABCD-abc123"})
    assert r.status_code == 200


def test_widget_js_and_cors():
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    r = c.get("/widget.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    pre = c.options("/api/chat", headers={"Origin": "https://shop-khac.vn", "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 204 and pre.headers["access-control-allow-origin"] == "*"
    r = c.post("/api/chat", json={"message": "phí ship bao nhiêu", "session_id": "w-abcd-test0001"},
               headers={"Origin": "https://shop-khac.vn"})
    assert r.status_code == 200 and r.headers.get("access-control-allow-origin") == "*"
    # API quản trị không mở CORS cho website khác
    r = c.get("/api/conversations", headers={"Origin": "https://shop-khac.vn"})
    assert "access-control-allow-origin" not in r.headers
