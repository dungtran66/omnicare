"""Kiểm thử đa cửa hàng: shop TECHZONE chạy độc lập, và NORIVA vẫn là mặc định."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from app import shop  # noqa: E402
from app.agent import OmniCareAgent, rule_intent  # noqa: E402
from app.observability import TraceStore  # noqa: E402


@pytest.fixture()
def techzone(tmp_path):
    shop.activate("techzone")
    try:
        yield OmniCareAgent(store=TraceStore(tmp_path / "tz.db"))
    finally:
        shop.activate(shop.DEFAULT_SHOP)


def test_default_shop_is_noriva():
    assert shop.active().id == "noriva" and shop.active().name == "NORIVA"
    assert shop.active().data_dir.name == "data" and shop.active().size_advice


def test_list_shops_hides_template():
    ids = {s["id"] for s in shop.list_shops()}
    assert {"noriva", "techzone"} <= ids and "_template" not in ids


def test_unknown_shop_rejected():
    with pytest.raises(ValueError):
        shop.load_profile("khong-ton-tai")


def test_techzone_uses_its_own_data(techzone):
    assert "OC20001" in techzone.tools.orders and "OC10001" not in techzone.tools.orders
    assert any("iPhone" in p["name"] for p in techzone.tools.products)
    assert {c.source for c in techzone.tools.retriever.chunks} >= {"bao_hanh.md", "thanh_toan.md"}


def test_techzone_knowledge_and_orders(techzone):
    r = techzone.handle("t1", "Phí ship cho đơn 200 nghìn là bao nhiêu?")
    assert r.status == "ok" and "25.000" in r.answer
    r = techzone.handle("t2", "Đơn OC20001 của tôi đang ở đâu?")
    assert r.intent == "order_status" and "GHN7731022" in r.answer


def test_techzone_product_and_complaint_rules(techzone):
    assert rule_intent("iPhone 15 giá bao nhiêu?") == "product_inquiry"
    assert rule_intent("Máy của tôi bị sập nguồn, đơn OC20003") == "complaint"
    r = techzone.handle("t3", "iPhone 15 giá bao nhiêu?")
    assert "19.990.000" in r.answer and "phiên bản" in r.answer.lower()


def test_techzone_has_no_size_advice(techzone):
    r = techzone.handle("t4", "Cao 1m70 nặng 60kg mặc size gì?")
    assert r.intent != "size_advice"


def test_techzone_guest_message_uses_shop_name(techzone):
    r = techzone.handle("t5", "Đơn OC20001 của tôi đâu?", customer={"type": "guest", "guest_code": "TZ01"})
    assert r.status == "login_required" and "TECHZONE" in r.answer and "size" not in r.answer
