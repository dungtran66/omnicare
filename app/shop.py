"""Hồ sơ cửa hàng (shop profile): mọi thứ riêng của một shop nằm trong `shops/<id>/shop.json`.

Agent, RAG, công cụ và trang bán hàng đọc hồ sơ đang bật qua `active()`, nên cùng một hệ thống
có thể demo nhiều loại shop khác nhau. Mặc định là NORIVA (thời trang), dữ liệu ở thư mục `data/`.

Đổi shop: biến môi trường `SHOP=<id>` khi chạy server, hoặc `POST /api/shops/active` khi server đang chạy.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

SHOPS_DIR = config.BASE_DIR / "shops"
DEFAULT_SHOP = "noriva"


def _norm(text: str) -> str:
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


@dataclass
class ShopProfile:
    id: str
    name: str
    data_dir: Path
    trace_db: Path
    intro: str                     # câu giới thiệu shop đưa vào system prompt
    intent_label: str              # vd "shop thời trang NORIVA"
    knowledge_examples: str        # ví dụ chủ đề chính sách cho prompt phân loại
    freeship_threshold: int
    size_advice: bool              # bật tư vấn size theo chiều cao/cân nặng (shop quần áo)
    size_exchange_note: str        # câu nhắc đổi size sau khi tư vấn size (để trống nếu không có)
    product_keywords: list[str]
    defect_keywords: list[str]
    synonyms: dict[str, list[str]]
    storefront: dict[str, Any] = field(default_factory=dict)
    demo_accounts: list[dict[str, Any]] = field(default_factory=list)
    is_default: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def variant_label(self) -> str:
        """Tên gọi biến thể sản phẩm: "Size" (quần áo) hoặc "Phiên bản" (128GB, 41mm...)."""
        return self.raw.get("variant_label") or ("Size" if self.size_advice else "Phiên bản")

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"

    @property
    def product_pattern(self) -> str:
        """Regex (trên chữ đã bỏ dấu) nhận ra tên sản phẩm của shop."""
        words = [re.escape(w) for w in self.product_keywords if w]
        return r"\b(" + "|".join(words) + r")\b" if words else r"(?!x)x"

    @property
    def defect_pattern(self) -> str:
        return "|".join(self.defect_keywords)

    def public(self) -> dict[str, Any]:
        """Thông tin an toàn để trả cho trình duyệt (không có đường dẫn trên máy)."""
        return {
            "id": self.id,
            "name": self.name,
            "intro": self.intro,
            "size_advice": self.size_advice,
            "freeship_threshold": self.freeship_threshold,
            "storefront": self.storefront,
            "demo_accounts": [
                {k: v for k, v in a.items() if k != "password"} for a in self.demo_accounts
            ],
            "is_default": self.is_default,
        }


def _derive_keywords(products: list[dict[str, Any]]) -> list[str]:
    """Shop không khai báo product_keywords: lấy cụm 2 từ đầu của tên sản phẩm (đã bỏ dấu)."""
    out: list[str] = []
    for p in products:
        words = re.findall(r"[a-z0-9]+", _norm(p.get("name", "")))
        if len(words) >= 2:
            out.append(f"{words[0]} {words[1]}")
        for w in words:
            if len(w) >= 4 and w not in out:
                out.append(w)
    return out


def load_profile(shop_id: str) -> ShopProfile:
    folder = SHOPS_DIR / shop_id
    cfg_file = folder / "shop.json"
    if not cfg_file.exists():
        raise ValueError(f"Không tìm thấy shop '{shop_id}' (thiếu {cfg_file.relative_to(config.BASE_DIR)})")
    raw = json.loads(cfg_file.read_text(encoding="utf-8"))
    data_dir = (config.BASE_DIR / raw["data_dir"]) if raw.get("data_dir") else folder
    is_default = shop_id == DEFAULT_SHOP
    # Shop mặc định dùng DB cũ để giữ nguyên lịch sử; shop khác có DB riêng (đơn, phiếu, hội thoại không lẫn nhau)
    trace_db = config.TRACE_DB if is_default else config.BASE_DIR / f"omnicare_traces_{shop_id}.db"
    keywords = raw.get("product_keywords")
    if not keywords:
        prod_file = data_dir / "products.json"
        products = json.loads(prod_file.read_text(encoding="utf-8")) if prod_file.exists() else []
        keywords = _derive_keywords(products)
    name = raw["name"]
    return ShopProfile(
        id=shop_id,
        name=name,
        data_dir=data_dir,
        trace_db=trace_db,
        intro=raw.get("intro", name),
        intent_label=raw.get("intent_label", f"cửa hàng {name}"),
        knowledge_examples=raw.get("knowledge_examples", "đổi trả, bảo hành, vận chuyển, thanh toán..."),
        freeship_threshold=config.FREESHIP_THRESHOLD if is_default else int(raw.get("freeship_threshold", 500000)),
        size_advice=bool(raw.get("size_advice", False)),
        size_exchange_note=raw.get("size_exchange_note", ""),
        product_keywords=[_norm(k) for k in keywords],
        defect_keywords=[_norm(k) for k in raw.get("defect_keywords", [])],
        synonyms=raw.get("synonyms", {}),
        storefront=raw.get("storefront", {}),
        demo_accounts=raw.get("demo_accounts", []),
        is_default=is_default,
        raw=raw,
    )


def list_shops() -> list[dict[str, Any]]:
    out = []
    for cfg in sorted(SHOPS_DIR.glob("*/shop.json")):
        if cfg.parent.name.startswith("_"):
            continue  # _template
        try:
            raw = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"id": cfg.parent.name, "name": raw.get("name", cfg.parent.name),
                    "tagline": raw.get("storefront", {}).get("tagline", "")})
    return out


_ACTIVE: ShopProfile | None = None


def active() -> ShopProfile:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = load_profile(os.getenv("SHOP", DEFAULT_SHOP))
    return _ACTIVE


def activate(shop_id: str) -> ShopProfile:
    """Bật shop khác (dùng khi server đang chạy; người gọi tự tạo lại agent)."""
    global _ACTIVE
    _ACTIVE = load_profile(shop_id)
    return _ACTIVE
