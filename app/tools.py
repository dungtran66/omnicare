"""Tầng công cụ (tool layer) theo phong cách MCP: mỗi tool có name, description, inputSchema (JSON Schema).

Registry này được dùng cho:
  * Agent nội bộ (function calling của LLM).
  * Endpoint `/mcp` (JSON-RPC 2.0: `tools/list`, `tools/call`) để client MCP bên ngoài gọi.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .rag import BM25Retriever, normalize, tokenize
from .shop import active as active_shop


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., dict[str, Any]]
    requires_confirmation: bool = False  # tool có tác dụng phụ (Human-in-the-loop)

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "requiresConfirmation": self.requires_confirmation,
        }

    def openai_spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.input_schema}}


class ToolError(Exception):
    pass


def _validate(schema: dict[str, Any], args: dict[str, Any]) -> None:
    """Validate tối giản theo JSON Schema (required + type + pattern) để chặn tham số bậy từ LLM."""
    if not isinstance(args, dict):
        raise ToolError("arguments phải là object")
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args or args[req] in (None, ""):
            raise ToolError(f"thiếu tham số bắt buộc: {req}")
    for key, val in args.items():
        if key not in props:
            raise ToolError(f"tham số không hợp lệ: {key}")
        if val is None:
            continue
        p = props[key]
        if p.get("type") == "string":
            if not isinstance(val, str):
                raise ToolError(f"{key} phải là string")
            if "maxLength" in p and len(val) > p["maxLength"]:
                raise ToolError(f"{key} quá dài")
            if "pattern" in p and not re.fullmatch(p["pattern"], val):
                raise ToolError(f"{key} sai định dạng")
            if "enum" in p and val not in p["enum"]:
                raise ToolError(f"{key} phải thuộc {p['enum']}")


class ToolRegistry:
    def __init__(self, retriever: BM25Retriever | None = None, store: Any | None = None):
        self.retriever = retriever or BM25Retriever()
        self.store = store
        self.last_image_url: str | None = None
        seed_orders: dict[str, dict] = json.loads((active_shop().data_dir / "orders.json").read_text(encoding="utf-8"))
        if self.store is not None:
            db_orders = self.store.load_orders()
            # Đơn mẫu: giữ trạng thái/địa chỉ đã đổi trong DB, nhưng tên sản phẩm & giá luôn theo orders.json
            for oid, seed in seed_orders.items():
                if oid in db_orders:
                    db_orders[oid] = {**db_orders[oid], "product": seed["product"], "total": seed["total"]}
            self.orders = {**seed_orders, **db_orders}
        else:
            self.orders = seed_orders
        prod_file = active_shop().data_dir / "products.json"
        self.products: list[dict] = json.loads(prod_file.read_text(encoding="utf-8")) if prod_file.exists() else []
        self.tickets: dict[str, dict] = self.store.load_tickets() if self.store is not None else {}
        self.tools: dict[str, Tool] = {}
        self.restock_subscriptions: list[dict[str, Any]] = []
        self._register_defaults()

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def list_specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def openai_specs(self) -> list[dict[str, Any]]:
        return [t.openai_spec() for t in self.tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"tool không tồn tại: {name}")
        _validate(tool.input_schema, args)
        return tool.handler(**args)

    # ---------------- 11 MCP tools mặc định ----------------
    def _register_defaults(self) -> None:
        self.register(Tool(
            name="search_knowledge_base",
            description="Tìm thông tin trong kho tri thức Hybrid RAG (chính sách đổi trả, vận chuyển, bảo hành, thanh toán, bảng size).",
            input_schema={"type": "object", "properties": {"query": {"type": "string", "maxLength": 500}},
                          "required": ["query"]},
            handler=lambda query: {"results": self.retriever.search(query)},
        ))
        self.register(Tool(
            name="get_order_status",
            description="Tra cứu trạng thái đơn hàng theo mã đơn (dạng OC + 5 chữ số, ví dụ OC10001).",
            input_schema={"type": "object",
                          "properties": {"order_id": {"type": "string", "pattern": r"OC\d{5}"}},
                          "required": ["order_id"]},
            handler=self._get_order_status,
        ))
        self.register(Tool(
            name="create_support_ticket",
            description="Tạo phiếu hỗ trợ khi khách gặp sự cố cần bộ phận chuyên trách xử lý (lỗi sản phẩm, khiếu nại, hoàn tiền).",
            input_schema={"type": "object", "properties": {
                "summary": {"type": "string", "maxLength": 500},
                "category": {"type": "string", "enum": ["return", "refund", "warranty", "complaint", "other"]},
                "order_id": {"type": "string", "pattern": r"OC\d{5}"}},
                "required": ["summary", "category"]},
            handler=self._create_ticket,
            requires_confirmation=False,
        ))
        self.register(Tool(
            name="handoff_to_human",
            description="Chuyển cuộc hội thoại cho nhân viên CSKH khi khách yêu cầu, bức xúc, hoặc agent không đủ thông tin.",
            input_schema={"type": "object", "properties": {"reason": {"type": "string", "maxLength": 300}},
                          "required": ["reason"]},
            handler=lambda reason: {"handoff": True, "reason": reason, "queue": "tier1", "eta_minutes": 5},
        ))
        self.register(Tool(
            name="search_products",
            description=f"Tra cứu giá bán, màu sắc và danh mục sản phẩm {active_shop().name}.",
            input_schema={"type": "object", "properties": {
                "query": {"type": "string", "maxLength": 300},
                "size": {"type": "string", "maxLength": 20}},
                "required": ["query"]},
            handler=self._search_products,
        ))
        self.register(Tool(
            name="check_inventory",
            description="Kiểm tra chi tiết số lượng tồn kho theo sản phẩm, kích cỡ (S/M/L/XL) và màu sắc.",
            input_schema={"type": "object", "properties": {
                "product": {"type": "string", "maxLength": 300},
                "size": {"type": "string", "maxLength": 20},
                "color": {"type": "string", "maxLength": 50}},
                "required": ["product"]},
            handler=self._check_inventory,
        ))
        self.register(Tool(
            name="cancel_order",
            description="Hủy đơn hàng đang xử lý hoặc đang chuẩn bị hàng (bắt buộc khách xác nhận trước khi thực thi).",
            input_schema={"type": "object", "properties": {
                "order_id": {"type": "string", "pattern": r"OC\d{5}"},
                "reason": {"type": "string", "maxLength": 300}},
                "required": ["order_id"]},
            handler=self._cancel_order,
            requires_confirmation=True,
        ))
        self.register(Tool(
            name="update_shipping_address",
            description="Đổi địa chỉ giao hàng cho đơn chưa giao (bắt buộc khách xác nhận trước khi thực thi).",
            input_schema={"type": "object", "properties": {
                "order_id": {"type": "string", "pattern": r"OC\d{5}"},
                "new_address": {"type": "string", "maxLength": 300}},
                "required": ["order_id", "new_address"]},
            handler=self._update_shipping_address,
            requires_confirmation=True,
        ))
        self.register(Tool(
            name="request_refund",
            description="Gửi yêu cầu hoàn tiền cho đơn hàng đủ điều kiện (bắt buộc khách xác nhận trước khi thực thi).",
            input_schema={"type": "object", "properties": {
                "order_id": {"type": "string", "pattern": r"OC\d{5}"},
                "reason": {"type": "string", "maxLength": 300}},
                "required": ["order_id", "reason"]},
            handler=self._request_refund,
            requires_confirmation=True,
        ))
        self.register(Tool(
            name="recommend_size",
            description="Gợi ý size S/M/L/XL theo chiều cao (cm) và cân nặng (kg) dựa trên bảng size của shop, kèm tồn kho size đó nếu có tên sản phẩm.",
            input_schema={"type": "object", "properties": {
                "height_cm": {"type": "number"},
                "weight_kg": {"type": "number"},
                "product": {"type": "string", "maxLength": 300},
                "gender": {"type": "string", "enum": ["nam", "nu", ""]}},
                "required": ["height_cm", "weight_kg"]},
            handler=self._recommend_size,
        ))
        self.register(Tool(
            name="subscribe_restock",
            description="Đăng ký nhận thông báo khi sản phẩm/size hết hàng có hàng trở lại.",
            input_schema={"type": "object", "properties": {
                "product": {"type": "string", "maxLength": 300},
                "size": {"type": "string", "maxLength": 20},
                "contact": {"type": "string", "maxLength": 100}},
                "required": ["product", "contact"]},
            handler=self._subscribe_restock,
        ))

    def _get_order_status(self, order_id: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if not order:
            return {"found": False, "order_id": order_id}
        safe = {k: v for k, v in order.items() if k not in {"phone", "address", "email"}}
        return {"found": True, "order_id": order_id, **safe}

    def _create_ticket(self, summary: str, category: str, order_id: str | None = None) -> dict[str, Any]:
        tid = "TK-" + uuid.uuid4().hex[:6].upper()
        now = time.time()
        sla_hours = 4 if category in ("warranty", "refund") else 24
        default_assignee = {
            "warranty": "Linh (Bảo hành)",
            "refund": "Minh (Tài chính)",
            "return": "Hùng (Kho vận)",
            "complaint": "Lan (CSKH Tier-2)",
        }.get(category, "Chưa phân công")
        self.tickets[tid] = {
            "ticket_id": tid,
            "summary": summary,
            "category": category,
            "order_id": order_id,
            "created_at": now,
            "sla_hours": sla_hours,
            "sla_deadline": now + sla_hours * 3600,
            "assignee": default_assignee,
            "status": "open",
            "image_url": self.last_image_url,
        }
        if self.store is not None:
            self.store.save_ticket(tid, self.tickets[tid])
        return {"ticket_id": tid, "status": "open", "sla_hours": sla_hours, "assignee": default_assignee}

    # Bảng size lấy từ data/kb/thoi_trang_size.md (nam/unisex và nữ)
    SIZE_CHART = [("S", 155, 165, 45, 54), ("M", 163, 172, 55, 64), ("L", 170, 178, 65, 74), ("XL", 175, 185, 75, 86)]
    SIZE_CHART_F = [("S", 148, 156, 40, 46), ("M", 155, 162, 47, 52), ("L", 160, 167, 53, 59), ("XL", 165, 173, 60, 67)]

    @staticmethod
    def is_free_size(p: dict[str, Any]) -> bool:
        """Phụ kiện / sản phẩm một cỡ (kính, mũ...): không tư vấn size."""
        keys = [k for k in (p.get("sizes") or {})]
        return bool(p.get("accessory")) or (len(keys) == 1 and keys[0].lower().replace(" ", "") == "freesize")

    def _recommend_size(self, height_cm: float, weight_kg: float, product: str | None = None,
                        gender: str | None = None) -> dict[str, Any]:
        h, w = float(height_cm), float(weight_kg)
        prod = None
        if product:
            found = self._search_products(product)["products"]
            prod = found[0] if found else None
        if prod and self.is_free_size(prod):
            return {"height_cm": h, "weight_kg": w, "free_size": True, "product": prod["name"],
                    "product_id": prod["id"], "size_stock": sum((prod.get("sizes") or {}).values())}
        g = (gender or "").lower()
        if g not in ("nam", "nu"):
            g = prod.get("gender", "") if prod and prod.get("gender") in ("nam", "nu") else "nam"
        chart = self.SIZE_CHART_F if g == "nu" else self.SIZE_CHART
        order = [c[0] for c in chart]

        def pick(val: float, lo_i: int, hi_i: int) -> str:
            for row in chart:
                if row[lo_i] <= val <= row[hi_i]:
                    return row[0]
            return chart[0][0] if val < chart[0][lo_i] else chart[-1][0]

        by_h, by_w = pick(h, 1, 2), pick(w, 3, 4)
        size = by_h if order.index(by_h) >= order.index(by_w) else by_w
        out_of_chart = h < chart[0][1] or h > chart[-1][2] or w < chart[0][3] or w > chart[-1][4]
        res: dict[str, Any] = {"height_cm": h, "weight_kg": w, "size_by_height": by_h, "size_by_weight": by_w,
                               "recommended_size": size, "between_sizes": by_h != by_w, "out_of_chart": out_of_chart,
                               "chart": "nữ" if g == "nu" else "nam"}
        if prod:
            res.update(product=prod["name"], product_id=prod["id"],
                       size_stock=(prod.get("sizes") or {}).get(size, 0))
        return res

    def _subscribe_restock(self, product: str, contact: str, size: str | None = None) -> dict[str, Any]:
        res = self._search_products(product, size=size)
        p = res["products"][0] if res["products"] else None
        sub = {
            "subscription_id": "RS-" + uuid.uuid4().hex[:6].upper(),
            "product": p["name"] if p else product,
            "size": (size or "").upper() or None,
            "contact": contact,
            "status": "registered",
            "created_at": time.time(),
        }
        self.restock_subscriptions.append(sub)
        return {k: v for k, v in sub.items() if k != "created_at"}

    def _search_products(self, query: str, size: str | None = None) -> dict[str, Any]:
        q_tokens = set(tokenize(query))
        scored = []
        for p in self.products:
            p_tokens = set(tokenize(f"{p['name']} {p['category']} {' '.join(p.get('colors', []))}"))
            overlap = len(q_tokens & p_tokens)
            if overlap > 0 or normalize(query) in normalize(p["name"]):
                scored.append((overlap, p))
        scored.sort(key=lambda x: -x[0])
        matches = []
        for _, p in scored[:3]:
            item = dict(p)
            if self.is_free_size(p):
                item["free_size"] = True
                item["size_note"] = "Sản phẩm một cỡ (free size), không có size S/M/L/XL, không cần chọn size."
            elif size:
                su = size.strip().upper()
                item["requested_size"] = su
                item["requested_size_stock"] = p.get("sizes", {}).get(su, 0)
            matches.append(item)
        return {"found": bool(matches), "count": len(matches), "products": matches}

    def _check_inventory(self, product: str, size: str | None = None, color: str | None = None) -> dict[str, Any]:
        res = self._search_products(product, size=size)
        if not res["products"]:
            return {"found": False, "product": product, "in_stock": False}
        p = res["products"][0]
        sizes = p.get("sizes", {})
        total_stock = sum(sizes.values())
        su = size.strip().upper() if size else None
        free = self.is_free_size(p)
        if free:
            su = None  # kính/phụ kiện một cỡ: bỏ qua size khách hỏi
        size_stock = sizes.get(su, 0) if su else total_stock
        color_ok = True
        if color:
            color_ok = any(normalize(color) in normalize(c) for c in p.get("colors", []))
        return {
            "found": True,
            "product_id": p["id"],
            "name": p["name"],
            "price": p["price"],
            "sizes": sizes,
            "colors": p.get("colors", []),
            "requested_size": su,
            "free_size": free,
            "requested_color": color,
            "stock": size_stock,
            "in_stock": size_stock > 0 and color_ok,
        }

    def _cancel_order(self, order_id: str, reason: str = "Khách yêu cầu hủy") -> dict[str, Any]:
        order = self.orders.get(order_id)
        if not order:
            return {"cancelled": False, "order_id": order_id, "message": f"Không tìm thấy đơn hàng {order_id}."}
        if order["status"] in ("Đã giao thành công", "Đã hủy"):
            return {"cancelled": False, "order_id": order_id,
                    "message": f"Đơn hàng {order_id} đang ở trạng thái '{order['status']}' nên không thể hủy."}
        order["status"] = "Đã hủy"
        if self.store is not None:
            self.store.save_order(order_id, order)
        return {"cancelled": True, "order_id": order_id, "product": order.get("product"), "reason": reason,
                "status": "Đã hủy"}

    def _update_shipping_address(self, order_id: str, new_address: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if not order:
            return {"updated": False, "order_id": order_id, "message": f"Không tìm thấy đơn hàng {order_id}."}
        if order["status"] in ("Đã giao thành công", "Đã hủy"):
            return {"updated": False, "order_id": order_id,
                    "message": f"Đơn hàng {order_id} ở trạng thái '{order['status']}' nên không thể đổi địa chỉ."}
        order["address"] = new_address
        if self.store is not None:
            self.store.save_order(order_id, order)
        return {"updated": True, "order_id": order_id, "product": order.get("product"), "new_address": new_address}

    def _request_refund(self, order_id: str, reason: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if not order:
            return {"refund_requested": False, "order_id": order_id, "message": f"Không tìm thấy đơn hàng {order_id}."}
        tk = self._create_ticket(summary=f"Yêu cầu hoàn tiền đơn {order_id}: {reason}", category="refund", order_id=order_id)
        return {
            "refund_requested": True,
            "order_id": order_id,
            "product": order.get("product"),
            "amount": order.get("total", 0),
            "ticket_id": tk["ticket_id"],
            "sla_hours": tk["sla_hours"],
        }

    def update_ticket_status(self, ticket_id: str, status: str | None = None, assignee: str | None = None) -> dict[str, Any] | None:
        tk = self.tickets.get(ticket_id)
        if not tk:
            return None
        if status is not None:
            if status not in ("open", "in_progress", "resolved"):
                raise ToolError("status phải là open, in_progress hoặc resolved")
            tk["status"] = status
        if assignee is not None:
            tk["assignee"] = assignee
        if self.store is not None:
            self.store.save_ticket(ticket_id, tk)
        return tk
