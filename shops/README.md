# Các shop demo của OmniCare

Mỗi thư mục là một cửa hàng. Agent, RAG, công cụ và trang bán hàng đọc dữ liệu của shop đang bật.

| Shop | Ngành | Trang bán hàng | Dữ liệu |
|---|---|---|---|
| `noriva` (mặc định) | Thời trang | `/store` | thư mục `data/` (giữ nguyên như cũ) |
| `techzone` | Điện thoại, laptop, phụ kiện | `/shop` | `shops/techzone/` |

## Đổi shop đang demo

- Trên trang Admin: **Kênh kết nối → Cửa hàng đang demo**, chọn shop rồi bấm **Chuyển**.
- Hoặc chạy server với biến môi trường: `SHOP=techzone`.
- Hoặc gọi API: `POST /api/shops/active` với `{"shop_id": "techzone"}`.

Mỗi shop (trừ NORIVA) có cơ sở dữ liệu riêng `omnicare_traces_<id>.db`, nên đơn hàng, phiếu hỗ trợ và hội thoại không lẫn nhau.

## Tạo shop mới

1. Chép thư mục `_template` thành `shops/<id>/` (id viết thường, không dấu, vd `mypham`).
2. Sửa `shop.json`:
   - `name`, `intro`, `intent_label`: tên và mô tả shop, đưa vào lời dặn cho AI.
   - `freeship_threshold`: ngưỡng miễn phí vận chuyển (đ).
   - `size_advice`: `true` nếu bán quần áo và muốn tư vấn size theo chiều cao/cân nặng.
   - `variant_label`: tên gọi biến thể sản phẩm ("Size", "Phiên bản", "Dung tích"...).
   - `product_keywords`: từ khóa tên sản phẩm, viết **không dấu** (vd `"son moi"`, `"kem chong nang"`). Để trống thì tự lấy từ tên sản phẩm.
   - `defect_keywords`: lỗi riêng ngành để nhận ra khiếu nại, không dấu (vd `"kich ung"`, `"het han"`).
   - `synonyms`: từ đồng nghĩa giúp RAG tìm đúng tài liệu, không dấu, cụm từ nối bằng `_` (vd `"di_ung": ["kich_ung", "doi_tra"]`).
   - `storefront`: khẩu hiệu, màu chủ đạo, biểu tượng cho trang `/shop`.
   - `demo_accounts`: tài khoản thành viên để thử (mật khẩu demo `123456`), số điện thoại khớp với `orders.json`.
3. Sửa `products.json` (`sizes` là tồn kho theo biến thể) và `orders.json` (mã đơn dạng `OC` + 5 chữ số).
4. Viết chính sách vào `kb/*.md`, mỗi mục bắt đầu bằng `## `. Cũng có thể upload PDF/Word/Excel ở trang Knowledge Base sau khi bật shop.
5. Bật shop và thử trên `/shop`.
