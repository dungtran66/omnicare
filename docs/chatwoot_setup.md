# Ghép OmniCare làm Agent Bot cho Chatwoot

Chatwoot lo phần hộp thư đa kênh (Website, Zalo/Facebook/WhatsApp, Email…) và giao diện cho nhân viên.
OmniCare là "bộ não" trả lời tự động, có guardrails, MCP tools, xác nhận trước hành động nhạy cảm và evals.

```
Khách ──► Chatwoot inbox ──(webhook message_created)──► OmniCare /api/integrations/chatwoot
                ▲                                              │ agent.handle()
                └──────(API: gửi tin outgoing / chuyển "open")─┘
```

## 1. Chạy Chatwoot + OmniCare bằng Docker

```bash
cp chatwoot.env.example .env.chatwoot          # rồi đổi SECRET_KEY_BASE (openssl rand -hex 64)
docker compose -f docker-compose.chatwoot.yml run --rm rails bundle exec rails db:chatwoot_prepare
docker compose -f docker-compose.chatwoot.yml up -d
```

Mở http://localhost:3000, tạo tài khoản quản trị đầu tiên và một Inbox (ví dụ **Website**).

## 2. Tạo Agent Bot trỏ về OmniCare

1. Vào **Super Admin** (http://localhost:3000/super_admin) → **Agent Bots** → **New**.
2. Điền:
   - Name: `OmniCare AI`
   - Outgoing URL: `http://omnicare:8000/api/integrations/chatwoot`
     (nếu OmniCare chạy ngoài docker: `http://host.docker.internal:8000/api/integrations/chatwoot`)
3. Lưu lại và sao chép **Access Token** của bot.
4. Vào **Settings → Inboxes → (inbox của bạn) → Bot Configuration**, chọn `OmniCare AI`.

## 3. Cấu hình token cho OmniCare

Cách 1: biến môi trường khi chạy compose

```bash
CHATWOOT_BOT_TOKEN=<access_token_của_bot> docker compose -f docker-compose.chatwoot.yml up -d omnicare
```

Cách 2: gọi API lúc đang chạy

```bash
curl -X POST http://localhost:8000/api/integrations/config \
  -H "Content-Type: application/json" \
  -d '{"chatwoot_url": "http://rails:3000", "chatwoot_api_token": "<access_token_của_bot>"}'
```

## 4. Cách AI và nhân viên phối hợp

| Tình huống trong Chatwoot | OmniCare làm gì |
|---|---|
| Hội thoại mới, trạng thái `pending` (bot đang giữ) | AI trả lời bằng tin `outgoing` |
| AI quyết định chuyển nhân viên (khách yêu cầu, khách bức xúc) | Trả lời khách rồi gọi `toggle_status` → `open` để nhân viên nhận |
| Hội thoại `open` hoặc đã được gán cho nhân viên | AI **im lặng**, chỉ ghi nhận tin nhắn vào trang Admin |
| Nhân viên chuyển về `pending` hoặc đóng (`resolved`) | AI được phép trả lời lại ở tin nhắn tiếp theo |

Liên hệ có số điện thoại trùng với đơn hàng được coi là **thành viên** và chỉ được tra cứu, thao tác trên đơn của chính mình.
Liên hệ không có số điện thoại là **khách vãng lai**: chỉ được tư vấn sản phẩm, size và chính sách.

## 5. Kiểm tra nhanh không cần Chatwoot

```bash
curl -X POST http://localhost:8000/api/integrations/chatwoot -H "Content-Type: application/json" -d '{
  "event": "message_created", "message_type": "incoming", "content": "Cao 1m68 nặng 60kg mặc size gì?",
  "conversation": {"id": 501, "status": "pending"}, "account": {"id": 1},
  "sender": {"name": "Chị Lan", "phone_number": "0918222333"}
}'
```

Kết quả có `"reply"`, `"status"`, `"intent"`, và `"pushed_to_chatwoot_api": false` nếu chưa cấu hình token.
