# OmniCare — AI Agent chăm sóc khách hàng cho shop quần áo online

OmniCare là trợ lý AI trả lời khách ngay trên trang bán hàng: tư vấn sản phẩm, chọn size, phối đồ, tra đơn, đổi trả, hủy đơn.
Câu nào AI không chắc hoặc khách cần người thật thì chuyển cho nhân viên, và nhân viên tiếp quản trên trang Admin.

Bản demo chạy trên shop thời trang **NORIVA** với hai trang:

| Trang | Địa chỉ | Dành cho |
|---|---|---|
| Trang bán hàng | `/store` | Khách mua hàng: xem sản phẩm, giỏ hàng, đặt hàng, chat với AI |
| Admin CRM | `/` | Nhân viên và quản lý: hộp thư hội thoại, tiếp quản, phiếu hỗ trợ, theo dõi chất lượng AI |

Ngoài thời trang, hệ thống chạy được cho ngành khác chỉ bằng cách thay dữ liệu shop (xem mục 6, ví dụ shop đồ điện tử `techzone`).

Hệ thống có đủ 6 thành phần theo yêu cầu: **Model · Workflow agent · RAG + MCP tools · Guardrails · Observability · Evals**.

- Kiến trúc, luồng xử lý, quyết định thiết kế và kết quả evals: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Kịch bản video demo: [`docs/VIDEO_DEMO.md`](docs/VIDEO_DEMO.md)

---

## 1. Chạy thử trong 3 bước

Cần Python 3.11 trở lên.

```bash
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
python -m uvicorn app.main:app --port 8000
```

Mở trình duyệt:
- Trang bán hàng: http://localhost:8000/store
- Admin: http://localhost:8000

File `.env` mặc định để `LLM_PROVIDER=mock`, nên hệ thống **chạy được ngay mà không cần khóa API**: AI trả lời theo luật và tài liệu.
Muốn dùng model thật thì xem mục 4.

### Tài khoản demo

| Vai trò | Đăng nhập | Mật khẩu | Ghi chú |
|---|---|---|---|
| Admin | `admin` | `admin123` | Nhân viên khác: `lan`, `linh` (mật khẩu `123456`) |
| Khách thành viên | `0918222333` | `123456` | Nguyễn Thu Lan, có đơn OC10005 |
| Khách thành viên | `0905666777` | `123456` | Minh Khoa, đơn OC10006 |
| Khách thành viên | `0977123456` | `123456` | Hà My, đơn OC10007 |

Đổi mật khẩu Admin: `python -m app.auth set admin <mật khẩu mới>`.

---

## 2. AI làm được gì

**Với khách vãng lai** (chưa đăng nhập):
- Hỏi chính sách đổi trả, vận chuyển, bảo hành, thanh toán. AI trả lời dựa trên tài liệu của shop, không bịa.
- Hỏi giá, tồn kho, size còn hàng. Bấm "Hỏi AI" trên thẻ sản phẩm để hỏi riêng về món đó.
- Tư vấn size theo chiều cao, cân nặng, có bảng size riêng cho nam và nữ.
- Phối đồ theo dịp ("đi biển 3 ngày nên mặc gì"), kèm nút thêm cả bộ vào giỏ.
- Hỏi tổng tiền giỏ hàng, phí ship, còn thiếu bao nhiêu để được miễn phí ship.
- Gửi ảnh để tìm mẫu tương tự trong cửa hàng.

**Với khách đã đăng nhập** (thêm):
- Tra đơn của chính mình. Hỏi đơn của người khác sẽ bị từ chối.
- Hủy đơn, đổi địa chỉ, yêu cầu hoàn tiền. AI luôn hỏi lại và chỉ làm khi khách bấm **Xác nhận**.
- Gửi ảnh sản phẩm lỗi để tạo phiếu hỗ trợ.
- AI nhớ size, số đo và món khách quan tâm cho lần chat sau.

**Với nhân viên** (trang Admin):
- Hộp thư chia nhóm Thành viên / Khách vãng lai, có trạng thái 🤖 AI / ⏳ Chờ NV / 👩‍💼 NV xử lý và thời gian chờ.
- Khi AI chuyển cho người thật, Admin có sẵn **tóm tắt bàn giao**: khách là ai, vấn đề gì, AI đã làm gì, mức độ gấp.
- **Tiếp quản** hội thoại (AI ngừng trả lời), **Copilot** gợi ý câu trả lời và viết lại cho lịch sự hoặc ngắn gọn hơn.
- **Đóng** hội thoại: khách được mời chấm sao; nhắn tiếp sẽ mở hội thoại mới.
- Chấm chất lượng hội thoại bằng AI, phiếu hỗ trợ, báo cáo tuần, Auto-FAQ gợi ý tài liệu còn thiếu.

**Với quản lý** (Observability):
- Xem từng tin nhắn đi qua những bước nào, mất bao lâu, dùng model nào (Traces).
- Chỉ số: tỉ lệ AI tự xử lý, tỉ lệ chuyển nhân viên, độ trễ, chi phí model, điểm hài lòng của khách.

---

## 3. Sáu thành phần chính

| Thành phần | File | Làm gì |
|---|---|---|
| **Model** | `app/llm.py` | Kết nối GPT, Gemini, Claude, Ollama hoặc Mock (chạy offline). Tự chuyển sang model dự phòng khi lỗi, thử lại khi bị giới hạn lượt (429), giới hạn số lượt gọi mỗi khách mỗi ngày. Hỗ trợ gọi công cụ và đọc ảnh |
| **Workflow agent** | `app/agent.py` | Mỗi tin nhắn đi qua: kiểm tra an toàn đầu vào → đọc ảnh (nếu có) → nhận diện ý định → kiểm tra quyền → gọi công cụ / tra tài liệu → model viết câu trả lời → kiểm tra an toàn đầu ra → lưu vết |
| **RAG + MCP tools** | `app/rag.py`, `app/tools.py` | Tìm tài liệu bằng 3 lớp (từ khóa BM25, vector, xếp hạng lại). Nạp được PDF, Word, Excel, CSV, Markdown. 11 công cụ (tra đơn, tồn kho, hủy đơn, hoàn tiền…), gọi được từ ngoài qua chuẩn MCP tại `POST /mcp` |
| **Guardrails** | `app/guardrails.py` | Chặn lệnh lừa AI (prompt injection), che số điện thoại/email/số thẻ, xử lý lời lẽ thô tục, kéo về chủ đề mua sắm. Ở đầu ra: chặn lộ prompt, cam kết sai chính sách, con số không có căn cứ, giá sai so với danh mục |
| **Observability** | `app/observability.py` | Ghi trace và các bước con vào SQLite; tính độ trễ p50/p95, tỉ lệ xử lý, token, chi phí theo từng model, điểm hài lòng |
| **Evals** | `evals/` | 62 tình huống kiểm thử chính + 12 tình huống cần model thật; chấm bằng luật hoặc để GPT làm giám khảo (thang 5 điểm) |

Hủy đơn, hoàn tiền, đổi địa chỉ **không bao giờ để model tự làm**: model chỉ được dùng các công cụ chỉ đọc, các thao tác thay đổi dữ liệu luôn cần khách xác nhận.

---

## 4. Dùng model thật

Sửa file `.env` (không commit file này). Ví dụ với API tương thích OpenAI:

```ini
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=<khóa của bạn>
LLM_MODEL=gpt-4o-mini
LLM_FALLBACK_MODELS=          # tùy chọn: model dự phòng, cách nhau bằng dấu phẩy
MODEL_CALLS_PER_DAY=40        # giới hạn lượt gọi model mỗi khách mỗi ngày; 0 = không giới hạn
```

Cấu hình mẫu cho Claude và Gemini có sẵn trong `.env.example`. Có thể đổi model khi đang chạy qua `POST /api/llm/config` (cần đăng nhập Admin).
Nếu model lỗi, hết lượt hoặc thiếu khóa, agent tự rơi về Mock để khách vẫn nhận được câu trả lời.

---

## 5. Kiểm thử

```bash
pytest tests -q                                                       # 64 unit test, luôn chạy Mock, không tốn API
python -m evals.run_evals --threshold 0.95                            # 62 tình huống, chấm bằng luật (cổng CI)
python -m evals.run_evals --judge llm --delay 2                       # như trên nhưng GPT trả lời và GPT chấm
python -m evals.run_evals --dataset evals/dataset_gpt.jsonl --judge llm   # 12 tình huống tính năng GPT (đọc ảnh, tự gọi công cụ…)
```

Kết quả gần nhất (23/11/2026): 64/64 unit test, 62/62 tình huống Mock đạt. GPT chấm trung bình 4.36/5, nhưng lượt đó API quá tải
nên chỉ khoảng 21% câu do GPT trả lời; cần chạy lại khi API ổn định. Chi tiết ở `docs/ARCHITECTURE.md` mục 6.

GitHub Actions (`.github/workflows/evals.yml`) chạy unit test và 62 tình huống Mock ở mỗi lần push.

---

## 6. Cấu trúc thư mục

```text
omnicare/
├── app/
│   ├── main.py            # FastAPI: API, streaming, WebSocket /ws/crm, webhook
│   ├── agent.py           # Workflow agent
│   ├── llm.py             # Kết nối model
│   ├── rag.py, tools.py   # Tra tài liệu + 11 công cụ MCP
│   ├── guardrails.py      # Kiểm tra an toàn
│   ├── observability.py   # Trace, chỉ số, SQLite
│   ├── ai_features.py     # Tóm tắt bàn giao, Copilot, chấm chất lượng, báo cáo tuần
│   ├── autofaq.py         # Gợi ý tài liệu còn thiếu
│   ├── auth.py            # Đăng nhập Admin
│   ├── shop.py            # Hỗ trợ nhiều cửa hàng
│   └── static/            # store.html (bán hàng), index.html (Admin), shop.html, img/
├── data/                  # Tài liệu (kb/), sản phẩm, đơn hàng mẫu của NORIVA
├── shops/                 # Cửa hàng khác (vd techzone — đồ điện tử) và mẫu tạo shop mới
├── evals/                 # Bộ tình huống kiểm thử và chương trình chấm
├── tests/                 # Unit test
├── scripts/smoke_test.py  # Kiểm tra nhanh qua HTTP
└── docs/                  # Kiến trúc, kịch bản video
```

Đổi sang shop khác: đặt `SHOP=techzone` trong `.env`, hoặc chọn ở Admin → Kênh kết nối. Cách tạo shop mới: `shops/README.md`.

---

## 7. Kênh kết nối

| Kênh | Endpoint |
|---|---|
| Website | `POST /api/chat`, `POST /api/chat/stream` (trả lời dần từng chữ) |
| Telegram | `POST /api/integrations/telegram/webhook` |
| Zalo, Facebook | `POST /api/webhook/{kênh}` |
| Client MCP | `POST /mcp` (JSON-RPC 2.0) |
| Admin (cập nhật trực tiếp) | WebSocket `/ws/crm` |

Khách từ Zalo, Facebook, Telegram được coi là khách vãng lai (không tra được đơn).

---

## 8. Lưu ý bảo mật

- Không commit `.env`, file `*.db`, thư mục `backups/` và `data/admin_users.json` (đã có trong `.gitignore`).
- API quản trị yêu cầu token đăng nhập Admin (hết hạn sau 12 giờ). API khách (chat, đặt hàng) để mở.
- Giá đơn hàng do server tính theo danh mục, không tin số tiền trình duyệt gửi lên.
- Tóm tắt bàn giao, điểm chất lượng và trí nhớ về khách là dữ liệu nội bộ, không trả về cho khách.
- Mỗi lần mở trang bán hàng, khách vãng lai bắt đầu một hội thoại mới, nên người dùng chung máy không xem được lịch sử chat của nhau.
- Tài khoản khách trên trang demo lưu trong trình duyệt (`localStorage`); bản thật cần thay bằng hệ thống tài khoản phía server.
