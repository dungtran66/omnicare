# CLAUDE.md — OmniCare AI Agent & OMNIWEAR Storefront

File này cung cấp toàn bộ ngữ cảnh kiến trúc, lịch sử phát triển và lệnh vận hành để tiếp tục phát triển trên **Claude Code**.

## 1. Tổng quan dự án

**OmniCare** là hệ thống AI Agent hỗ trợ chăm sóc khách hàng đa kênh (lấy cảm hứng từ kiến trúc [TGO](https://github.com/tgoai/tgo)), tích hợp cùng website bán hàng thời trang **OMNIWEAR** (phong cách UNIQLO LifeWear, tông màu Xanh Royal Navy `#1e3a8a`).

Hệ thống bao gồm đầy đủ **6 thành phần MVP cốt lõi** và **4 nhóm nâng cấp Production**:
1. **Model Layer (`app/llm.py`)**: Hỗ trợ `MockLLM` (offline deterministic) và `OpenAICompatibleLLM` (OpenAI, Gemini, Groq, OpenRouter, Ollama). Có thể đổi model runtime qua `POST /api/llm/config`.
2. **Workflow Agent (`app/agent.py`)**: Đồ thị xử lý trạng thái (DAG):
   `guardrail:input (+ sentiment)` → `[vision:analyze_image nếu có ảnh]` → `classify_intent` → `[route & call MCP tools / Human-in-the-loop confirm]` → `llm:generate` → `guardrail:output` → `memory & trace`.
3. **Hybrid RAG & 11 MCP Tools (`app/rag.py`, `app/tools.py`)**:
   - **Hybrid RAG 3 tầng**: BM25 Lexical (kèm mở rộng từ đồng nghĩa tiếng Việt) + Char 3-gram Vector Cosine + Cross-Encoder Reranker (`RAG_MIN_SCORE = 2.0`).
   - Hỗ trợ upload & tự động chuyển đổi `.pdf`, `.docx`, `.xlsx`, `.csv`, `.md`, `.txt` sang Markdown chunks (`parse_document_to_markdown`).
   - **11 MCP Tools** (giao tiếp nội bộ & qua JSON-RPC 2.0 `POST /mcp`):
     - `search_knowledge_base`
     - `get_order_status`
     - `create_support_ticket` (tự động gán `assignee` và `sla_hours`)
     - `handoff_to_human`
     - `search_products`
     - `check_inventory`
     - `cancel_order` (`requires_confirmation=True` — Human-in-the-loop)
     - `update_shipping_address` (`requires_confirmation=True` — Human-in-the-loop)
     - `request_refund` (`requires_confirmation=True` — Human-in-the-loop)
     - `recommend_size` (gợi ý size theo chiều cao/cân nặng từ bảng size KB)
     - `subscribe_restock` (đăng ký báo khi có hàng lại)
4. **Guardrails 2 chiều (`app/guardrails.py`)**:
   - **Input**: Giới hạn độ dài, chặn Prompt Injection (Việt + Anh), chặn Off-topic, ẩn PII (`phone`, `email`, `card` theo thuật toán Luhn, `national_id`), phát hiện ngôn từ bức xúc (`abusive_language` → handoff), phân tích cảm xúc (`analyze_sentiment` từ `-1.0` đến `+1.0`).
   - **Output**: Chặn lộ system prompt, chặn cam kết vượt chính sách (`cam kết hoàn tiền 100%`...), kiểm tra số liệu không có căn cứ (`ungrounded_numbers`).
5. **Observability (`app/observability.py`)**:
   - Ghi nhận Trace & Span Waterfall lưu vào SQLite (`omnicare_traces.db`).
   - Thống kê Metrics: Latency `p50/p95`, `deflection_rate`, `handoff_rate`, `blocked_rate`, `estimated_cost_usd`, `negative_sentiment_count`, `feedback` (CSAT).
6. **Evals & LLM-as-a-Judge (`evals/run_evals.py`, `evals/dataset.jsonl`)**:
   - Bộ 50 kịch bản kiểm thử tự động + chấm điểm `llm_judge_score` (Faithfulness, Relevance, Tone/Safety trên thang 5.0).
   - GitHub Actions CI gate tại `.github/workflows/evals.yml`.

---

## 2. Cấu trúc thư mục

```text
omnicare/
├── app/
│   ├── config.py           # Đọc biến môi trường / .env
│   ├── llm.py              # MockLLM & OpenAICompatibleLLM + build_llm()
│   ├── rag.py              # Hybrid RAG 3 tầng + parser PDF/DOCX/XLSX/CSV/MD/TXT
│   ├── tools.py            # ToolRegistry (11 MCP tools + JSON Schema validator)
│   ├── auth.py             # Đăng nhập Admin phía server (PBKDF2, token Bearer, danh sách API bảo vệ)
│   ├── autofaq.py          # Auto-FAQ: lọc câu hỏi + model soạn FAQ nháp
│   ├── guardrails.py       # Input/Output guardrails + PII redaction + Sentiment
│   ├── agent.py            # OmniCareAgent DAG + Multimodal Vision + AI Copilot
│   ├── observability.py    # Trace, Span, SQLite TraceStore & Business Metrics
│   ├── main.py             # FastAPI endpoints, SSE streaming, Webhooks, WebSocket /ws/crm
│   └── static/
│       ├── index.html      # Giao diện Admin CRM (TGO Dark Theme UI, 5 trang)
│       └── store.html      # Website bán hàng demo NORIVA (giao diện OMNIWEAR cũ đổi tên; đăng ký/đăng nhập, Đơn hàng của tôi, chat OmniCare)
├── data/
│   ├── kb/                 # Tài liệu Markdown RAG (doi_tra, van_chuyen, bao_hanh, thanh_toan, thoi_trang_size)
│   ├── orders.json         # Đơn hàng mẫu OC10001 - OC10007
│   └── products.json       # Danh mục 8 sản phẩm thời trang OMNIWEAR + tồn kho theo size
├── evals/
│   ├── dataset.jsonl       # 50 test cases
│   └── run_evals.py        # Eval runner + LLM-as-a-Judge
├── tests/
│   ├── test_omnicare.py, test_shops.py, test_features.py  # 40 unit tests (pytest, luôn chạy Mock)
│   └── conftest.py         # ép LLM_PROVIDER=mock, tắt embedding, DB tạm
└── scripts/
    ├── deploy_hf.py        # Build static bundle (nhúng browser fallback engine) & deploy lên Hugging Face Static Space
    └── smoke_test.py       # End-to-end HTTP smoke test
```

---

## 3. Các lệnh thường dùng

```bash
# 1. Chạy server FastAPI ở local (http://localhost:8000 và http://localhost:8000/store)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Chạy 40 unit tests
pytest tests/test_omnicare.py -q

# 3. Chạy bộ 50 Evals + LLM-as-a-Judge (đặt LLM_PROVIDER=mock EMBEDDING_MODEL= để không tốn lượt API)
python -m evals.run_evals --threshold 0.95

# 4. Build và cập nhật lên Hugging Face Static Space (https://dungtran1606-omnicare-ai-agent.static.hf.space)
python scripts/deploy_hf.py <HF_WRITE_TOKEN>
```

---

## 4. Các quy ước thiết kế quan trọng (UI/UX Constraints)

Khi chỉnh sửa tiếp giao diện, cần giữ các quy ước mà người dùng đã yêu cầu:
1. **Không đặt link/nút dẫn sang trang Admin CRM (`index.html`) trên trang bán hàng (`store.html`)**: Trang `store.html` đóng vai trò là trang khách mua hàng độc lập.
2. **Không hiển thị các thẻ telemetry (`intent`, `status`, tên tool, thẻ nguồn RAG, latency, nút 👍/👎) bên dưới bong bóng chat** trên cả `index.html` và `store.html`. Các thông tin kỹ thuật đó chỉ hiển thị ở cột phải (`Visitor Profile -> AI INSIGHTS`) và trong tab `Observability -> Traces`.
3. **Khi sửa `app/static/index.html`, `app/static/store.html` hoặc logic backend**: Nếu muốn cập nhật lên link Cloud `https://dungtran1606-omnicare-ai-agent.static.hf.space`, hãy cập nhật đồng bộ trong `scripts/deploy_hf.py` và chạy `python scripts/deploy_hf.py <HF_TOKEN>`.
4. **Theme & Logo trang Admin (`index.html`)**: Logo OmniCare (bong bóng chat + trái tim + tia AI, nền gradient theo màu nhấn) nằm ở nav rail; bấm logo hoặc nút bảng màu (cuối nav rail) để mở Theme panel (Sáng/Tối/Hệ thống, 8 màu nhấn + màu tùy chọn, 5 kiểu nền tối: Midnight, Graphite, OLED, Forest, Plum). Lưu ở `localStorage['omnicare_admin_theme']`. Khi thêm CSS mới, **dùng biến `--accent`, `--accent-soft`, `--accent-text`, `--accent-glow`** thay vì mã màu xanh cố định để tự đổi theo theme. Script khởi tạo theme trong `<head>` phải viết trên 1 dòng (`<script>(...)</script>`) để không trùng marker `"<script>\n"` của `deploy_hf.py`.
5. **Bản trên Claude Artifact** (Claude tự cập nhật được): **2 link riêng (22/11/2026)**: Admin https://claude.ai/artifact/FNHEVRFd6JfMbb6D3AqMFr (từ `.hf_static_build/two/admin/index.html`, link "Mở trang bán hàng" trỏ thẳng URL store), bán hàng https://claude.ai/artifact/A3fDPqwJJWppoAEfytGewg (`two/store/index.html`); 2 origin khác nhau nên chat bên store không tự hiện bên Admin. Bản gộp cũ: https://claude.ai/artifact/CGswT5ofAHKM1iwHvGbCKW (đăng 22/11/2026, gồm `index.html` + `store.html` + `img/`; title trang đổi thành "OmniCare Admin CRM", link "Mở trang bán hàng" trỏ `store.html`); bản cũ https://claude.ai/artifact/VGSuF1qoLSLKuAPcVPSgHi. Riêng tư tới khi bấm Share. `deploy_hf.py` tự build bằng DB tạm (không đưa lịch sử chat local lên mạng). Bản đăng nằm ở `.hf_static_build/artifact/`: build bằng `HF_TOKEN= python scripts/deploy_hf.py` (nếu SQLite lỗi trên ổ mount thì build trên bản sao tạm), rồi sửa trên cả 2 file: `_EDGE_MODE = true` và bọc `localStorage.getItem("omnicare_custom_orders")` trong try/catch; riêng index.html đổi `<!doctype html>` → `<!DOCTYPE html>`.
6. **Phân loại khách & phân quyền AI**: `POST /api/chat` nhận thêm `customer` = `{type:"guest", guest_code}` hoặc `{type:"member", member_id, name, phone, order_ids}`. Khách vãng lai (định danh `Khách #XXXX`) chỉ được tư vấn sản phẩm/chính sách; các intent `order_status, cancel_order, update_address, request_refund, complaint, handoff, vision_defect` trả `status="login_required"` (flag `member_only`). Thành viên chỉ truy cập đơn của mình (`order_ids` hoặc trùng `phone` trong orders.json), đơn khác → `status="denied"` (flag `order_not_owned`). Không gửi `customer` = kênh nội bộ/evals, giữ hành vi cũ. Webhook Zalo/Facebook/Telegram mặc định là khách vãng lai. Logic nằm ở `OmniCareAgent._access_check` (app/agent.py) và `_accessEdge` (engine JS trong scripts/deploy_hf.py — f-string, nhớ nhân đôi `{}` và `\\`). Trang Admin chia danh sách thành 2 nhóm Thành viên / Khách vãng lai. 
7. **Website NORIVA (`store.html`)**: giữ bố cục trang OMNIWEAR cũ (người dùng chọn), thương hiệu đổi thành NORIVA ở mọi nơi (store, admin, agent, KB). Header: ô tìm kiếm gạch chân + 3 nút icon Đăng nhập / Đơn hàng / Giỏ hàng. Ảnh sản phẩm Unsplash tự rơi về hình SVG khi không tải được (`imgFallback`). Drawer giỏ/đơn có z-index 105 để nằm trên nút chat nổi. Không có link sang Admin. Tài khoản demo lưu `localStorage['noriva_accounts']` (mật khẩu `123456`: 0918222333 Thu Lan/M-1024/OC10005, 0905666777 Minh Khoa/M-1031/OC10006, 0977123456 Hà My/M-1047/OC10007). Đặt hàng bắt buộc đăng nhập, đơn mới được thêm vào `orderIds` của tài khoản. Chat gửi `customer` (guest: `Khách #XXXX` từ `noriva_guest_code`; member: `sess-m1024`…) và lưu hội thoại vào `localStorage['omnicare_store_sessions']`; Admin gộp vào danh sách qua `mergeStoreSessions()` + sự kiện `storage` (cập nhật trực tiếp khi 2 trang cùng origin). Bản gốc trước khi sửa: `backups/store.backup-omniwear.html`.

8. **Nâng cấp Agent (đợt A/B/C, 21/11/2026)**:
   - **Ý định mới** `size_advice`, `restock_notify` + giải quyết tham chiếu nhiều lượt (`OmniCareAgent._pre_route`: "hủy nó đi", "cái áo đó còn size L không"). Gợi ý mua kèm để đủ freeship (`_upsell_hint`, ngưỡng `FREESHIP_THRESHOLD`), guardrail đầu ra `price_mismatch` (`_price_mismatch`) so giá với `products.json`. Guardrails có thêm mẫu prompt injection trộn Anh-Việt/teencode.
   - **Embedding tùy chọn** cho RAG: đặt `EMBEDDING_MODEL` (+ `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, API tương thích OpenAI `/embeddings`, vd bge-m3/multilingual-e5); để trống thì dùng char 3-gram như cũ (`retriever.embedding_mode`).
   - **Engine trình duyệt**: các tính năng trên được viết ở `scripts/edge_extensions.js` (JS thuần, bọc `_handleChatEdge`/`_callToolEdge`), `deploy_hf.py` nối file này vào cuối engine. Sửa logic agent thì sửa cả Python lẫn file này.
   - **Chatwoot**: `/api/integrations/chatwoot` im lặng khi hội thoại `open`/đã gán nhân viên, xử lý `conversation_status_changed` (open → takeover, pending/resolved → trả lại AI). Token qua env `CHATWOOT_URL`, `CHATWOOT_API_TOKEN`. Chạy thử bằng `docker-compose.chatwoot.yml` + `chatwoot.env.example`, hướng dẫn ở `docs/chatwoot_setup.md`.
   - **Admin vận hành hội thoại** (`index.html`, lưu `localStorage['omnicare_conv_meta']`): trạng thái 🤖 AI / ⏳ Chờ NV / 👩‍💼 NV xử lý / ✅ Đã xong, bộ lọc theo trạng thái, badge chưa đọc, sắp xếp theo độ gấp, ghi chú nội bộ (khách không thấy), câu trả lời mẫu gõ `/`, chọn người phụ trách, nút "Đóng" gửi khảo sát CSAT (store hiển thị sao, lưu `sess.csat`), Copilot "Tạo phiếu từ hội thoại", bảng "Thống kê theo ngày" trong Observability → Metrics.
   - **Lưu ý build**: `deploy_hf.py` xóa sạch `.hf_static_build/` mỗi lần build (kể cả thư mục `artifact/`), nên tạo lại bản artifact sau khi build.
9. **Cập nhật mới nhất (Session hiện tại)**:
   - **Storefront (`app/static/store.html`)**: Dùng ảnh thật lưu nội bộ tại `app/static/img/` (`hero1.jpg`, `hero2.jpg`, `p1.jpg`–`p8.jpg`; `main.py` mount `/img` và `deploy_hf.py` copy `img/` vào `.hf_static_build/img/`). Đã xóa `.oc-head-sub` và `#ocWho` khỏi widget chat theo yêu cầu.
   - **SQLite Persistence (`app/observability.py`, `app/tools.py`)**: `TraceStore` (`omnicare_traces.db`) lưu thêm các bảng `orders`, `tickets`, `conversations`, `customers`, `kb_gaps_resolved`.
   - **Chat 2 chiều & Human Takeover (`app/static/index.html`, `app/static/store.html`, `app/main.py`)**: Phím `Enter` và nút chính trên trang Admin gọi `sendHumanReply()` (nhân viên trả lời trực tiếp với bong bóng chat bình thường, không đẩy sang phía khách), đồng bộ qua `localStorage` + WebSocket `/ws/crm` + `POST /api/conversations/{sid}/reply`; nút `🧪 Giả lập khách hỏi AI` dùng để test AI.
   - **Đăng nhập Admin phía server (`app/auth.py`)**: `POST /api/admin/login` → token Bearer (12 giờ, lưu trong bộ nhớ), tài khoản băm PBKDF2 trong `data/admin_users.json` (git-ignore; demo admin/admin123, lan/123456, linh/123456; đổi bằng `python -m app.auth set <user> <pass>`). Middleware `admin_guard` trả 401 cho API trong `auth.PROTECTED` (conversations list/reply/takeover, customers GET, llm config POST, kb upload/reload/gaps, copilot, integrations config, traces/metrics/state, evals, shops/active). API khách (chat, stream, orders/create, feedback, webhook, /mcp, GET conversation theo id, /ws/crm) vẫn mở. `index.html` bọc `window.fetch` để gắn token, gặp 401 thì hiện lại form đăng nhập; phiên lưu `localStorage['omnicare_admin_auth']` = `{user, name, token, loginAt}`. Bản tĩnh: engine JS trả token demo (`/api/admin/login` trong `deploy_hf.py`). `ADMIN_AUTH=0` để tắt.
   - **Ẩn box chat (`isHiddenChat`)**: khách vãng lai chỉ hiện nếu đang `NV xử lý`, hoặc nhắn sau `loginAt` và tin cuối < 10 phút (`GUEST_IDLE_MS`). Nút × trên từng box và "Ẩn tất cả" lưu `localStorage['omnicare_admin_hidden']` = `{id: thời điểm ẩn}`; box hiện lại khi khách nhắn tiếp. Chỉ ẩn khỏi danh sách, không xóa dữ liệu.
   - **RAG cho NORIVA**: KB `data/kb/` viết lại cho shop thời trang (bỏ nội dung điện thoại/laptop/OmniShop). `rag.py` thêm từ đồng nghĩa thời trang, cắt đoạn dài > `CHUNK_MAX_CHARS` (900) theo câu, rerank so khớp nguyên từ, và lọc `MIN_QUERY_COVERAGE` (0.34) để chặn khớp nhầm do bỏ dấu ("máy" ~ "may"). `render_context` (llm.py) bọc tài liệu trong thẻ `<document>`. Dùng Claude thì embedding không mượn `LLM_BASE_URL`/`LLM_API_KEY`. Engine JS trên bản tĩnh lấy chunk mới khi build nhưng chưa có bộ lọc coverage.
   - **Claude (`AnthropicLLM` trong app/llm.py)**: provider `anthropic` dùng SDK chính thức `anthropic` (`client.beta.messages.create`), model mặc định `claude-opus-5-5` (`config.CLAUDE_DEFAULT_MODEL`), đổi qua `LLM_MODEL`; `LLM_EFFORT` (mặc định medium). Bật `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`), cache system prompt, `json_mode` dùng structured outputs (`output_config.format`). Lỗi/thiếu khóa/refusal → `LLMError` → agent rơi về MockLLM (flag `llm_fallback`). Khóa: `ANTHROPIC_API_KEY` trong `.env`. Ước tính chi phí theo model ở `observability._PRICES`.
   - **Đa cửa hàng (`app/shop.py`, thư mục `shops/`)**: mỗi shop có `shops/<id>/shop.json` (+ `kb/`, `products.json`, `orders.json`). NORIVA (`noriva`, mặc định) trỏ `data_dir: "data"` nên dữ liệu cũ giữ nguyên; shop khác dùng DB riêng `omnicare_traces_<id>.db`. Code đọc shop đang bật qua `shop.active()`: system/intent prompt, từ khóa sản phẩm & lỗi (`rule_intent`), từ đồng nghĩa RAG (`_BASE_SYNONYMS` + `synonyms` của shop), bật/tắt tư vấn size, nhãn biến thể ("Size"/"Phiên bản"), ngưỡng freeship. Đổi shop: env `SHOP=<id>`, `POST /api/shops/active`, hoặc Admin → Kênh kết nối → Cửa hàng đang demo. Trang bán hàng dùng chung `/shop` (`static/shop.html`, đọc `GET /api/shop`, đơn thành viên `GET /api/shop/orders?phone=`); NORIVA vẫn dùng `/store`. Shop mẫu thứ hai: `techzone` (điện tử). Hướng dẫn tạo shop: `shops/README.md`, mẫu `shops/_template/`. Test: `tests/test_shops.py`. Bản tĩnh Hugging Face (`deploy_hf.py`) vẫn chỉ có NORIVA.
   - **Captain Auto-FAQ, Conversational Commerce & Channels**: Thêm `GET /api/kb/gaps`, `POST /api/kb/gaps/approve`, trả về `products` kèm thẻ sản phẩm + nút `+ Thêm vào giỏ` trong chat, nén ảnh thật Data URL hiển thị ở cả Inbox và Support Tickets, cùng `POST /api/integrations/telegram/webhook`, `POST /api/integrations/telegram/poll`, `POST /api/integrations/chatwoot`.
   - **Gemini (22/11/2026)**: `.env` dùng `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-4-hydrogen`, `LLM_FALLBACK_MODELS` (danh sách phẩy, `OpenAICompatibleLLM` thử lần lượt khi model chính lỗi 503/429/timeout), `LLM_TIMEOUT=25`, `LLM_PRIMARY_TIMEOUT=8` (chỉ chờ model chính 8 giây khi có dự phòng). `tests/conftest.py` ép `LLM_PROVIDER=mock` để unit test không gọi API thật.
   - **Câu tục / ngoài lề (22/11/2026)**: `guardrails._INSULT_RE` (so chữ còn dấu, tránh nhầm "đồ ngủ") → action `calm` (trả lời xoa dịu, intent `abusive`, status `declined`); chửi kèm vấn đề cụ thể (`_COMPLAINT_HINT_RE`) vẫn `handoff`, khách vãng lai nhận `abusive_guest_msg`. Không có tài liệu và không dính chủ đề mua sắm (`_SHOP_TOPIC_RE`) → `off_topic_msg`. Đồng bộ trong `deploy_hf.py`.
   - **Auto-FAQ dùng model (22/11/2026)**: `app/autofaq.py` — lọc trace fallback/no_grounding (bỏ câu bị chặn, xúc phạm, ngoài lề `_SHOP_TOPIC_RE`), gộp trùng + đếm lượt, model đánh giá `relevant` và soạn Markdown không bịa (chỗ thiếu ghi `[Cần bổ sung: ...]`), cache theo câu, tối đa `MAX_AI_CALLS`=8 câu mới/lần. Câu mẫu `_SEED_KB_GAPS` cũng đi qua model. `POST /api/kb/gaps/dismiss` để bỏ qua. Mock/lỗi model → bản nháp mẫu. Bản tĩnh chưa có.
   - **Đợt sửa tổng (22/11/2026)**:
     - Model: `.env` dùng `gemini-3.5-flash-lite` (2-5 giây; `gemini-3.5-flash` hay hết lượt miễn phí 429, `gemini-4-hydrogen` hay quá tải 503), dự phòng `gemini-flash-lite-latest,gemini-3.5-flash`, `EMBEDDING_MODEL=gemini-embedding-001` (benchmark 10 câu diễn đạt khác: 8/10 so với char 3-gram 6/10, chậm thêm ~0,6 giây/câu).
     - `_classify`: chỉ gọi model khi luật trả `knowledge`; model có thêm nhãn `abusive`, `off_topic`.
     - Streaming: `POST /api/chat/stream` (SSE) gửi `step` (tên span, qua `Trace.on_span` / tham số `progress` của `handle`), rồi `token` và `done` sau khi đã qua guardrail đầu ra. `store.html` dùng `streamStoreChat`, lỗi thì rơi về `/api/chat` (bản tĩnh).
     - `store.html`: ô size theo `sizes` của sản phẩm (`sizeSelect`; FreeSize → "Free size"), không gửi mật khẩu khách lên `/api/customers` (server cũng lọc).
     - Metrics: `negative_sentiment_count` tính cả `abusive_language`. Auto-FAQ lưu kết quả model vào bảng SQLite `autofaq_cache`.
     - File backup chuyển sang `backups/`.
   - **Trang "Cách Agent xử lý" (thay "Agent Teams" hư cấu, 22/11/2026)**: 5 thẻ nhân vật viết cứng (Mr. Zhang, Ren Botong…) đã bỏ. `agent.BRANCHES` là nguồn sự thật (7 nhánh: ý định → công cụ → tài liệu, chỉ thành viên / hỏi xác nhận); `GET /api/agent/overview` (bảo vệ token) trả model, công cụ + số lượt dùng, tài liệu, số tin mỗi nhánh; `renderAgentCards()` vẽ từ API. Bản tĩnh: `branches` nằm trong seed, engine JS trả cùng định dạng. Sửa luồng trong `handle()` thì cập nhật `BRANCHES`.
   - **Tính năng AI cho nhân viên & quản lý (`app/ai_features.py`, 22/11/2026)** — mọi hàm có đường lui bằng luật khi Mock/lỗi model:
     - **Tóm tắt bàn giao**: khi `status == "handoff"`, `main._schedule_handoff_summary` chạy luồng nền → `sess["handoff_summary"]` (khách, vấn đề, AI đã làm, cảm xúc, mức gấp, việc nên làm), lưu SQLite, phát WS `summary_ready` (không kèm nội dung). `handoff_summary` là dữ liệu nội bộ: `_public_conv` lọc khỏi WS và `GET /api/conversations/{id}`. Admin: ô "Bàn giao từ AI" ở cột phải + nút tóm tắt tay (`POST /api/copilot/summary`).
     - **Copilot**: `copilot_suggest` gọi model soạn 3 câu theo hội thoại + tài liệu + đơn khách nhắc tới; `POST /api/copilot/rewrite` (polite/short/apology/friendly) cho các nút "Viết lại" dưới ô soạn tin.
     - **Stylist AI**: ý định `outfit_advice` (`OUTFIT_RE`, chỉ shop có `size_advice`), `pick_outfit` chỉ nhận id có trong danh mục và còn hàng, bổ sung nếu model chọn < 2 món, cộng tổng tiền; thẻ sản phẩm giữ đủ bộ (không lọc theo câu trả lời). `store.html`: nút "Thêm cả bộ vào giỏ" (`addOutfitToCart`). Bản tĩnh: `ruleOutfit` trong `edge_extensions.js`.
     - **Báo cáo tuần**: `POST /api/reports/weekly` (`collect_week` + `weekly_report`, nhãn tiếng Việt `LABEL_VI`), lưu bảng `ai_reports`; `GET` lấy bản mới nhất. Admin: thẻ đầu trang Chỉ số vận hành.
     - Các API trên đều nằm trong `auth.PROTECTED`.
   - **Sửa theo feedback web bán hàng (23/11/2026, `D:\Downloads\CDTT.pdf`)**: nút "Hỏi AI" trên thẻ sản phẩm gọi `askAboutProduct(id)` (bong bóng "Bạn có câu hỏi gì về …", gửi `page.product_id` một lần). `POST /api/chat` nhận `page` = `{cart:[{id,size,qty}], product_id, gender:"nam"|"nu"}` → `agent._apply_page_context` (giá lấy từ danh mục). Ý định mới `cart_inquiry` (`CART_RE`, `_cart_answer`: tạm tính + phí ship `shipping_fee()` + tổng, bỏ qua guardrail đầu ra). Size theo giới tính: `products.json` có `gender` (nam/nu/unisex), `SIZE_CHART_F` cho đồ nữ, giới tính lấy từ câu khách (`FEMALE_RE`/`MALE_RE`) → khu Nam/Nữ đang xem (`VIEW_GENDER`) → giới tính sản phẩm. Kính/phụ kiện (`accessory`, FreeSize, `ToolRegistry.is_free_size`) không tư vấn size, ẩn size trên thẻ và giỏ. Đặt hàng: server tự tính `subtotal`, `shipping_fee`, `total`, `created_at`, `eta` = ngày đặt + 3. "đơn này giá bao nhiêu" đi nhánh `order_status`; nhánh đơn hàng không chạy `_price_mismatch`. Chưa đồng bộ vào engine tĩnh (`edge_extensions.js`). Bản trước khi sửa: `backups/store.before-feedback.html`.
   - **Tận dụng model GPT qua codex.fosta.app (23/11/2026)** — `.env`: `LLM_PROVIDER=openai`, `LLM_BASE_URL=https://codex.fosta.app/v1`, `LLM_MODEL=gpt-6-luna`, dự phòng `gpt-6.2-astra`, không có embedding (Gemini cũ ở `backups/.env.gemini.bak`):
     - **Agent tự gọi công cụ**: `OpenAICompatibleLLM.run_tools` (function calling, tối đa `AGENT_MAX_STEPS`=3 vòng) + `OmniCareAgent._agentic_generate` dùng ở nhánh `product_inquiry`, `order_status`, `knowledge` (kể cả khi chưa tìm thấy tài liệu). Model chỉ được công cụ CHỈ ĐỌC (`AGENT_TOOL_NAMES` + `get_order_status` cho thành viên, chặn đơn không thuộc khách); hủy/đổi/hoàn/phiếu vẫn đi nhánh cố định có xác nhận. Model không hỗ trợ (Mock) → `_generate`. Bước đã gọi lưu `trace.attributes["agent_steps"]`, span `agent:tool_loop`. `plain_text()` bỏ Markdown.
     - **Đọc ảnh thật**: `OpenAICompatibleLLM.see` + `OmniCareAgent._vision` (`VISION_PROMPT`, JSON kind/finding/severity/category/order_id). `not_product` → hỏi chụp lại (status clarify, không tạo phiếu). Ảnh SVG demo / lỗi → luật cũ `analyze_multimodal_image`.
     - **Giới hạn lượt**: `MODEL_CALLS_PER_DAY` (mặc định 40) theo phiên khách, `llm.QUOTA_KEY` (contextvar, `handle()` đặt khi có `customer`), `quota_take()` trong `_post`; hết lượt → `QuotaExceeded` → rơi về Mock, cờ `quota_exceeded`. Admin/nội bộ không giới hạn.
     - **Cột Model AI** trong Observability → Traces (`traceModelCell`, `trace.attributes["llm_used"]` ghi bởi `note_llm`). HTML trả `Cache-Control: no-cache`.
     - **Evals giám khảo là model thật**: `python -m evals.run_evals --judge llm [--limit N]` (`LLMJudge`, `JUDGE_MODEL`, đạt khi trung bình ≥ 3.5, báo `llm_judge_by_criterion`). Mặc định vẫn `--judge rule`. Chạy thử 15 case: 4.82/5.
     - Test: `tests/test_agentic.py` (59 test tổng). Engine tĩnh chưa có các tính năng này.
     - **Đợt 2 (23/11/2026)**: (a) Streaming hiện bước agent: `STEP_LABEL`/`STEP_DONE` trong store.html, dòng "Đã tra đơn → xem tồn kho" (`.oc-trail`). (b) **Tìm sản phẩm bằng ảnh** (intent `image_search`, khách vãng lai dùng được): `VISION_PROMPT` có kind `fashion_item` + `matches` (mã trong danh mục cùng loại), `_image_search_answer`; ảnh có chữ báo lỗi (`DEFECT_RE`) vẫn đi nhánh phiếu (chỉ thành viên); store: chip "📷 Tìm theo ảnh" (`pickImage('search')`), chữ kèm ảnh lấy từ ô nhập (`imageCaption`). (c) **Chấm chất lượng hội thoại**: `ai_features.score_conversation` (điểm, resolved, accuracy, tone, issues, suggestion; luật dự phòng), `POST /api/conversations/{sid}/qa` (bảo vệ token, đưa tài liệu tra theo cả câu khách lẫn câu AI + danh mục), lưu `sess["qa"]`; Admin tự chấm khi bấm Đóng, ô "Chấm chất lượng (AI)". (d) **Trí nhớ thành viên**: bảng SQLite `customer_memory` (`save_memory`/`load_memory`), `OmniCareAgent._memory/_remember/memory_brief`: số đo + size, giới tính (chỉ khi khách tự nói), 5 món quan tâm, số lần khiếu nại; đưa vào ngữ cảnh model (`customer_memory` trong `render_context`), chào lại theo tên, tư vấn size dùng số đo đã nhớ; Admin ô "AI ghi nhớ về khách". `qa`, `memory` thuộc `_INTERNAL_KEYS`.
     - **Deploy backend thật**: `scripts/deploy_hf_docker.py` (Docker Space `omnicare-live`, key thành Secret, cần `HF_TOKEN` trong .env). Tạm thời đang chia sẻ qua đường hầm Cloudflare (`cloudflared` trong scratchpad, link đổi mỗi lần chạy).
   - **Bỏ thẻ Chatwoot khỏi trang Kênh kết nối (22/11/2026)**: giao diện chỉ còn Telegram; endpoint `/api/integrations/chatwoot` và cấu hình env `CHATWOOT_URL`, `CHATWOOT_API_TOKEN` vẫn giữ ở backend (xem `docs/chatwoot_setup.md`).
