"""Deploy OmniCare to Hugging Face Static Spaces (free 24/7 permanent URL).

Generates a self-contained static build (`index.html` + `README.md`) that:
- Uses the Python FastAPI backend when available (`/health` returns JSON)
- Automatically falls back to the embedded in-browser OmniCare Engine (with all 6 MVP
  components: Model, Workflow Agent DAG, BM25 RAG + 4 MCP Tools, Input/Output Guardrails,
  Observability TraceStore + Metrics, and 24-case Evals runner) when hosted on a static CDN.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Build bằng cơ sở dữ liệu tạm trống: bản công khai chỉ chứa trace từ bộ eval, không mang
# lịch sử chat thử / đơn hàng / khách hàng trong omnicare_traces.db ở máy lên mạng.
import tempfile  # noqa: E402

if "TRACE_DB" not in os.environ:
    _tmp_db = Path(tempfile.gettempdir()) / "omnicare_hf_build.db"
    _tmp_db.unlink(missing_ok=True)
    os.environ["TRACE_DB"] = str(_tmp_db)

from app.agent import OmniCareAgent  # noqa: E402
from evals.run_evals import evaluate_case, summarize  # noqa: E402

DATASET_PATH = ROOT / "evals" / "dataset.jsonl"


def run_all() -> dict:
    cases = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ag = OmniCareAgent()
    results = [evaluate_case(ag, c) for c in cases]
    return {"model": ag.llm.name, "summary": summarize(results), "results": results}


def build_embedded_bundle() -> str:
    agent = OmniCareAgent()
    kb_docs: dict[str, dict] = {}
    for c in agent.tools.retriever.chunks:
        doc = kb_docs.setdefault(
            c.source,
            {"source": c.source, "title": c.title.split(" - ")[0], "chunks": []},
        )
        doc["chunks"].append({"id": c.id, "title": c.title, "text": c.text, "source": c.source})

    eval_cases = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_report = run_all()

    seed_data = {
        "kb_documents": list(kb_docs.values()),
        "chunks": [
            {"id": c.id, "source": c.source, "title": c.title, "text": c.text}
            for c in agent.tools.retriever.chunks
        ],
        "orders": [
            {"order_id": k, **{ik: iv for ik, iv in v.items() if ik not in {"phone", "address", "email"}}}
            for k, v in agent.tools.orders.items()
        ],
        "products": agent.tools.products,
        "tools": agent.tools.list_specs(),
        "branches": __import__("app.agent", fromlist=["BRANCHES"]).BRANCHES,
        "eval_cases": eval_cases,
        "initial_eval_report": eval_report,
        "initial_traces": agent.store.list(25),
    }

    seed_json = json.dumps(seed_data, ensure_ascii=False)

    # Dùng chung từ điển với bản Python (rag.py + hồ sơ shop) để engine trình duyệt không lệch
    from app import rag as _rag
    from app.shop import active as _active_shop

    _shop = _active_shop()
    stopwords_json = json.dumps(sorted(_rag._STOPWORDS))
    synonyms_json = json.dumps(_rag._synonyms(), ensure_ascii=False)
    product_re_json = json.dumps(_shop.product_pattern)
    defect_re_json = json.dumps(_shop.defect_pattern)
    min_coverage = _rag.MIN_QUERY_COVERAGE

    js_engine = f"""
// =========================================================================
// OMNICARE BROWSER ENGINE FALLBACK (for 24/7 Static Cloud Hosting)
// =========================================================================
const OMNICARE_SEED = {seed_json};
let _EDGE_MODE = false;

const _STOPWORDS = new Set({stopwords_json});
const _PRODUCT_RE = new RegExp({product_re_json});
const _DEFECT_ALT = {defect_re_json};
const _MIN_QUERY_COVERAGE = {min_coverage};

function _norm(text) {{
  return String(text || "").toLowerCase().replace(/đ/g, "d").normalize("NFD").replace(/[\\u0300-\\u036f]/g, "");
}}

const _SYNONYMS = {synonyms_json};

function _tokenize(text, expand = false) {{
  const tokens = _norm(text).match(/[a-z0-9]+/g) || [];
  const words = tokens.filter(t => !_STOPWORDS.has(t));
  const bigrams = [];
  for (let i = 0; i < words.length - 1; i++) bigrams.push(words[i] + "_" + words[i + 1]);
  const out = words.concat(bigrams);
  if (expand) {{
    for (const w of words.concat(bigrams)) if (_SYNONYMS[w]) out.push(..._SYNONYMS[w]);
  }}
  return out;
}}

// Build Hybrid BM25 index in browser (re-indexable when user uploads KB)
const _BM25 = (() => {{
  let tfList = [], df = {{}}, avgdl = 1;
  function rebuild() {{
    tfList = []; df = {{}}; let totalLen = 0;
    for (const c of OMNICARE_SEED.chunks) {{
      const toks = _tokenize(c.title + " " + c.text);
      const tf = {{}};
      for (const t of toks) tf[t] = (tf[t] || 0) + 1;
      tfList.push({{ tf, dl: toks.length }});
      totalLen += toks.length;
      for (const k of Object.keys(tf)) df[k] = (df[k] || 0) + 1;
    }}
    avgdl = totalLen / Math.max(1, OMNICARE_SEED.chunks.length);
  }}
  rebuild();
  return {{
    rebuild,
    search(query, topK = 3, minScore = 2.0) {{
      const q = _tokenize(query, true);
      // Từ đơn của câu hỏi (không bigram), để lọc khớp nhầm do bỏ dấu ("máy" ~ "may") như bản Python
      const qWords = Array.from(new Set(_tokenize(query).filter(t => !t.includes("_"))));
      const chunks = OMNICARE_SEED.chunks;
      const n = chunks.length;
      const k1 = 1.5, b = 0.75;
      const scored = [];
      for (let i = 0; i < n; i++) {{
        const {{ tf, dl }} = tfList[i];
        let score = 0;
        for (const term of q) {{
          if (!tf[term]) continue;
          const dfv = df[term] || 0;
          const idf = Math.log(1 + (n - dfv + 0.5) / (dfv + 0.5));
          const f = tf[term];
          score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl));
        }}
        if (qWords.length) {{
          const covered = qWords.filter(w => tf[w] || (_SYNONYMS[w] || []).some(s => tf[s])).length;
          if (covered / qWords.length < _MIN_QUERY_COVERAGE) continue;
        }}
        if (score >= minScore) scored.push([score, i]);
      }}
      scored.sort((a, b) => b[0] - a[0]);
      return scored.slice(0, topK).map(([s, i]) => {{
        const vecSim = Number(Math.min(0.42, s * 0.045).toFixed(3));
        const rerank = Number(Math.min(0.8, s * 0.06).toFixed(3));
        return {{
          id: chunks[i].id,
          source: chunks[i].source,
          title: chunks[i].title,
          text: chunks[i].text,
          bm25_score: Number(s.toFixed(3)),
          vector_score: vecSim,
          rerank_score: rerank,
          score: Number((s + vecSim + rerank).toFixed(3))
        }};
      }});
    }}
  }};
}})();

const _EDGE_STATE = {{
  orders: Object.assign(
    Object.fromEntries(OMNICARE_SEED.orders.map(o => [o.order_id.toUpperCase(), o])),
    JSON.parse(localStorage.getItem("omnicare_custom_orders") || "{{}}")
  ),
  tickets: [],
  sessions: {{}},
  traces: OMNICARE_SEED.initial_traces || [],
  eval_report: OMNICARE_SEED.initial_eval_report
}};

function _luhnOk(digits) {{
  let total = 0, alt = false;
  for (let i = digits.length - 1; i >= 0; i--) {{
    let n = parseInt(digits[i], 10);
    if (alt) {{ n *= 2; if (n > 9) n -= 9; }}
    total += n;
    alt = !alt;
  }}
  return total % 10 === 0;
}}

function _redactPii(text) {{
  const found = new Set();
  let out = String(text).replace(/\\b(?:\\d[ -]?){{13,19}}\\b/g, m => {{
    const d = m.replace(/\\D/g, "");
    if (d.length >= 13 && d.length <= 19 && _luhnOk(d)) {{ found.add("pii_card"); return "[REDACTED_CARD]"; }}
    return m;
  }});
  out = out.replace(/[\\w.+-]+@[\\w-]+\\.[\\w.-]+/g, () => {{ found.add("pii_email"); return "[REDACTED_EMAIL]"; }});
  out = out.replace(/\\b\\d{{12}}\\b/g, () => {{ found.add("pii_national_id"); return "[REDACTED_NATIONAL_ID]"; }});
  out = out.replace(/(?<!\\d)(?:\\+?84|0)(?:[ .-]?\\d){{9}}(?!\\d)/g, () => {{ found.add("pii_phone"); return "[REDACTED_PHONE]"; }});
  return {{ text: out, flags: Array.from(found).sort() }};
}}

function _checkInput(text) {{
  if (!text || !text.trim()) return {{ allowed: false, text, flags: ["empty_input"], reason: "Tin nhắn trống.", action: "block" }};
  if (text.length > 2000) return {{ allowed: false, text: text.slice(0, 2000), flags: ["input_too_long"], reason: "Tin nhắn vượt quá 2000 ký tự.", action: "block" }};
  const norm = _norm(text);
  const inj = [
    /ignore (all|any|the)? ?(previous|above|prior) (instructions|prompts?)/,
    /disregard (the|all|your) (rules|instructions)/,
    /(reveal|show|print|repeat) (your|the) (system )?(prompt|instructions)/,
    /you are now (dan|developer mode)/,
    /\\bjailbreak\\b/,
    /bo qua (tat ca |moi )?(cac )?(huong dan|chi dan|quy tac|lenh)/,
    /(tiet lo|in ra|cho (toi|minh) xem) (system prompt|prompt he thong|huong dan he thong)/,
    /gia vo (ban|minh) la/,
    /dong vai (admin|quan tri|hacker)/
  ];
  if (inj.some(r => r.test(norm))) return {{ allowed: false, text, flags: ["prompt_injection"], reason: "Yêu cầu chứa nội dung cố thay đổi chỉ dẫn hệ thống.", action: "block" }};
  const off = [
    /\\b(viet|code|lap trinh) (code|chuong trinh|ham|script)\\b/,
    /\\b(chinh tri|bau cu|ton giao)\\b/,
    /\\b(hack|bom|vu khi|ma tuy)\\b/,
    /\\b(chung khoan|tien ao|crypto) nen mua\\b/,
    /\\b(thoi tiet|du bao thoi tiet|ke chuyen cuoi|chuyen cuoi|lam tho|xo so|so de|ket qua bong da|tong thong|thu tuong)\\b/,
    /\\b(giai|lam) (bai tap|toan|phuong trinh)\\b/,
    /^\\s*\\d+\\s*[-+*\\/x:]\\s*\\d+\\s*(=|bang)?\\s*(may|bao nhieu)?\\s*\\??\\s*$/
  ];
  if (off.some(r => r.test(norm))) return {{ allowed: false, text, flags: ["off_topic"], reason: "Câu hỏi nằm ngoài phạm vi hỗ trợ.", action: "decline" }};
  const {{ text: redacted, flags }} = _redactPii(text);
  const words = new Set(norm.match(/[a-z]+/g) || []);
  const abuse = ["dm", "dmm", "vcl", "vl", "ngu vai", "lua dao", "cut"];
  // Xúc phạm: so trên chữ CÒN DẤU để không nhầm "đồ ngủ", "đeo".
  const insult = /(?<![\\p{{L}}\\d])(ngu|đồ ngu|ngu ngốc|ngu như bò|óc chó|oc cho|óc lợn|đồ chó|thằng chó|con chó|chó chết|điên|khùng|khốn nạn|mất dạy|láo|đéo|địt|dit me|đm|đmm|dm|dmm|dcm|đcm|dkm|đkm|cc|clm|cl|vcl|vkl|vl|cút|mẹ mày|bố mày|má mày|thg ngu|thằng ngu|con ngu|rác rưởi|đồ rác)(?![\\p{{L}}\\d])/iu;
  if (insult.test(text) || abuse.some(w => w.includes(" ") ? norm.includes(w) : words.has(w))) {{
    flags.push("abusive_language");
    if (/\\b(lua dao|giao (cham|sai|thieu|lau)|cham|don|oc\\d+|loi|hong|rach|hoan tien|bom hang|chua nhan|khong nhan|ship)\\b/.test(norm))
      return {{ allowed: true, text: redacted, flags, reason: "Khách hàng đang bức xúc.", action: "handoff" }};
    return {{ allowed: true, text: redacted, flags, reason: "Khách dùng lời lẽ xúc phạm.", action: "calm" }};
  }}
  const negPats = [/rat that vong/, /qua te/, /cho mai khong thay/, /lam an (chan|qua te|kieu gi)/, /buc minh/, /khong hai long/];
  if (negPats.some(r => r.test(norm))) flags.push("negative_sentiment");
  return {{ allowed: true, text: redacted, flags, reason: null, action: "allow" }};
}}

function _ruleIntent(text) {{
  const t = _norm(text);
  if (/\\b(nhan vien|nguoi that|tong dai vien|gap (nguoi|quan ly)|noi chuyen voi nguoi)\\b/.test(t)) return "handoff";
  if (/\\b(huy don|muon huy|huy giup)\\b/.test(t)) return "cancel_order";
  if (/\\b(doi dia chi|thay doi dia chi|giao sang dia chi)\\b/.test(t)) return "update_address";
  if (/\\b(yeu cau hoan tien|hoan tien don)\\b/.test(t) && !/\\b(bao lau|chinh sach|quy dinh)\\b/.test(t)) return "request_refund";
  const _complaintRe = new RegExp("\\\\b(loi|hong|vo|rach|" + (_DEFECT_ALT ? _DEFECT_ALT + "|" : "") + "khong (hoat dong|len nguon|sac)|giao sai|thieu hang|khieu nai|doi hoan tien|hoan tien cho)\\\\b");
  if (_complaintRe.test(t)
      && /\\b(cua (toi|minh|em)|toi|minh|em|don)\\b/.test(t)
      && !/\\b(chinh sach|quy dinh|nhu the nao|bao lau)\\b/.test(t)) return "complaint";
  if (_PRODUCT_RE.test(t)
      && /\\b(gia|bao nhieu|con hang|con size|ton kho|mau|kiem tra kho|het hang)\\b|\\b(con|co ban|ban)\\b.*\\bkhong\\b/.test(t)) return "product_inquiry";
  if (/\\bOC\\s?-?(\\d{{5}})\\b/i.test(text) || /\\b(don (hang )?(cua (toi|minh|em)|toi|minh|em)|kiem tra don|tra cuu don|don .*(dau|chua|khi nao))\\b/.test(t)) return "order_status";
  if (/^\\s*(xin chao|chao|hello|hi|alo|chao ban|chao shop)[\\s!.]*$/.test(t)) return "greeting";
  return "knowledge";
}}

function _callToolEdge(name, args) {{
  if (name === "search_knowledge_base") {{
    return {{ results: _BM25.search(args.query || "", args.top_k || 3) }};
  }}
  if (name === "get_order_status") {{
    const oid = String(args.order_id || "").trim().toUpperCase();
    const o = _EDGE_STATE.orders[oid];
    return o ? {{ found: true, ...o }} : {{ found: false, order_id: oid }};
  }}
  if (name === "search_products") {{
    const qToks = new Set(_tokenize(args.query || ""));
    const scored = [];
    for (const p of (OMNICARE_SEED.products || [])) {{
      const pToks = new Set(_tokenize(p.name + " " + p.category + " " + (p.colors || []).join(" ")));
      let ov = 0; for (const tk of qToks) if (pToks.has(tk)) ov++;
      if (ov > 0 || _norm(p.name).includes(_norm(args.query || ""))) scored.push([ov, p]);
    }}
    scored.sort((a, b) => b[0] - a[0]);
    const matches = scored.slice(0, 3).map(([, p]) => {{
      const item = {{ ...p }};
      if (args.size) {{
        const su = String(args.size).toUpperCase();
        item.requested_size = su;
        item.requested_size_stock = (p.sizes || {{}})[su] || 0;
      }}
      return item;
    }});
    return {{ found: matches.length > 0, count: matches.length, products: matches }};
  }}
  if (name === "cancel_order") {{
    const oid = String(args.order_id || "").trim().toUpperCase();
    const o = _EDGE_STATE.orders[oid];
    if (!o) return {{ cancelled: false, order_id: oid, message: `Không tìm thấy đơn hàng ${{oid}}.` }};
    if (o.status === "Đã giao thành công" || o.status === "Đã hủy") {{
      return {{ cancelled: false, order_id: oid, message: `Đơn hàng ${{oid}} đang ở trạng thái '${{o.status}}' nên không thể hủy.` }};
    }}
    o.status = "Đã hủy";
    return {{ cancelled: true, order_id: oid, product: o.product, reason: args.reason || "Khách yêu cầu hủy", status: "Đã hủy" }};
  }}
  if (name === "check_inventory") {{
    const sr = _callToolEdge("search_products", {{ query: args.product || "", size: args.size }});
    if (!sr.products || !sr.products.length) return {{ found: false, product: args.product, in_stock: false }};
    const p = sr.products[0];
    const su = args.size ? String(args.size).toUpperCase() : null;
    const st = su ? ((p.sizes || {{}})[su] || 0) : Object.values(p.sizes || {{}}).reduce((a, b) => a + b, 0);
    return {{ found: true, product_id: p.id, name: p.name, price: p.price, sizes: p.sizes, colors: p.colors, requested_size: su, stock: st, in_stock: st > 0 }};
  }}
  if (name === "update_shipping_address") {{
    const oid = String(args.order_id || "").trim().toUpperCase();
    const o = _EDGE_STATE.orders[oid];
    if (!o) return {{ updated: false, order_id: oid, message: `Không tìm thấy đơn ${{oid}}.` }};
    o.address = args.new_address;
    return {{ updated: true, order_id: oid, product: o.product, new_address: args.new_address }};
  }}
  if (name === "request_refund") {{
    const oid = String(args.order_id || "").trim().toUpperCase();
    const o = _EDGE_STATE.orders[oid];
    const tk = _callToolEdge("create_support_ticket", {{ summary: `Yêu cầu hoàn tiền đơn ${{oid}}: ${{args.reason || ""}}`, category: "refund", order_id: oid }});
    return {{ refund_requested: true, order_id: oid, amount: o ? o.total : 500000, ticket_id: tk.ticket_id, sla_hours: tk.sla_hours }};
  }}
  if (name === "create_support_ticket") {{
    const tid = `TK-${{String(_EDGE_STATE.tickets.length + 1).padStart(4, "0")}}`;
    const slaH = (args.category === "warranty" || args.category === "refund") ? 4 : 24;
    const defaultAssignee = {{ warranty: "Linh (Bảo hành)", refund: "Minh (Tài chính)", return: "Hùng (Kho vận)", complaint: "Lan (CSKH Tier-2)" }}[args.category] || "Chưa phân công";
    const ticket = {{
      ticket_id: tid,
      summary: args.summary,
      category: args.category,
      order_id: args.order_id || null,
      created_at: Date.now() / 1000,
      sla_hours: slaH,
      assignee: defaultAssignee,
      status: "open",
      image_url: args.image_url || null
    }};
    _EDGE_STATE.tickets.push(ticket);
    return ticket;
  }}
  if (name === "handoff_to_human") {{
    return {{ handoff: true, queue: "tier2_cs", reason: args.reason, eta_minutes: 3 }};
  }}
  return {{ error: "Unknown tool " + name }};
}}

function _mockGenerate(user, context) {{
  for (const tr of (context.tool_results || [])) {{
    const r = tr.result;
    if (tr.tool === "get_order_status") {{
      if (!r.found) return `Mình chưa tìm thấy đơn hàng ${{r.order_id}}. Bạn vui lòng kiểm tra lại mã đơn (bắt đầu bằng OC) trong email xác nhận nhé.`;
      let msg = `Đơn hàng ${{r.order_id}} (${{r.product || ""}}) hiện đang ở trạng thái: ${{r.status}}.`;
      if (r.carrier) msg += ` Đơn vị vận chuyển: ${{r.carrier}}, mã vận đơn ${{r.tracking || ""}}.`;
      if (r.eta) msg += ` Dự kiến giao: ${{r.eta}}.`;
      return msg;
    }}
    if (tr.tool === "search_products") {{
      const prods = r.products || [];
      if (!prods.length) return "Xin lỗi, mình chưa tìm thấy sản phẩm phù hợp trong danh mục NORIVA.";
      const p = prods[0];
      let msg = `${{p.name}} hiện có giá ${{Number(p.price).toLocaleString("vi-VN")}}đ.`;
      if (p.requested_size) {{
        msg += p.requested_size_stock > 0 ? ` Size ${{p.requested_size}} hiện còn ${{p.requested_size_stock}} sản phẩm.` : ` Size ${{p.requested_size}} hiện tạm hết hàng.`;
      }} else {{
        const okSizes = Object.entries(p.sizes || {{}}).filter(([, v]) => v > 0).map(([k]) => k);
        if (okSizes.length) msg += ` Các size còn hàng: ${{okSizes.join(", ")}}.`;
      }}
      return msg;
    }}
    if (tr.tool === "cancel_order") {{
      if (r.cancelled) return `Đơn hàng ${{r.order_id}} (${{r.product || ""}}) đã được hủy thành công theo xác nhận của bạn.`;
      return r.message || `Không thể hủy đơn hàng ${{r.order_id}}.`;
    }}
    if (tr.tool === "update_shipping_address") {{
      return r.updated ? `Đã cập nhật địa chỉ giao hàng mới cho đơn ${{r.order_id}} thành: "${{r.new_address}}".` : (r.message || "Không thể đổi địa chỉ.");
    }}
    if (tr.tool === "request_refund") {{
      return `Đã tạo yêu cầu hoàn tiền (mã phiếu ${{r.ticket_id}}) cho đơn ${{r.order_id}}. Bộ phận tài chính sẽ xử lý trong vòng ${{r.sla_hours}} giờ.`;
    }}
    if (tr.tool === "create_support_ticket") {{
      return `Mình đã tạo phiếu hỗ trợ ${{r.ticket_id}} cho bạn. Bộ phận chuyên trách sẽ liên hệ trong vòng ${{r.sla_hours}} giờ.`;
    }}
    if (tr.tool === "handoff_to_human") {{
      return `Mình đã chuyển cuộc trò chuyện tới nhân viên chăm sóc khách hàng. Thời gian chờ dự kiến khoảng ${{r.eta_minutes}} phút, bạn vui lòng giữ kết nối nhé.`;
    }}
  }}
  const docs = context.documents || [];
  if (!docs.length) return "Xin lỗi, mình chưa có thông tin chính xác cho câu hỏi này. Bạn có muốn mình chuyển tới nhân viên hỗ trợ không?";
  const q = new Set(_tokenize(user));
  const sents = [];
  docs.forEach((doc, dRank) => {{
    doc.text.split(/(?<=[.!?])\\s+|\\n+/).forEach(raw => {{
      const s = raw.replace(/^[\\s\\-*]+|[\\s\\-*]+$/g, "");
      if (s.length < 15) return;
      let overlap = 0;
      for (const tk of new Set(_tokenize(s))) if (q.has(tk)) overlap++;
      sents.push([overlap - dRank * 0.5, s]);
    }});
  }});
  sents.sort((a, b) => b[0] - a[0]);
  const pos = sents.slice(0, 2).filter(x => x[0] > 0).map(x => x[1]);
  const best = pos.length ? pos : (sents.length ? [sents[0][1]] : []);
  return best.join(" ");
}}

const _MEMBER_ONLY = new Set(["order_status", "cancel_order", "update_address", "request_refund", "complaint", "handoff", "vision_defect"]);
const _ORDER_INTENTS = new Set(["order_status", "cancel_order", "update_address", "request_refund", "complaint"]);
const _MEMBER_ONLY_MSG = "Tra cứu đơn hàng, báo lỗi sản phẩm và chat trực tiếp với nhân viên là quyền lợi dành cho khách đã có tài khoản NORIVA. Bạn vui lòng đăng nhập (tài khoản được tạo khi đặt hàng) để được hỗ trợ nhé. Trong lúc chờ, mình vẫn có thể tư vấn sản phẩm, size và chính sách cho bạn.";

const _OFF_TOPIC_MSG = "Câu này nằm ngoài phạm vi mình hỗ trợ rồi ạ. Mình là trợ lý của NORIVA, có thể giúp bạn tìm sản phẩm, tư vấn size, phí ship, đổi trả hay kiểm tra đơn hàng. Bạn đang cần gì nè?";
const _SHOP_TOPIC_RE = /\\b(shop|cua hang|don|hang|san pham|mua|gia|size|mau|ship|giao|van chuyen|doi|tra|hoan|bao hanh|thanh toan|voucher|ma giam|khuyen mai|sale|tai khoan|thanh vien|diem|ao|quan|vay|giay|dep|tui|mu|kinh|vai|chat lieu|nhan vien|ho tro)\\b/;

function _accessEdge(sess, intent, text, addSpan) {{
  const cust = sess.customer;
  if (!cust) return null;
  const ctype = cust.type || "guest";
  if (ctype !== "member") {{
    if (_MEMBER_ONLY.has(intent)) {{ addSpan("access:check", "ok", {{ customer_type: ctype, intent, decision: "login_required" }}, 0.1); return [_MEMBER_ONLY_MSG, "login_required", "member_only"]; }}
    addSpan("access:check", "ok", {{ customer_type: ctype, intent, decision: "allow" }}, 0.1);
    return null;
  }}
  const owned = new Set((cust.order_ids || []).map(o => String(o).toUpperCase()));
  if (_ORDER_INTENTS.has(intent)) {{
    const m = text.match(/\\bOC\\s?-?(\\d{{5}})\\b/i);
    const oid = m ? `OC${{m[1]}}` : null;
    if (oid && !owned.has(oid)) {{
      addSpan("access:check", "ok", {{ customer_type: ctype, intent, decision: "order_not_owned", order_id: oid }}, 0.1);
      return [`Đơn hàng ${{oid}} không thuộc tài khoản của bạn nên mình không thể truy cập. Các đơn của bạn: ${{[...owned].sort().join(", ") || "chưa có"}}.`, "denied", "order_not_owned"];
    }}
    if (!oid && !sess.last_order_id && owned.size) sess.last_order_id = [...owned].sort().pop();
  }}
  addSpan("access:check", "ok", {{ customer_type: ctype, intent, decision: "allow" }}, 0.1);
  return null;
}}

function _handleChatEdge(sessionId, message, imageInfo = null, customer = null) {{
  const t0 = performance.now();
  const spans = [];
  const addSpan = (name, status, attrs, dur = 0.6) => spans.push({{ name, duration_ms: dur, status, attributes: attrs }});
  const sess = (_EDGE_STATE.sessions[sessionId] = _EDGE_STATE.sessions[sessionId] || {{ history: [], last_order_id: null, pending: null }});
  if (customer) sess.customer = customer;
  const flags = [];
  const calls = [];
  let citations = [];
  let matchedProducts = [];
  let intent = "unknown", status = "ok", answer = "";
  const effMsg = (message && message.trim()) ? message : (imageInfo ? `[Gửi ảnh: ${{imageInfo.name || "image.jpg"}}]` : "");
  let isTakeover = Boolean(sess.takeover);
  try {{
    const allLs = JSON.parse(localStorage.getItem("omnicare_store_sessions") || "{{}}");
    if (allLs[sessionId] && typeof allLs[sessionId].takeover === "boolean") isTakeover = allLs[sessionId].takeover;
  }} catch (e) {{}}
  if (isTakeover) {{
    return {{
      answer: "👩‍💼 Nhân viên CSKH đang trực tiếp phụ trách phiên chat này và đã nhận được tin nhắn của bạn.",
      trace_id: Math.random().toString(16).slice(2, 14),
      intent: "handoff",
      status: "human_active",
      citations: [],
      tool_calls: [],
      guardrail_flags: [],
      products: [],
      latency_ms: 0.9
    }};
  }}

  const gin = _checkInput(effMsg);
  flags.push(...gin.flags);
  addSpan("guardrail:input", "ok", {{ action: gin.action, flags: gin.flags }}, 0.4);

  if (gin.action === "block") {{
    answer = `Xin lỗi, mình không thể xử lý yêu cầu này. ${{gin.reason}}`;
    status = "blocked"; intent = "blocked";
  }} else if (gin.action === "decline") {{
    answer = _OFF_TOPIC_MSG;
    status = "declined"; intent = "off_topic";
  }} else if (gin.action === "calm") {{
    answer = "Mình rất tiếc nếu NORIVA đã làm bạn chưa vui. Bạn có thể chia sẻ điều gì khiến bạn chưa hài lòng để mình hỗ trợ không ạ? Mình luôn sẵn sàng giúp bạn tìm sản phẩm, kiểm tra đơn hay giải đáp chính sách.";
    status = "declined"; intent = "abusive";
  }} else {{
    const text = gin.text;
    const orderMatch = text.match(/\\bOC\\s?-?(\\d{{5}})\\b/i) || (imageInfo && String(imageInfo.name || "").match(/\\bOC\\s?-?(\\d{{5}})\\b/i));
    const context = {{ documents: [], tool_results: [], history: sess.history.slice(-4) }};
    const callTool = (name, args) => {{
      const res = _callToolEdge(name, args);
      calls.push({{ tool: name, args, result: res }});
      addSpan("tool:" + name, "ok", {{ args, result_preview: JSON.stringify(res).slice(0, 200) }}, 0.8);
      return res;
    }};

    const imgDenied = imageInfo ? _accessEdge(sess, "vision_defect", text, addSpan) : null;
    if (imgDenied) {{
      [answer, status] = imgDenied; flags.push(imgDenied[2]); intent = "vision_defect";
    }} else if (imageInfo) {{
      const oid = orderMatch ? `OC${{orderMatch[1]}}` : (sess.last_order_id || "OC10005");
      sess.last_order_id = oid;
      const finding = `Vision AI: Phát hiện vết rách đường chỉ / lỗi bề mặt vải trên sản phẩm thuộc đơn ${{oid}}.`;
      addSpan("vision:analyze_image", "ok", {{ image_name: imageInfo.name, detected_order_id: oid, finding }}, 1.1);
      flags.push("multimodal_vision");
      intent = "vision_defect";
      const tk = callTool("create_support_ticket", {{ summary: `[${{finding}}] ${{text}}`.slice(0, 500), category: "return", order_id: oid, image_url: imageInfo.image_url || null }});
      answer = `📸 Đã phân tích hình ảnh (${{imageInfo.name}}): ${{finding}} Mình đã tự động tạo phiếu hỗ trợ ${{tk.ticket_id}} (phụ trách: ${{tk.assignee}}, SLA ${{tk.sla_hours}} giờ) cho đơn ${{oid}}.`;
    }}

    if (!answer) {{
      intent = (sess.pending === "order_id" && orderMatch) ? "order_status" : null;
      if (gin.action === "handoff") intent = "handoff";
      if (!intent) {{
        intent = _ruleIntent(text);
        addSpan("classify_intent", "ok", {{ rule_intent: intent, intent }}, 0.3);
      }}
      sess.pending = null;
    }}

    if (sess.pending_action) {{
      const pa = sess.pending_action;
      const nt = _norm(text);
      if (/\\b(xac nhan|dong y|ok|co|chac chan|huy di)\\b/.test(nt)) {{
        sess.pending_action = null;
        intent = pa.tool;
        const r = callTool(pa.tool, pa.args);
        context.tool_results.push({{ tool: pa.tool, result: r }});
        answer = _mockGenerate(text, context);
      }} else if (/\\b(khong|thoi|giu nguyen|dung huy)\\b/.test(nt)) {{
        sess.pending_action = null;
        intent = pa.tool;
        answer = `Đã ghi nhận! Mình giữ nguyên đơn hàng ${{pa.args.order_id || ""}} cho bạn nhé.`;
      }} else {{
        sess.pending_action = null;
      }}
    }}

    if (!answer) {{
      const denied = _accessEdge(sess, intent, text, addSpan);
      if (denied) {{
        [answer, status] = denied; flags.push(denied[2]); sess.pending_action = null;
        if (denied[2] === "member_only" && flags.includes("abusive_language"))
          answer = "Mình rất tiếc vì trải nghiệm chưa tốt của bạn. Để nhân viên kiểm tra và xử lý trực tiếp, bạn đăng nhập tài khoản NORIVA giúp mình nhé. Mình vẫn có thể giải đáp chính sách đổi trả, vận chuyển ngay tại đây.";
      }}
    }}

    if (!answer) {{
      if (intent === "greeting") {{
        answer = "Chào bạn! Mình là trợ lý của NORIVA. Mình có thể giúp bạn tìm sản phẩm, tư vấn size, kiểm tra đơn hàng và giải đáp đổi trả, vận chuyển. Bạn cần gì nè?";
      }} else if (intent === "product_inquiry") {{
        const sm = text.match(/\\bsize\\s*(xs|s|m|l|xl|xxl|freesize)\\b/i);
        const pArgs = {{ query: text }};
        if (sm) pArgs.size = sm[1].toUpperCase();
        const r = callTool("search_products", pArgs);
        context.tool_results.push({{ tool: "search_products", result: r }});
        matchedProducts = (r.products || []).slice(0, 3).map(p => ({{ ...p, img: `img/${{p.id}}.jpg` }}));
        answer = _mockGenerate(text, context);
        addSpan("llm:generate", "ok", {{ model: "mock" }}, 0.5);
      }} else if (intent === "cancel_order") {{
        const oid = orderMatch ? `OC${{orderMatch[1]}}` : sess.last_order_id;
        if (!oid) {{
          sess.pending = "order_id";
          answer = "Bạn vui lòng cho mình xin mã đơn hàng muốn hủy (dạng OC + 5 chữ số, ví dụ OC10003) nhé.";
          status = "clarify";
        }} else {{
          sess.last_order_id = oid;
          sess.pending_action = {{ tool: "cancel_order", args: {{ order_id: oid, reason: text.slice(0, 200) }} }};
          flags.push("requires_confirmation");
          answer = `Bạn có chắc chắn muốn hủy đơn hàng ${{oid}} không? Vui lòng nhắn 'Xác nhận' để thực hiện hoặc 'Không' để giữ nguyên đơn.`;
          status = "confirm";
        }}
      }} else if (intent === "update_address") {{
        const oid = orderMatch ? `OC${{orderMatch[1]}}` : (sess.last_order_id || "OC10001");
        const mAddr = text.match(/(?:sang|thanh|den|la)\\s+(.+)$/i);
        const newAddr = mAddr ? mAddr[1] : text;
        sess.last_order_id = oid;
        sess.pending_action = {{ tool: "update_shipping_address", args: {{ order_id: oid, new_address: newAddr }} }};
        flags.push("requires_confirmation");
        answer = `Bạn xác nhận đổi địa chỉ giao hàng của đơn ${{oid}} sang "${{newAddr}}" chứ? Vui lòng bấm hoặc nhắn 'Xác nhận' để thực hiện.`;
        status = "confirm";
      }} else if (intent === "request_refund") {{
        const oid = orderMatch ? `OC${{orderMatch[1]}}` : (sess.last_order_id || "OC10002");
        sess.last_order_id = oid;
        sess.pending_action = {{ tool: "request_refund", args: {{ order_id: oid, reason: text.slice(0, 200) }} }};
        flags.push("requires_confirmation");
        answer = `Bạn xác nhận gửi yêu cầu hoàn tiền cho đơn hàng ${{oid}} chứ? Vui lòng bấm hoặc nhắn 'Xác nhận' để hệ thống tạo hồ sơ hoàn tiền.`;
        status = "confirm";
      }} else if (intent === "order_status") {{
      const oid = orderMatch ? `OC${{orderMatch[1]}}` : sess.last_order_id;
      addSpan("extract_order_id", "ok", {{ order_id: oid }}, 0.2);
      if (!oid) {{
        sess.pending = "order_id";
        answer = "Bạn vui lòng cho mình xin mã đơn hàng (dạng OC + 5 chữ số, ví dụ OC10001) nhé.";
        status = "clarify";
      }} else {{
        sess.last_order_id = oid;
        const r = callTool("get_order_status", {{ order_id: oid }});
        context.tool_results.push({{ tool: "get_order_status", result: r }});
        answer = _mockGenerate(text, context);
        addSpan("llm:generate", "ok", {{ model: "mock" }}, 0.5);
      }}
    }} else if (intent === "handoff") {{
      const reason = flags.includes("abusive_language") ? "Khách bức xúc" : "Khách yêu cầu gặp nhân viên";
      const r = callTool("handoff_to_human", {{ reason }});
      context.tool_results.push({{ tool: "handoff_to_human", result: r }});
      answer = _mockGenerate(text, context);
      status = "handoff";
      addSpan("llm:generate", "ok", {{ model: "mock" }}, 0.5);
    }} else if (intent === "complaint") {{
      const oid = orderMatch ? `OC${{orderMatch[1]}}` : sess.last_order_id;
      const kb = callTool("search_knowledge_base", {{ query: text }});
      context.documents = kb.results || [];
      const tNorm = _norm(text);
      const cat = tNorm.includes("hoan tien") ? "refund" : tNorm.includes("bao hanh") ? "warranty" : /doi|tra hang/.test(tNorm) ? "return" : "complaint";
      const tArgs = {{ summary: text.slice(0, 500), category: cat }};
      if (oid) tArgs.order_id = oid;
      const r = callTool("create_support_ticket", tArgs);
      context.tool_results.push({{ tool: "create_support_ticket", result: r }});
      answer = _mockGenerate(text, context);
      addSpan("llm:generate", "ok", {{ model: "mock" }}, 0.5);
    }} else {{
      const kb = callTool("search_knowledge_base", {{ query: text }});
      const docs = kb.results || [];
      context.documents = docs;
      if (!docs.length && !_SHOP_TOPIC_RE.test(_norm(text))) {{
        answer = _OFF_TOPIC_MSG; status = "declined"; intent = "off_topic"; flags.push("off_topic");
      }} else if (!docs.length) {{
        answer = "Xin lỗi, mình chưa thể trả lời câu này một cách chính xác. Bạn có muốn mình chuyển tới nhân viên chăm sóc khách hàng không?";
        status = "fallback";
        flags.push("no_grounding");
      }} else {{
        answer = _mockGenerate(text, context);
        addSpan("llm:generate", "ok", {{ model: "mock" }}, 0.5);
      }}
    }}
    }}
    citations = (context.documents || []).map(d => ({{ title: d.title, source: d.source, score: d.score }}));
    if (!["clarify", "fallback", "login_required", "denied"].includes(status) && intent !== "greeting") {{
      addSpan("guardrail:output", "ok", {{ action: "allow", flags: [] }}, 0.3);
    }}
  }}

  const latency = Number(Math.max(0.8, performance.now() - t0).toFixed(2));
  const traceId = Math.random().toString(16).slice(2, 14);
  sess.history.push({{ role: "user", content: gin.text }}, {{ role: "assistant", content: answer }});
  _EDGE_STATE.traces.unshift({{
    trace_id: traceId,
    session_id: sessionId,
    created_at: Date.now() / 1000,
    user_input: gin.text,
    output: answer,
    status,
    intent,
    latency_ms: latency,
    prompt_tokens: 42,
    completion_tokens: 28,
    guardrail_flags: flags,
    spans,
    attributes: {{ model: "mock", citations }},
    feedback: null
  }});

  return {{
    answer,
    trace_id: traceId,
    intent,
    status,
    citations,
    tool_calls: calls.map(c => ({{ tool: c.tool, args: c.args }})),
    guardrail_flags: flags,
    products: matchedProducts,
    latency_ms: latency
  }};
}}

const _SEED_GAPS_EDGE = [
  {{
    gap_id: "gap-alteration-pants",
    question: "Shop có hỗ trợ cắt/sửa gấu quần dài Smart Ankle miễn phí không?",
    count: 5,
    source_type: "fallback_trace",
    suggested_filename: "dich_vu_sua_gau_quan.md",
    suggested_title: "Dịch vụ sửa gấu quần và chăm sóc trang phục NORIVA",
    suggested_markdown: "# Dịch vụ sửa gấu quần và chăm sóc trang phục NORIVA\\n\\n## Chính sách cắt/sửa gấu quần miễn phí\\n- NORIVA hỗ trợ cắt và lên gấu quần miễn phí cho tất cả các dòng Quần Dài Smart Ankle, quần tây và quần jeans.\\n- Khách đặt online chỉ cần ghi chú số đo chiều dài ống quần mong muốn (cm); thời gian xử lý cộng thêm 24 giờ trước khi giao."
  }},
  {{
    gap_id: "gap-b2b-uniform",
    question: "Mua số lượng lớn làm đồng phục công ty từ 50 áo có chiết khấu và xuất VAT không?",
    count: 4,
    source_type: "fallback_trace",
    suggested_filename: "don_hang_doanh_nghiep_b2b.md",
    suggested_title: "Chính sách đơn hàng doanh nghiệp B2B & Đồng phục",
    suggested_markdown: "# Chính sách đơn hàng doanh nghiệp B2B & Đồng phục\\n\\n## Chiết khấu số lượng lớn và hóa đơn VAT\\n- Đơn hàng doanh nghiệp từ 30 đến 99 sản phẩm được chiết khấu 10% và từ 100 sản phẩm trở lên được chiết khấu 15%.\\n- Hỗ trợ xuất hóa đơn đỏ VAT 8% và dịch vụ thêu/in logo công ty trên Áo Polo Dry Pique, Áo Thun AIRism và Áo Khoác Parka."
  }},
  {{
    gap_id: "gap-click-collect",
    question: "Có thể đặt online rồi qua cửa hàng nhận (Click & Collect) để khỏi tốn phí ship không?",
    count: 3,
    source_type: "resolved_ticket",
    suggested_filename: "nhan_tai_cua_hang.md",
    suggested_title: "Dịch vụ Đặt Online Nhận Tại Cửa Hàng (Click & Collect)",
    suggested_markdown: "# Dịch vụ Đặt Online Nhận Tại Cửa Hàng (Click & Collect)\\n\\n## Quy trình nhận hàng tại showroom NORIVA\\n- Khách chọn hình thức Nhận tại cửa hàng (Click & Collect) được miễn phí vận chuyển 100% cho mọi giá trị đơn hàng.\\n- Đơn hàng sẵn sàng tại quầy sau 2 giờ kể từ lúc đặt và được giữ tại cửa hàng tối đa 7 ngày."
  }}
];
const _RESOLVED_GAPS_EDGE = new Set();

const _origFetch = window.fetch.bind(window);
window.fetch = async function(url, opts = {{}}) {{
  const path = typeof url === "string" ? url : url.url;
  if (!_EDGE_MODE) {{
    try {{
      const r = await _origFetch(url, opts);
      const ct = r.headers.get("content-type") || "";
      if (r.ok && ct.includes("application/json")) return r;
      if (path.startsWith("/api/") || path === "/health" || path === "/mcp") _EDGE_MODE = true;
      else return r;
    }} catch (e) {{
      _EDGE_MODE = true;
    }}
  }}
  const jsonResp = obj => new Response(JSON.stringify(obj), {{ status: 200, headers: {{ "Content-Type": "application/json" }} }});
  const body = opts.body ? JSON.parse(opts.body) : {{}};

  if (path === "/health") return jsonResp({{ status: "ok", model: "mock (cloud-edge)", kb_chunks: OMNICARE_SEED.chunks.length }});
  if (path === "/api/orders/create") {{
    const nextNum = 10001 + Object.keys(_EDGE_STATE.orders).length;
    const oid = "OC" + nextNum;
    const newOrder = {{
      order_id: oid,
      product: body.product || "Sản phẩm NORIVA",
      status: "Đang chuẩn bị hàng",
      carrier: "GHN",
      tracking: "GHN" + nextNum + "99",
      eta: "15/11/2026",
      total: body.total || 500000
    }};
    _EDGE_STATE.orders[oid] = newOrder;
    try {{
      const custom = JSON.parse(localStorage.getItem("omnicare_custom_orders") || "{{}}");
      custom[oid] = newOrder;
      localStorage.setItem("omnicare_custom_orders", JSON.stringify(custom));
    }} catch (e) {{}}
    return jsonResp(newOrder);
  }}
  // Bản tĩnh không có server: đăng nhập Admin chỉ là demo (tài khoản demo cố định, không bảo mật thật).
  if (path === "/api/admin/login") {{
    const demo = {{ admin: ["admin123", "Quản trị viên"], lan: ["123456", "Lan (CSKH Tier-2)"], linh: ["123456", "Linh (Bảo hành)"] }};
    const u = String(body.user || "").toLowerCase(), acc = demo[u];
    if (!acc || acc[0] !== body.password) return new Response(JSON.stringify({{ detail: "Sai tên đăng nhập hoặc mật khẩu." }}), {{ status: 401, headers: {{ "Content-Type": "application/json" }} }});
    return jsonResp({{ token: "edge-" + u, user: u, name: acc[1] }});
  }}
  if (path === "/api/admin/me" || path === "/api/admin/logout") return jsonResp({{ ok: true }});
  if (path === "/api/agent/overview") {{
    const docs = [...new Set(OMNICARE_SEED.chunks.map(c => c.source))].sort();
    const toolNames = new Set((OMNICARE_SEED.tools || []).map(t => t.name));
    const traces = _EDGE_STATE.traces || [];
    const byIntent = {{}}, toolUse = {{}};
    traces.forEach(t => {{ byIntent[t.intent] = (byIntent[t.intent] || 0) + 1; (t.spans || []).forEach(sp => {{ if ((sp.name || "").startsWith("tool:")) toolUse[sp.name.slice(5)] = (toolUse[sp.name.slice(5)] || 0) + 1; }}); }});
    const branches = (OMNICARE_SEED.branches || []).map(b => ({{ ...b,
      kb: (b.kb.length === 1 && b.kb[0] === "*") ? docs : b.kb.filter(d => docs.includes(d)),
      tools: b.tools.filter(t => toolNames.has(t)),
      count: [...b.intents, ...(b.intent_aliases || [])].reduce((n, i) => n + (byIntent[i] || 0), 0) }}));
    return jsonResp({{ shop: "NORIVA", model: {{ name: "mock (bản tĩnh)", fallbacks: [], embedding: "char-3gram" }},
      tools: (OMNICARE_SEED.tools || []).map(t => ({{ ...t, uses: toolUse[t.name] || 0 }})), kb_docs: docs,
      kb_chunks: OMNICARE_SEED.chunks.length, branches, total_requests: traces.length }});
  }}
  if (path === "/api/chat") return jsonResp(_handleChatEdge(body.session_id || "default", body.message || "", body.image_info || null, body.customer || null));
  if (path === "/api/llm/config") {{
    if (opts.method === "POST") _EDGE_STATE.active_model = `${{body.provider || "mock"}}:${{body.model || "mock-1"}}`;
    return jsonResp({{ ok: true, active_model: _EDGE_STATE.active_model || "mock (cloud-edge)" }});
  }}
  if (path.startsWith("/api/webhook/")) {{
    const ch = path.split("/").pop();
    const sid = `${{ch}}-${{body.sender_id || "user"}}`;
    const r = _handleChatEdge(sid, body.text || "", null, {{ type: "guest", guest_code: sid }});
    const chLabel = {{ facebook: "Facebook Messenger", zalo: "Zalo OA", telegram: "Telegram" }}[ch] || ch;
    return jsonResp({{ channel: ch, channel_label: chLabel, session_id: sid, sender_id: body.sender_id, reply: r.answer, status: r.status, intent: r.intent, trace_id: r.trace_id }});
  }}
  if (path === "/api/feedback") {{
    const t = _EDGE_STATE.traces.find(x => x.trace_id === body.trace_id);
    if (t) t.feedback = body.score;
    return jsonResp({{ ok: true }});
  }}
  if (path.startsWith("/api/traces/")) {{
    const tid = path.split("/").pop();
    return jsonResp(_EDGE_STATE.traces.find(x => x.trace_id === tid) || {{}});
  }}
  if (path.startsWith("/api/traces")) return jsonResp(_EDGE_STATE.traces.slice(0, 100));
  if (path === "/api/metrics") {{
    const tr = _EDGE_STATE.traces;
    const n = tr.length;
    if (!n) return jsonResp({{ total_requests: 0 }});
    const lats = tr.map(x => x.latency_ms).sort((a, b) => a - b);
    const byStatus = {{}}, byIntent = {{}};
    let negCount = 0;
    tr.forEach(x => {{
      byStatus[x.status] = (byStatus[x.status] || 0) + 1;
      byIntent[x.intent] = (byIntent[x.intent] || 0) + 1;
      if ((x.guardrail_flags || []).includes("negative_sentiment")) negCount++;
    }});
    const deflected = tr.filter(x => ["ok", "clarify", "confirm"].includes(x.status)).length;
    return jsonResp({{
      total_requests: n,
      latency_ms: {{ p50: lats[Math.floor(n * 0.5)] || 1, p95: lats[Math.floor(n * 0.95)] || 2, avg: 1.4 }},
      by_status: byStatus,
      by_intent: byIntent,
      deflection_rate: Number((deflected / n).toFixed(3)),
      handoff_rate: Number(((byStatus.handoff || 0) / n).toFixed(3)),
      blocked_rate: Number((((byStatus.blocked || 0) + (byStatus.declined || 0)) / n).toFixed(3)),
      negative_sentiment_count: negCount,
      estimated_cost_usd: Number(((n * 42 * 0.15 + n * 28 * 0.60) / 1000000).toFixed(6)),
      tokens: {{ prompt: n * 42, completion: n * 28, total: n * 70 }},
      feedback: {{ positive: tr.filter(x => x.feedback === 1).length, negative: tr.filter(x => x.feedback === -1).length }}
    }});
  }}
  if (path === "/api/kb/upload") {{
    const fname = (body.filename || "uploaded.md").replace(/[^a-zA-Z0-9_.-]/g, "_");
    const content = body.content || "";
    const title = (content.match(/^#\\s+(.+)$/m) || [])[1] || fname;
    const newChunk = {{ id: fname + "#0", source: fname, title, text: content }};
    OMNICARE_SEED.chunks.push(newChunk);
    OMNICARE_SEED.kb_documents.push({{ source: fname, title, chunks: [newChunk] }});
    _BM25.rebuild();
    return jsonResp({{ ok: true, filename: fname, total_chunks: OMNICARE_SEED.chunks.length }});
  }}
  if (path === "/api/copilot/suggest") {{
    const uMsgs = (body.messages || []).filter(m => m.role === "visitor" || m.role === "user").map(m => m.text || m.content || "");
    const lastQ = uMsgs[uMsgs.length - 1] || "Khách cần hỗ trợ";
    const intent = _ruleIntent(lastQ);
    const kb = _BM25.search(lastQ, 1);
    const suggestions = [];
    if (kb.length) suggestions.push(`Dạ chào anh/chị, theo quy định của shop: ${{kb[0].text.slice(0, 160)}} (Nguồn: ${{kb[0].title}}). Anh/chị cần em hỗ trợ thêm gì không ạ?`);
    suggestions.push("Dạ em đã tiếp nhận thông tin và đang ưu tiên kiểm tra xử lý cho anh/chị ngay ạ.");
    suggestions.push("Dạ cảm ơn anh/chị đã liên hệ OmniCare! Anh/chị vui lòng gửi thêm mã đơn hàng để em hỗ trợ nhanh nhất ạ.");
    return jsonResp({{ summary: `Khách hỏi ${{uMsgs.length}} lượt · Ý định: ${{intent}} · "${{lastQ.slice(0, 90)}}"`, detected_intent: intent, suggestions: suggestions.slice(0, 3) }});
  }}
  if (path.startsWith("/api/tickets/")) {{
    const tid = decodeURIComponent(path.split("/").pop());
    const tk = _EDGE_STATE.tickets.find(x => x.ticket_id === tid);
    if (tk) {{
      if (body.status) tk.status = body.status;
      if (body.assignee) tk.assignee = body.assignee;
    }}
    return jsonResp({{ ok: true, ticket: tk }});
  }}
  if (path === "/api/kb/reload") return jsonResp({{ chunks: OMNICARE_SEED.chunks.length }});
  if (path === "/api/kb/gaps") {{
    return jsonResp({{ gaps: _SEED_GAPS_EDGE.filter(g => !_RESOLVED_GAPS_EDGE.has(g.gap_id)), resolved_count: _RESOLVED_GAPS_EDGE.size }});
  }}
  if (path === "/api/kb/gaps/approve") {{
    const fname = (body.filename || "auto_faq.md").replace(/[^a-zA-Z0-9_.-]/g, "_");
    const content = body.content || "";
    const title = (content.match(/^#\\s+(.+)$/m) || [])[1] || fname;
    const newChunk = {{ id: fname + "#0", source: fname, title, text: content }};
    OMNICARE_SEED.chunks.push(newChunk);
    OMNICARE_SEED.kb_documents.push({{ source: fname, title, chunks: [newChunk] }});
    _BM25.rebuild();
    if (body.gap_id) _RESOLVED_GAPS_EDGE.add(body.gap_id);
    return jsonResp({{ ok: true, gap_id: body.gap_id, filename: fname, total_chunks: OMNICARE_SEED.chunks.length }});
  }}
  if (path === "/api/customers") return jsonResp({{ ok: true }});
  if (path === "/api/conversations" || path === "/api/omnichannel/conversations") {{
    let all = {{}};
    try {{ all = JSON.parse(localStorage.getItem("omnicare_store_sessions") || "{{}}"); }} catch (e) {{}}
    const out = Object.entries(all).map(([id, s]) => ({{ id, name: s.label || id, channel: s.channel || "Website NORIVA", takeover: Boolean(s.takeover), customer: s.customer, messages: s.messages || [], updated_at: Date.now() / 1000 }}));
    return jsonResp(out);
  }}
  if (path.startsWith("/api/conversations/")) {{
    const parts = path.split("/");
    const sid = decodeURIComponent(parts[3] || "");
    const action = parts[4] || "";
    let all = {{}};
    try {{ all = JSON.parse(localStorage.getItem("omnicare_store_sessions") || "{{}}"); }} catch (e) {{}}
    const sess = all[sid] || {{ id: sid, messages: [], takeover: false }};
    if (action === "takeover") {{
      sess.takeover = Boolean(body.takeover);
      all[sid] = sess;
      try {{ localStorage.setItem("omnicare_store_sessions", JSON.stringify(all)); }} catch (e) {{}}
      return jsonResp({{ ok: true, session_id: sid, takeover: sess.takeover }});
    }}
    if (action === "reply") {{
      sess.takeover = true;
      const msg = {{ role: "human", sender_name: body.agent_name || "Lan (CSKH Tier-2)", text: body.text || "", status: "human_reply", at: Date.now() }};
      const prev = (sess.messages || []).slice(-1)[0];
      const dup = prev && prev.role === "human" && prev.text === msg.text && Math.abs((prev.at || 0) - msg.at) < 10000;
      if (!dup) sess.messages = (sess.messages || []).concat([msg]);
      all[sid] = sess;
      try {{ localStorage.setItem("omnicare_store_sessions", JSON.stringify(all)); }} catch (e) {{}}
      return jsonResp({{ ok: true, session_id: sid, takeover: true, message: dup ? prev : msg, conversation: {{ id: sid, ...sess }} }});
    }}
    return jsonResp({{ id: sid, ...sess }});
  }}
  if (path === "/api/integrations/config") return jsonResp({{ ok: true, telegram_configured: Boolean(body.telegram_bot_token), telegram_token_masked: "Đã lưu", chatwoot_configured: Boolean(body.chatwoot_api_token) }});
  if (path === "/api/integrations/telegram/poll") return jsonResp({{ ok: true, processed_count: 0, results: [] }});
  if (path === "/api/integrations/telegram/webhook") {{
    const msg = body.message || {{}};
    const chatId = (msg.chat || {{}}).id || "849102";
    const name = [(msg.from || {{}}).first_name, (msg.from || {{}}).last_name].filter(Boolean).join(" ") || "Khách Telegram";
    const text = msg.text || "";
    const sid = `telegram-${{chatId}}`;
    const r = _handleChatEdge(sid, text, null, {{ type: "guest", guest_code: String(chatId), name, channel: "Telegram Bot" }});
    let all = {{}};
    try {{ all = JSON.parse(localStorage.getItem("omnicare_store_sessions") || "{{}}"); }} catch (e) {{}}
    const sess = all[sid] || {{ id: sid, label: name, channel: "Telegram Bot", customer: {{ type: "guest", guest_code: String(chatId), name, channel: "Telegram Bot" }}, messages: [] }};
    sess.messages.push({{ role: "visitor", text, at: Date.now() }}, {{ role: "agent", text: r.answer, intent: r.intent, status: r.status, at: Date.now() + 1 }});
    all[sid] = sess;
    try {{ localStorage.setItem("omnicare_store_sessions", JSON.stringify(all)); }} catch (e) {{}}
    return jsonResp({{ ok: true, session_id: sid, chat_id: chatId, sender_name: name, reply: r.answer, status: r.status, intent: r.intent }});
  }}
  if (path === "/api/integrations/chatwoot") {{
    const convId = (body.conversation || {{}}).id || "508";
    const sender = body.sender || {{}};
    const name = sender.name || `Chatwoot #${{convId}}`;
    const text = body.content || "";
    const sid = `chatwoot-${{convId}}`;
    const r = _handleChatEdge(sid, text, null, {{ type: "guest", guest_code: String(convId), name, channel: "Chatwoot" }});
    let all = {{}};
    try {{ all = JSON.parse(localStorage.getItem("omnicare_store_sessions") || "{{}}"); }} catch (e) {{}}
    const sess = all[sid] || {{ id: sid, label: name, channel: "Chatwoot", customer: {{ type: "guest", guest_code: String(convId), name, channel: "Chatwoot" }}, messages: [] }};
    sess.messages.push({{ role: "visitor", text, at: Date.now() }}, {{ role: "agent", text: r.answer, intent: r.intent, status: r.status, at: Date.now() + 1 }});
    all[sid] = sess;
    try {{ localStorage.setItem("omnicare_store_sessions", JSON.stringify(all)); }} catch (e) {{}}
    return jsonResp({{ ok: true, bridge: "chatwoot_agent_bot", session_id: sid, conversation_id: convId, reply: r.answer, status: r.status, intent: r.intent, chatwoot_action: r.status === "handoff" ? "toggle_status_open" : "reply_outgoing" }});
  }}
  // Bản tĩnh chỉ có cửa hàng NORIVA (đổi cửa hàng cần server Python)
  if (path === "/api/shops") return jsonResp({{ active: "noriva", shops: [{{ id: "noriva", name: "NORIVA", tagline: "Thời trang LifeWear — bản demo trực tuyến" }}] }});
  if (path === "/api/shops/active") return new Response(JSON.stringify({{ ok: false, detail: "Bản demo trực tuyến chỉ có NORIVA" }}), {{ status: 400, headers: {{ "Content-Type": "application/json" }} }});
  if (path === "/api/kb") return jsonResp({{ total_chunks: OMNICARE_SEED.chunks.length, documents: OMNICARE_SEED.kb_documents }});
  if (path === "/api/state") return jsonResp({{
    model: "mock",
    orders: Object.values(_EDGE_STATE.orders),
    tickets: _EDGE_STATE.tickets,
    tools: OMNICARE_SEED.tools,
    eval_report: _EDGE_STATE.eval_report
  }});
  if (path === "/api/evals/run") return jsonResp(_EDGE_STATE.eval_report);
  if (path === "/mcp") {{
    const {{ id, method, params = {{}} }} = body;
    if (method === "tools/list") return jsonResp({{ jsonrpc: "2.0", id, result: {{ tools: OMNICARE_SEED.tools }} }});
    if (method === "tools/call") {{
      const out = _callToolEdge(params.name, params.arguments || {{}});
      return jsonResp({{ jsonrpc: "2.0", id, result: {{ content: [{{ type: "text", text: JSON.stringify(out) }}] }} }});
    }}
  }}
  return _origFetch(url, opts);
}};
"""
    # Phần mở rộng (viết JS thuần, không qua f-string): nhớ ngữ cảnh, tư vấn size, báo có hàng, freeship, injection biến thể.
    ext = (ROOT / "scripts" / "edge_extensions.js").read_text(encoding="utf-8")
    return js_engine + "\n" + ext


def main() -> None:
    token = os.getenv("HF_TOKEN") or (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    space_name = os.getenv("HF_SPACE_NAME", "omnicare-ai-agent")

    src_html = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    store_html = (ROOT / "app" / "static" / "store.html").read_text(encoding="utf-8")
    engine_js = build_embedded_bundle()

    # Inject engine_js right at the beginning of the main <script> block
    marker = "<script>\n"
    if marker not in src_html or marker not in store_html:
        raise RuntimeError("Marker not found in index.html or store.html")
    built_html = src_html.replace(marker, "<script>\n" + engine_js + "\n", 1)
    built_store_html = store_html.replace(marker, "<script>\n" + engine_js + "\n", 1)

    import shutil

    build_dir = ROOT / ".hf_static_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(exist_ok=True)
    src_img = ROOT / "app" / "static" / "img"
    if src_img.exists():
        shutil.copytree(src_img, build_dir / "img")
    (build_dir / "index.html").write_text(built_html, encoding="utf-8")
    (build_dir / "store.html").write_text(built_store_html, encoding="utf-8")
    (build_dir / "README.md").write_text(
        "---\n"
        "title: OmniCare AI Agent\n"
        "emoji: 🤖\n"
        "colorFrom: indigo\n"
        "colorTo: blue\n"
        "sdk: static\n"
        "pinned: false\n"
        "---\n\n"
        "# OmniCare — AI Agent Customer Support System (TGO Architecture)\n\n"
        "6 MVP Components: Model · Workflow Agent · RAG + MCP Tools · Observability · Evals · Guardrails.\n",
        encoding="utf-8",
    )
    print(f"Built self-contained static app at: {build_dir / 'index.html'}")

    if not token:
        print("No HF_TOKEN provided; skipping upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{space_name}"
    print(f"Creating/updating Static Space: {repo_id} ...")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
        private=False,
    )
    api.upload_folder(
        folder_path=str(build_dir),
        repo_id=repo_id,
        repo_type="space",
    )
    direct_url = f"https://{user.lower()}-{space_name.lower()}.static.hf.space"
    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n=== DEPLOY COMPLETED ===")
    print(f"Space URL       : {space_url}")
    print(f"Direct App URL  : {direct_url}")


if __name__ == "__main__":
    main()
