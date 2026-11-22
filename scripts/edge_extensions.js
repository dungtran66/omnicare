
// ===================== EDGE EXTENSIONS (đồng bộ với app/agent.py & app/guardrails.py) =====================
// Nhớ ngữ cảnh nhiều lượt, tư vấn size, đăng ký báo có hàng, gợi ý freeship, chặn prompt injection biến thể.
(function () {
  const FREESHIP = 500000;
  const SIZE_CHART = [["S", 155, 165, 45, 54], ["M", 163, 172, 55, 64], ["L", 170, 178, 65, 74], ["XL", 175, 185, 75, 86]];
  const PRODUCT_WORDS = /\b(parka|airism|cardigan|smart ankle|linen|ultra light|polo|kinh mat|ao chong uv|ao khoac|ao thun|quan dai|ao phao|so mi)\b/;
  const HEIGHT_RE = /(?:\b1\s?m\s?(\d{1,2})\b|\b(1[4-9]\d)\s?cm\b)/i;
  const WEIGHT_RE = /\b(\d{2,3})\s?(?:kg|ky|ký|kí|ki)\b/i;
  const SIZE_RE = /\bsize\s*(xs|s|m|l|xl|xxl)\b/i;
  const ORDER_RE = /\bOC\s?-?(\d{5})\b/i;
  const INJ_VARIANTS = [
    /\b(ignore|bo qua|quen|phot lo|lo di)\b.{0,30}\b(rule|rules|luat|quy tac|huong dan|instruction|instructions|lenh)\b/,
    /\b(cho|show|in|xem|dump|leak|lo)\b.{0,25}\b(system ?prompt|prompt (he thong|goc|an)|sys prompt|huong dan an)\b/,
    /\b(prompt|huong dan) (he thong|goc)\b.{0,20}\b(la gi|ra|di|xem)\b/
  ];
  const vnd = n => Number(n).toLocaleString("vi-VN") + "đ";
  const products = () => OMNICARE_SEED.products || [];
  const findProduct = q => {
    const r = _callToolEdge("search_products", { query: q });
    return (r.products || [])[0] || null;
  };
  _EDGE_STATE.restock = _EDGE_STATE.restock || [];

  // ---- Stylist phối đồ theo dịp (giống app/ai_features.py: OUTFIT_RE, _rule_outfit) ----
  const OUTFIT_RE = /\b(phoi (do|ao|quan|sao|the nao|cho)|mac gi|nen mac|mac sao|mac the nao|outfit|set do|bo do|di (bien|du lich|choi|lam|tiec|phong van|hen ho|cong so|leo nui|da lat|sapa|picnic)|chon do|tu van (do|trang phuc|cach mac))\b/;
  const ROLE = { "Quần": "bottom", "Áo Khoác": "outer", "Sơ Mi": "top", "AIRism": "top", "UV Protection": "outer" };
  const OUTFIT_RULES = [
    [/bien|nang|du lich|picnic|he|leo nui/, ["AIRism", "UV Protection", "Quần"]],
    [/phong van|cong so|di lam|lich su|tiec|hen ho/, ["Sơ Mi", "Quần"]],
    [/lanh|dong|da lat|sapa|ret/, ["Áo Khoác", "AIRism", "Quần"]],
  ];
  function ruleOutfit(t) {
    const cats = (OUTFIT_RULES.find(([rx]) => rx.test(t)) || [null, ["AIRism", "Quần"]])[1];
    const inStock = p => Object.values(p.sizes || {}).some(n => n > 0);
    const picked = [], used = new Set();
    for (const cat of cats) {
      const p = products().slice().sort((a, b) => a.price - b.price).find(p => {
        const role = _norm(p.name).includes("kinh") ? "acc" : (ROLE[p.category] || "top");
        return p.category === cat && !used.has(role) && inStock(p);
      });
      if (p) { picked.push(p); used.add(_norm(p.name).includes("kinh") ? "acc" : (ROLE[p.category] || "top")); }
    }
    if (/bien|nang|du lich|picnic|he/.test(t)) {
      const acc = products().find(p => _norm(p.name).includes("kinh"));
      if (acc && !picked.includes(acc)) picked.push(acc);
    }
    return picked;
  }

  // ---- Tool mới: recommend_size, subscribe_restock ----
  const baseCall = _callToolEdge;
  _callToolEdge = function (name, args) {
    if (name === "recommend_size") {
      const h = Number(args.height_cm), w = Number(args.weight_kg);
      const order = SIZE_CHART.map(r => r[0]);
      const pick = (v, lo, hi) => {
        for (const r of SIZE_CHART) if (r[lo] <= v && v <= r[hi]) return r[0];
        return v < SIZE_CHART[0][lo] ? SIZE_CHART[0][0] : SIZE_CHART[SIZE_CHART.length - 1][0];
      };
      const byH = pick(h, 1, 2), byW = pick(w, 3, 4);
      const size = order.indexOf(byH) >= order.indexOf(byW) ? byH : byW;
      const res = { height_cm: h, weight_kg: w, size_by_height: byH, size_by_weight: byW, recommended_size: size,
        between_sizes: byH !== byW, out_of_chart: h < 155 || h > 185 || w < 45 || w > 86 };
      if (args.product) {
        const p = findProduct(args.product);
        if (p) Object.assign(res, { product: p.name, product_id: p.id, size_stock: (p.sizes || {})[size] || 0 });
      }
      return res;
    }
    if (name === "subscribe_restock") {
      const p = findProduct(args.product || "");
      const sub = { subscription_id: "RS-" + Math.random().toString(16).slice(2, 8).toUpperCase(), product: p ? p.name : args.product,
        size: args.size ? String(args.size).toUpperCase() : null, contact: args.contact, status: "registered" };
      _EDGE_STATE.restock.push(sub);
      return sub;
    }
    return baseCall(name, args);
  };

  function isTakeover(sessionId, sess) {
    let t = Boolean(sess.takeover);
    try {
      const all = JSON.parse(localStorage.getItem("omnicare_store_sessions") || "{}");
      if (all[sessionId] && typeof all[sessionId].takeover === "boolean") t = all[sessionId].takeover;
    } catch (e) {}
    return t;
  }
  function respond(sessionId, sess, text, answer, intent, status, flags, calls, spans) {
    const traceId = Math.random().toString(16).slice(2, 14);
    const latency = 1.2;
    sess.history.push({ role: "user", content: text }, { role: "assistant", content: answer });
    _EDGE_STATE.traces.unshift({ trace_id: traceId, session_id: sessionId, created_at: Date.now() / 1000, user_input: text, output: answer,
      status, intent, latency_ms: latency, prompt_tokens: 30, completion_tokens: 40, guardrail_flags: flags, spans,
      attributes: { model: "mock", citations: [] }, feedback: null });
    return { answer, trace_id: traceId, intent, status, citations: [], tool_calls: calls.map(c => ({ tool: c.tool, args: c.args })),
      guardrail_flags: flags, products: [], latency_ms: latency };
  }
  // Thẻ sản phẩm = đúng các sản phẩm câu trả lời nhắc tới (giống OmniCareAgent._cards_for_answer).
  function cardsForAnswer(answer, matched) {
    const na = _norm(answer);
    const words = new Set(na.match(/\w+/g) || []);
    const found = [];
    for (const p of products()) {
      const n = _norm(p.name);
      const toks = (n.match(/\w+/g) || []).filter(w => w.length > 1);
      let pos = na.indexOf(n);
      if (pos < 0 && toks.length && toks.filter(w => words.has(w)).length / toks.length >= 0.75)
        pos = Math.min(...toks.filter(w => words.has(w)).map(w => na.indexOf(w)));
      if (pos >= 0) found.push([pos, p]);
    }
    if (!found.length) return matched.slice(0, 1);
    found.sort((a, b) => a[0] - b[0]);
    return found.map(([, p]) => matched.find(m => m.id === p.id) || { ...p, img: `img/${p.id}.jpg` });
  }

  function upsellHint(prods, text) {
    const hints = [];
    const top = prods[0];
    const sm = text.match(SIZE_RE);
    if (sm && (top.sizes || {})[sm[1].toUpperCase()] === 0)
      hints.push(`Size ${sm[1].toUpperCase()} đang hết, bạn nhắn "báo khi có hàng" để mình báo ngay khi về hàng nhé.`);
    const price = Number(top.price || 0);
    if (price > 0 && price < FREESHIP) {
      const o = products().filter(p => p.id !== top.id && price + p.price >= FREESHIP).sort((a, b) => a.price - b.price)[0];
      if (o) hints.push(`À, bạn lấy thêm ${o.name} (${vnd(o.price)}) là đơn được miễn phí giao hàng luôn đó, tiết kiệm được tiền ship nè.`);
    }
    return hints.join(" ");
  }

  const baseHandle = _handleChatEdge;
  _handleChatEdge = function (sessionId, message, imageInfo = null, customer = null) {
    const sess = (_EDGE_STATE.sessions[sessionId] = _EDGE_STATE.sessions[sessionId] || { history: [], last_order_id: null, pending: null });
    if (customer) sess.customer = customer;
    let text = (message || "").trim();
    if (imageInfo || !text || isTakeover(sessionId, sess)) return baseHandle(sessionId, message, imageInfo, customer);
    const t = _norm(text);
    const spans = [], calls = [], flags = [];
    const addSpan = (name, attrs) => spans.push({ name, duration_ms: 0.3, status: "ok", attributes: attrs });

    // 1) Prompt injection biến thể (trộn Anh-Việt, teencode)
    if (INJ_VARIANTS.some(r => r.test(t))) {
      addSpan("guardrail:input", { action: "block", flags: ["prompt_injection"] });
      return respond(sessionId, sess, text, "Xin lỗi, mình không thể xử lý yêu cầu này. Yêu cầu chứa nội dung cố thay đổi chỉ dẫn hệ thống.",
        "blocked", "blocked", ["prompt_injection"], calls, spans);
    }
    addSpan("guardrail:input", { action: "allow", flags: [] });

    // 1b) Stylist phối đồ (bản luật, giống ai_features._rule_outfit; bản server dùng model)
    if (OUTFIT_RE.test(t)) {
      const items = ruleOutfit(t);
      const total = items.reduce((n, p) => n + p.price, 0);
      let ans = items.length
        ? "Mình gợi ý bạn bộ này nè: " + items.map(p => `${p.name} (${vnd(p.price)})`).join(", ") + "."
        : "Hiện shop chưa có món phù hợp với nhu cầu này, bạn mô tả thêm dịp mặc để mình tư vấn lại nhé.";
      if (items.length > 1) ans += ` Cả bộ ${vnd(total)}` + (total >= FREESHIP ? ", được miễn phí giao hàng luôn." : ".");
      addSpan("stylist:pick_outfit", { items: items.map(p => p.id), total, ai: false });
      const r = respond(sessionId, sess, text, ans, "outfit_advice", "ok", flags, calls, spans);
      r.products = items.map(p => ({ id: p.id, name: p.name, price: p.price, category: p.category, sizes: p.sizes, colors: p.colors, img: `img/${p.id}.jpg` }));
      if (items.length) sess.last_product = items[0].name;
      return r;
    }

    // 2) Tư vấn size theo chiều cao + cân nặng
    const hm = text.match(HEIGHT_RE), wm = text.match(WEIGHT_RE);
    if (hm && wm) {
      const h = hm[1] ? 100 + (hm[1].length === 1 ? Number(hm[1]) * 10 : Number(hm[1])) : Number(hm[2]), w = Number(wm[1]);
      const args = { height_cm: h, weight_kg: w };
      const prod = PRODUCT_WORDS.test(t) ? text : sess.last_product;
      if (prod) args.product = prod;
      addSpan("context:resolve", { intent: "size_advice" });
      const r = _callToolEdge("recommend_size", args);
      calls.push({ tool: "recommend_size", args, result: r });
      addSpan("tool:recommend_size", { result_preview: JSON.stringify(r).slice(0, 200) });
      let ans = `Với chiều cao ${h}cm và cân nặng ${w}kg, bạn hợp nhất với size ${r.recommended_size} theo bảng size NORIVA.`;
      if (r.between_sizes) ans += ` Chiều cao của bạn ứng với size ${r.size_by_height} còn cân nặng ứng với size ${r.size_by_weight}, nên mình chọn size lớn hơn để mặc thoải mái.`;
      if (r.out_of_chart) ans += " Số đo của bạn nằm ngoài bảng size chuẩn, bạn nên thử trực tiếp hoặc hỏi nhân viên.";
      if (r.product) ans += r.size_stock ? ` ${r.product} size ${r.recommended_size} hiện còn ${r.size_stock} sản phẩm.`
        : ` ${r.product} size ${r.recommended_size} đang tạm hết, bạn nhắn 'báo khi có hàng' để mình báo nhé.`;
      ans += " Nếu mặc không vừa, bạn được đổi size miễn phí trong 30 ngày khi còn nguyên tem mác.";
      return respond(sessionId, sess, text, ans, "size_advice", "ok", flags, calls, spans);
    }

    // 3) Đăng ký báo khi có hàng
    if (/\b(bao|thong bao|nhan tin|bao lai)\b.*\b(khi|lai)\b.*\bco hang\b|\bco hang lai\b|\bkhi nao co hang\b/.test(t)) {
      const prod = PRODUCT_WORDS.test(t) ? text : (sess.last_product || "");
      if (!prod) return respond(sessionId, sess, text, "Bạn muốn nhận thông báo cho sản phẩm và size nào ạ?", "restock_notify", "clarify", flags, calls, spans);
      const c = sess.customer || {};
      const sm = text.match(SIZE_RE) || (sess.last_size_query || "").match(SIZE_RE);
      const args = { product: prod, contact: c.phone || (c.guest_code ? "Khách #" + c.guest_code : sessionId) };
      if (sm) args.size = sm[1].toUpperCase();
      const r = _callToolEdge("subscribe_restock", args);
      calls.push({ tool: "subscribe_restock", args, result: r });
      addSpan("tool:subscribe_restock", { result_preview: JSON.stringify(r).slice(0, 200) });
      return respond(sessionId, sess, text, `Mình đã ghi nhận yêu cầu ${r.subscription_id}: báo cho bạn khi ${r.product}${r.size ? " size " + r.size : ""} có hàng trở lại.`,
        "restock_notify", "ok", flags, calls, spans);
    }

    // 4) Giải quyết tham chiếu nhiều lượt ("hủy nó đi", "cái áo đó còn size L không")
    if (sess.last_order_id && !ORDER_RE.test(text)) {
      if (/\bhuy (no|di|luon|don (do|nay)|nhe)\b/.test(t) || /\b(don (do|nay)|no)\b.*\b(den dau|o dau|giao chua|trang thai)\b/.test(t)
          || /\b(hoan tien|tra tien)\b.*\b(don (do|nay)|no)\b/.test(t)) {
        text = `${text} ${sess.last_order_id}`;
      }
    }
    if (sess.last_product && !PRODUCT_WORDS.test(t) && /\b(con size|con hang|gia bao nhieu|bao nhieu tien|co mau|mau (gi|nao)|ton kho)\b/.test(t)) {
      text = /\bcon size|con hang|ton kho\b/.test(t) ? `${text} ${sess.last_product} kiem tra kho` : `${text} ${sess.last_product}`;
    }
    if (/\bhuy (no|di|luon|don (do|nay)|nhe)\b/.test(t) && !/\bhuy don\b/.test(_norm(text))) text = text.replace(/hủy|huy/i, "Hủy đơn");

    const r = baseHandle(sessionId, text, imageInfo, customer);
    const m = text.match(ORDER_RE);
    if (m && r.status !== "denied" && r.status !== "login_required") sess.last_order_id = "OC" + m[1];
    if (r.intent === "product_inquiry" && r.status === "ok" && (r.products || []).length) {
      sess.last_product = r.products[0].name;
      if (SIZE_RE.test(text)) sess.last_size_query = text;
      const full = products().find(p => p.id === r.products[0].id) || r.products[0];
      const extra = upsellHint([full, ...r.products.slice(1)], text);
      if (extra) {
        r.answer = `${r.answer} ${extra}`;
        if (_EDGE_STATE.traces[0] && _EDGE_STATE.traces[0].trace_id === r.trace_id) _EDGE_STATE.traces[0].output = r.answer;
        const h = sess.history[sess.history.length - 1];
        if (h && h.role === "assistant") h.content = r.answer;
      }
    }
    if ((r.products || []).length) r.products = cardsForAnswer(r.answer, r.products);
    return r;
  };
})();
