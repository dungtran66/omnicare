/*!
 * OmniCare chat widget — dán vào website bất kỳ:
 *   <script src="https://<server-omnicare>/widget.js" async></script>
 * Tùy chọn (thuộc tính data-*): data-title, data-color, data-welcome, data-position="left|right".
 * Widget nói chuyện với đúng server đã phục vụ file này (mỗi shop chạy một server OmniCare riêng).
 * Khách trên widget là khách vãng lai: hỏi sản phẩm, size, chính sách; tra đơn cần đăng nhập trên trang của shop.
 * MIT License.
 */
(function () {
  'use strict';
  if (window.__omnicareWidget) return;
  window.__omnicareWidget = true;

  var script = document.currentScript || (function () {
    var s = document.querySelectorAll('script[src*="widget.js"]'); return s[s.length - 1];
  })();
  var BASE = new URL(script.src, location.href).origin;
  var OPT = {
    title: script.getAttribute('data-title') || '',
    color: script.getAttribute('data-color') || '#1f1f1f',
    welcome: script.getAttribute('data-welcome') || '',
    left: script.getAttribute('data-position') === 'left'
  };
  if (!/^#[0-9a-fA-F]{3,8}$/.test(OPT.color)) OPT.color = '#1f1f1f';

  // ---- Lưu trữ theo tab (người sau dùng chung máy không thấy chat của người trước) ----
  var KEY = 'omnicare_widget:' + BASE;
  function load() { try { return JSON.parse(sessionStorage.getItem(KEY)) || null; } catch (e) { return null; } }
  function save() { try { sessionStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
  function rand(n) {
    var a = 'abcdefghijkmnpqrstuvwxyz23456789', b = new Uint8Array(n), o = '';
    (window.crypto || window.msCrypto).getRandomValues(b);
    for (var i = 0; i < n; i++) o += a[b[i] % a.length];
    return o;
  }
  function newSession() {
    var code = rand(4).toUpperCase();
    return { sid: 'w-' + code.toLowerCase() + '-' + rand(20), code: code, msgs: [], seen: 0, open: false };
  }
  var state = load() || newSession();

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function money(v) { return Number(v || 0).toLocaleString('vi-VN') + 'đ'; }
  function api(path, body) {
    return fetch(BASE + path, body ? {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    } : {}).then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); });
  }

  // ---- Giao diện trong Shadow DOM để không đụng CSS của website ----
  var host = document.createElement('div');
  host.id = 'omnicare-widget';
  document.body.appendChild(host);
  var root = host.attachShadow ? host.attachShadow({ mode: 'open' }) : host;
  var side = OPT.left ? 'left' : 'right';
  root.innerHTML =
    '<style>' +
    ':host{all:initial}' +
    '*{box-sizing:border-box;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}' +
    '.btn{position:fixed;' + side + ':20px;bottom:20px;width:58px;height:58px;border-radius:50%;border:0;cursor:pointer;' +
    'background:var(--c);color:#fff;box-shadow:0 8px 24px rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center;z-index:2147483000}' +
    '.btn svg{width:28px;height:28px}' +
    '.panel{position:fixed;' + side + ':20px;bottom:90px;width:370px;max-width:calc(100vw - 32px);height:560px;max-height:calc(100vh - 110px);' +
    'background:#fff;color:#111;border-radius:16px;box-shadow:0 16px 48px rgba(0,0,0,.28);display:none;flex-direction:column;overflow:hidden;z-index:2147483000}' +
    '.panel.on{display:flex}' +
    '.head{background:var(--c);color:#fff;padding:14px 16px;display:flex;align-items:center;gap:10px}' +
    '.head b{font-size:15px;flex:1}.head small{display:block;font-size:11.5px;opacity:.8;font-weight:400}' +
    '.x{background:none;border:0;color:#fff;font-size:22px;cursor:pointer;line-height:1}' +
    '.msgs{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;background:#f6f7f9}' +
    '.m{max-width:85%;padding:9px 12px;border-radius:14px;font-size:14px;line-height:1.45;white-space:pre-wrap;word-wrap:break-word}' +
    '.ai{background:#fff;border:1px solid #e5e7eb;align-self:flex-start;border-bottom-left-radius:4px}' +
    '.nv{background:#ecfdf5;border:1px solid #a7f3d0;align-self:flex-start;border-bottom-left-radius:4px}' +
    '.nv i{display:block;font-style:normal;font-size:11px;color:#047857;margin-bottom:2px;font-weight:600}' +
    '.me{background:var(--c);color:#fff;align-self:flex-end;border-bottom-right-radius:4px}' +
    '.sys{align-self:center;font-size:12px;color:#6b7280;text-align:center}' +
    '.cards{display:flex;gap:8px;overflow-x:auto;margin-top:8px;padding-bottom:2px}' +
    '.card{flex:0 0 130px;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;background:#fff;font-size:12px}' +
    '.card img{width:100%;height:96px;object-fit:cover;display:block;background:#eee}' +
    '.card div{padding:6px 8px}.card b{display:block;font-weight:600;line-height:1.3;max-height:2.6em;overflow:hidden}' +
    '.acts{display:flex;gap:6px;margin-top:8px}' +
    '.acts button{border:1px solid var(--c);background:#fff;color:var(--c);border-radius:999px;padding:5px 12px;font-size:12.5px;cursor:pointer}' +
    '.acts button:first-child{background:var(--c);color:#fff}' +
    '.typing{display:none;align-self:flex-start;font-size:12.5px;color:#6b7280}' +
    '.chips{display:flex;gap:6px;flex-wrap:wrap;padding:8px 12px 0;background:#fff}' +
    '.chips button{border:1px solid #d1d5db;background:#fff;border-radius:999px;padding:5px 10px;font-size:12.5px;cursor:pointer;color:#111}' +
    'form{display:flex;gap:8px;padding:10px 12px;border-top:1px solid #eee;background:#fff}' +
    'input{flex:1;border:1px solid #d1d5db;border-radius:999px;padding:9px 14px;font-size:14px;outline:none;color:#111;background:#fff}' +
    'input:focus{border-color:var(--c)}' +
    'form button{border:0;background:var(--c);color:#fff;border-radius:999px;padding:0 16px;font-size:14px;cursor:pointer}' +
    'form button:disabled{opacity:.5}' +
    '.foot{text-align:center;font-size:10.5px;color:#9ca3af;padding:0 0 8px;background:#fff}' +
    '</style>' +
    '<button class="btn" part="button" aria-label="Mở khung chat">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></button>' +
    '<div class="panel" role="dialog" aria-label="Chat hỗ trợ">' +
    '<div class="head"><b><span class="t">Hỗ trợ khách hàng</span><small>AI trả lời ngay · nhân viên hỗ trợ khi cần</small></b><button class="x" aria-label="Đóng">×</button></div>' +
    '<div class="msgs"></div>' +
    '<div class="chips"></div>' +
    '<form><input placeholder="Nhập câu hỏi…" maxlength="1000" aria-label="Tin nhắn"><button>Gửi</button></form>' +
    '<div class="foot">Được hỗ trợ bởi OmniCare AI</div>' +
    '</div>';
  host.style.setProperty('--c', OPT.color);
  var $ = function (s) { return root.querySelector(s); };
  var panel = $('.panel'), msgsEl = $('.msgs'), input = $('input'), sendBtn = $('form button');
  var typing = document.createElement('div'); typing.className = 'typing'; typing.textContent = 'Đang trả lời…';
  var shopName = '', busy = false, pollTimer = null;

  function img(src) { return !src ? '' : /^https?:|^data:/.test(src) ? src : BASE + (src[0] === '/' ? '' : '/') + src; }
  function bubble(m) {
    if (m.role === 'sys') return '<div class="sys">' + esc(m.text) + '</div>';
    if (m.role === 'me') return '<div class="m me">' + esc(m.text) + '</div>';
    if (m.role === 'nv') return '<div class="m nv"><i>' + esc(m.name || 'Nhân viên') + '</i>' + esc(m.text) + '</div>';
    var cards = (m.products || []).slice(0, 6).map(function (p) {
      return '<div class="card"><img src="' + esc(img(p.image || ('/img/' + p.id + '.jpg'))) + '" alt="" loading="lazy" onerror="this.style.visibility=\'hidden\'"><div><b>' +
        esc(p.name) + '</b>' + money(p.price) + '</div></div>';
    }).join('');
    var acts = m.confirm ? '<div class="acts"><button data-q="Xác nhận">Xác nhận</button><button data-q="Không, giữ nguyên">Không</button></div>' : '';
    return '<div class="m ai">' + esc(m.text) + (cards ? '<div class="cards">' + cards + '</div>' : '') + acts + '</div>';
  }
  function render() {
    var welcome = OPT.welcome || ('Chào bạn! Mình là trợ lý AI của ' + (shopName || 'shop') +
      '. Mình có thể tư vấn sản phẩm, chọn size, phí ship và chính sách đổi trả.');
    msgsEl.innerHTML = '<div class="m ai">' + esc(welcome) + '</div>' + state.msgs.map(bubble).join('');
    msgsEl.appendChild(typing);
    var last = state.msgs[state.msgs.length - 1];
    $('.chips').style.display = state.msgs.length ? 'none' : 'flex';
    if (last && last.confirm) {
      [].forEach.call(msgsEl.querySelectorAll('.acts button'), function (b) { b.onclick = function () { send(b.getAttribute('data-q')); }; });
    }
    msgsEl.scrollTop = msgsEl.scrollHeight;
  }

  function send(text) {
    text = String(text || '').trim();
    if (!text || busy) return;
    if (state.closed) { state = newSession(); state.open = true; }
    state.msgs.push({ role: 'me', text: text });
    state.msgs.forEach(function (m) { m.confirm = false; });
    busy = true; sendBtn.disabled = true; render(); typing.style.display = 'block'; msgsEl.scrollTop = msgsEl.scrollHeight;
    var channel = 'Website ' + (location.hostname || 'nhúng widget');
    api('/api/chat', { message: text, session_id: state.sid,
      customer: { type: 'guest', guest_code: state.code, channel: channel } })
      .then(function (res) {
        if (res.answer && !res.takeover) {
          state.msgs.push({ role: 'ai', text: res.answer, products: res.products || [], confirm: res.status === 'confirm' });
        }
      })
      .catch(function () { state.msgs.push({ role: 'sys', text: 'Chưa gửi được tin nhắn, bạn thử lại sau ít giây nhé.' }); })
      .then(function () { busy = false; sendBtn.disabled = false; typing.style.display = 'none'; save(); render(); });
  }

  // Tin nhắn của nhân viên (tiếp quản trên Admin) và việc đóng hội thoại
  function poll() {
    if (!state.msgs.length || state.closed) return;
    api('/api/conversations/' + encodeURIComponent(state.sid)).then(function (c) {
      var human = (c.messages || []).filter(function (m) { return m.role === 'human'; });
      if (human.length > (state.seen || 0)) {
        human.slice(state.seen || 0).forEach(function (m) { state.msgs.push({ role: 'nv', text: m.text, name: m.sender_name }); });
        state.seen = human.length; save(); render();
      }
      if (c.closed_at) {
        state.closed = true;
        state.msgs.push({ role: 'sys', text: 'Hội thoại đã kết thúc. Cảm ơn bạn! Nhắn tiếp để bắt đầu hội thoại mới.' });
        save(); render();
      }
    }).catch(function () {});
  }
  function toggle(on) {
    state.open = on; save();
    panel.classList.toggle('on', on);
    clearInterval(pollTimer);
    if (on) { render(); poll(); pollTimer = setInterval(poll, 4000); setTimeout(function () { input.focus(); }, 50); }
  }

  $('.btn').onclick = function () { toggle(!panel.classList.contains('on')); };
  $('.x').onclick = function () { toggle(false); };
  $('form').onsubmit = function (e) { e.preventDefault(); var t = input.value; input.value = ''; send(t); };
  [['👕 Tư vấn size', 'Mình cao 1m65 nặng 55kg mặc size gì?'], ['🚚 Phí ship', 'Phí giao hàng bao nhiêu, bao nhiêu thì được freeship?'],
   ['🔄 Đổi trả', 'Chính sách đổi trả trong bao nhiêu ngày?']].forEach(function (c) {
    var b = document.createElement('button'); b.textContent = c[0]; b.onclick = function () { send(c[1]); }; $('.chips').appendChild(b);
  });

  api('/api/shop').then(function (s) {
    shopName = s.name || '';
    $('.t').textContent = OPT.title || (shopName ? shopName + ' hỗ trợ' : 'Hỗ trợ khách hàng');
    render();
  }).catch(function () { render(); });
  if (OPT.title) $('.t').textContent = OPT.title;
  render();
  if (state.open) toggle(true);
})();
