/* 체험 데모 — 브라우저 판.
 *
 * 로컬 데모(demo/serve.py)는 PyTorch 가 /api/route 를 계산한다. 여기서는 같은 계산을 브라우저가 한다:
 *   qenc.onnx   그림 one-hot -> q, e_q
 *   slot.onnx   (e_q, 고른 그룹의 KV) -> 히트 · 위치 · 크기
 *   나머지(top-k · 임계값 · 인내 · q 보정 · 모양 판정 · 채점)는 아래 JS.
 * 규칙은 web/verify.py 의 onnx_loop · demo/serve.py 의 _detect_shape · diag/diag_m15.py 의 classify 와 같다.
 * route() 의 반환 모양은 serve.py do_route 와 같다 — 화면 코드는 로컬 데모 것을 그대로 쓴다.
 */
var Truffle = (function(){
  var BASE = new URL('assets/', document.currentScript.src).href;
  var M, keys, kv, qenc, slot, cands;         // 자산은 원본 모델 그대로 fp32 (web/export.py)

  function round3(x){                              // 파이썬 round(x, 3) — .5 는 짝수 쪽으로
    var v = x * 1000, f = Math.floor(v);
    return (v - f === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(v)) / 1000;
  }

  function fetchBytes(url, onBytes){
    return fetch(url).then(function(r){
      if (!r.ok) throw new Error(url + ' ' + r.status);
      if (!r.body || !r.body.getReader) return r.arrayBuffer();
      var rd = r.body.getReader(), parts = [], got = 0;
      function pump(){
        return rd.read().then(function(x){
          if (x.done) {
            var buf = new Uint8Array(got), o = 0;
            parts.forEach(function(p){ buf.set(p, o); o += p.length; });
            return buf.buffer;
          }
          parts.push(x.value); got += x.value.length; onBytes(x.value.length);
          return pump();
        });
      }
      return pump();
    });
  }

  // 생성기 마스크 (src/data.py _mask) — 모양 · 반지름 · 중심 전부. serve.py _detect_shape 와 같은 순서
  function buildCands(){
    var G = M.grid, out = [];
    for (var s = 0; s < M.n_shapes; s++)
      for (var ri = 0; ri < M.n_sizes; ri++) {
        var r = M.radii[ri], lo = Math.trunc(r), hi = Math.trunc((G - 1) - r);
        var thin = Math.max(Math.fround(r / 3), 1);
        for (var cy = lo; cy <= hi; cy++) for (var cx = lo; cx <= hi; cx++) {
          var m = new Uint8Array(G * G), n = 0;
          for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) {
            var dy = y - cy, dx = x - cx, on;
            if (s === 0) on = dy * dy + dx * dx <= r * r;
            else if (s === 1) on = dy >= -r && dy <= r && 2 * Math.abs(dx) <= dy + r;
            else if (s === 2) on = Math.max(Math.abs(dy), Math.abs(dx)) <= r;
            else on = (Math.abs(dy) <= r && Math.abs(dx) <= thin) || (Math.abs(dx) <= r && Math.abs(dy) <= thin);
            if (on) { m[y * G + x] = 1; n++; }
          }
          out.push({s: s, m: m, n: n});
        }
      }
    return out;
  }
  function detectShape(mask, nOn){
    if (nOn === 0) return {shape: -1, iou: 0};
    var best = -1, bi = -1;
    for (var c = 0; c < cands.length; c++) {
      var m = cands[c].m, inter = 0;
      for (var i = 0; i < m.length; i++) if (m[i] && mask[i]) inter++;
      var iou = Math.fround(inter / Math.max(cands[c].n + nOn - inter, 1));   // torch 는 float32 로 잰다
      if (iou > best) { best = iou; bi = c; }                                // 동점이면 앞 후보 (argmax)
    }
    return {shape: cands[bi].s, iou: round3(best)};
  }

  function load(onProgress){
    var total = 0, got = 0;
    function tick(n){ got += n; onProgress(got, total); }
    return fetch(BASE + 'episode.json').then(function(r){ return r.json(); }).then(function(meta){
      M = meta;
      total = 73.8 * 1048576;                      // web/export.py 산출물 합계 (표시용)
      ort.env.wasm.numThreads = 1;                 // GitHub Pages 는 교차 출처 격리가 없어 스레드를 못 쓴다
      return Promise.all([
        fetchBytes(BASE + 'keys.bin', tick), fetchBytes(BASE + 'kv.bin', tick),
        fetchBytes(BASE + 'qenc.onnx', tick), fetchBytes(BASE + 'slot.onnx', tick),
      ]);
    }).then(function(b){
      keys = new Float32Array(b[0]);
      kv = new Float32Array(b[1]);
      var opt = {executionProviders: ['wasm']};
      return Promise.all([
        ort.InferenceSession.create(new Uint8Array(b[2]), opt),
        ort.InferenceSession.create(new Uint8Array(b[3]), opt),
      ]);
    }).then(function(s){
      qenc = s[0]; slot = s[1]; cands = buildCands();
      return M;
    });
  }

  // frame: 16×16 (행 배열) — serve.py RouteIn 과 같은 입력
  async function route(frame, color, thr, patience){
    var G = M.grid, N = M.n_rooms, k = M.k, R = M.max_rounds, S = M.n_sym, dk = M.d_key;
    var L = M.n_layers, blk = M.frames_per_room * M.n_heads * M.d_head, half = kv.length / 2;
    patience = Math.max(0, patience);

    var flat = [], mask = new Uint8Array(G * G), nOn = 0;
    for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) {
      var v = Math.min(Math.max(frame[y][x] | 0, 0), M.n_colors);
      flat.push(v);
      if (v > 0) { mask[y * G + x] = 1; nOn++; }
    }
    var det = detectShape(mask, nOn);                                  // 모양은 그림에서만 나온다
    if (det.shape < 0)
      return {empty: true, shape: -1, shape_name: null, iou: 0, rounds: [], read: [], output: [],
              answers: [], solved: [], n_read: 0, read_frac: 0, n_rounds: 0, exact: false, cause: null};
    var target = color * M.n_shapes + det.shape;
    var truth = M.rooms.map(function(rm){ return rm.combo === target; });

    // 질의 인코딩
    var oh = new Float32Array(G * G * S);
    for (var c = 0; c < G * G; c++) oh[c * S + flat[c]] = 1;
    var qo = await qenc.run({oh: new ort.Tensor('float32', oh, [1, 1, G * G * S])});
    var q = Float64Array.from(qo.q.data), eq = Float32Array.from(qo.e_q.data);
    var dm = eq.length;

    var read = new Uint8Array(N), rounds = [], streak = 0;
    for (var rd = 0; rd < R; rd++) {
      var score = new Float64Array(N);
      for (var i = 0; i < N; i++) {
        if (read[i]) { score[i] = -30; continue; }
        var s = 0;
        for (var d = 0; d < dk; d++) s += keys[i * dk + d] * q[d];
        score[i] = s;
      }
      var sel = Array.from({length: N}, function(_, i){ return i; })
                     .sort(function(a, b){ return score[b] - score[a] || a - b; }).slice(0, k);
      var valid = sel.map(function(i){ return score[i] > -29; });
      sel.forEach(function(i, j){ if (valid[j]) read[i] = 1; });

      // 고른 그룹의 KV 를 잘라 붙인다 — kv.bin 은 (L,N,n,H,dh) K 먼저 V 나중
      var e = new Float32Array(k * dm), kg = new Float32Array(L * k * blk), vg = new Float32Array(L * k * blk);
      for (var j = 0; j < k; j++) e.set(eq, j * dm);
      for (var l = 0; l < L; l++) for (var j2 = 0; j2 < k; j2++) {
        var o = (l * N + sel[j2]) * blk, to = (l * k + j2) * blk;
        kg.set(kv.subarray(o, o + blk), to);
        vg.set(kv.subarray(half + o, half + o + blk), to);
      }
      var dims = [L, k, M.frames_per_room, M.n_heads, M.d_head];
      var so = await slot.run({e: new ort.Tensor('float32', e, [k, 1, dm]),
                               kg: new ort.Tensor('float32', kg, dims),
                               vg: new ort.Tensor('float32', vg, dims)});
      var hit = so.hit.data, pos = so.pos.data, size = so.size.data;
      var nP = pos.length / k, nS = size.length / k;

      var slots = [], nHit = 0, qa = new Float64Array(dk);
      for (var j3 = 0; j3 < k; j3++) {
        var room = sel[j3], p = 0, z = 0;
        for (var a = 1; a < nP; a++) if (pos[j3 * nP + a] > pos[j3 * nP + p]) p = a;
        for (var b2 = 1; b2 < nS; b2++) if (size[j3 * nS + b2] > size[j3 * nS + z]) z = b2;
        var dec = hit[j3] > thr && valid[j3], rm = M.rooms[room];
        if (dec) { nHit++; for (var d2 = 0; d2 < dk; d2++) qa[d2] += keys[room * dk + d2]; }
        slots.push({room: room, valid: valid[j3], hit: dec, logit: Math.round(hit[j3] * 100) / 100,
                    truth: truth[room], cy: Math.floor(p / G), cx: p % G, size: z,
                    ok: p === rm.cy * G + rm.cx && z === rm.size});
      }
      rounds.push(slots);

      if (nHit) {                                                      // q 보정 — 히트 판정 방들의 키 평균
        var nn = 0;
        for (var d3 = 0; d3 < dk; d3++) { q[d3] += qa[d3] / nHit; nn += q[d3] * q[d3]; }
        nn = Math.max(Math.sqrt(nn), 1e-12);
        for (var d4 = 0; d4 < dk; d4++) q[d4] /= nn;
        streak = 0;
      } else if (++streak > patience) break;                            // 0히트가 patience 번 넘게 이어지면 정지
    }

    // 채점 — diag_m15.classify 와 같은 정의
    var readL = [], outL = [], ansL = [], solvedL = [];
    var fp = false, fnRead = false, reconBad = false, nSolved = 0, lastHasAns = false;
    rounds.forEach(function(slots, r){
      slots.forEach(function(s){
        if (!s.valid) return;
        if (readL.indexOf(s.room) < 0) readL.push(s.room);
        if (s.hit && outL.indexOf(s.room) < 0) outL.push(s.room);
        if (s.hit && !s.truth) fp = true;
        if (s.truth) {
          if (r === rounds.length - 1) lastHasAns = true;
          if (!s.hit) fnRead = true;
          else if (!s.ok) reconBad = true;
          else { nSolved++; solvedL.push(s.room); }
        }
      });
    });
    truth.forEach(function(t, i){ if (t) ansL.push(i); });
    var unreadAns = ansL.some(function(i){ return !read[i]; });
    var exact = nSolved === ansL.length && !fp, fail = !exact, hitMax = rounds.length >= R;
    var cause = {A: fail && unreadAns && !hitMax && !lastHasAns, B: fail && unreadAns && !hitMax && lastHasAns,
                 C: fail && unreadAns && hitMax, D: fail && !unreadAns && fnRead, E: fail && fp, F: fail && reconBad};
    var num = function(a, b){ return a - b; };
    return {
      empty: false, shape: det.shape, shape_name: M.shapes[det.shape], iou: det.iou, color: color,
      rounds: rounds, read: readL.sort(num), output: outL.sort(num), answers: ansL, solved: solvedL.sort(num),
      n_read: readL.length, read_frac: Math.round(1000 * readL.length / N) / 10, n_rounds: rounds.length,
      exact: exact, cause: 'ABCDEF'.split('').filter(function(c){ return cause[c]; })[0] || null,
    };
  }

  return {load: load, route: route};
})();

(function(){
  var ui = document.getElementById('dm-ui'), off = document.getElementById('dm-off');
  if (!ui) return;
  var G = 16, cv = document.getElementById('dmc'), ctx = cv.getContext('2d');
  var frame = [], ep = null, color = 0, erase = false, busy = false, again = false, timer = null;
  for (var i = 0; i < G * G; i++) frame.push(0);

  function cssVar(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
  function paint(){
    ctx.clearRect(0, 0, G, G);
    if (!ep) return;
    for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) {
      var v = frame[y * G + x];
      if (!v) continue;
      ctx.fillStyle = cssVar(ep.colors[v - 1].var);
      ctx.fillRect(x, y, 1, 1);
    }
  }
  function glyph(shape, col, size){
    var c = 'var(' + ep.colors[col].var + ')', r = 8 + size * 3;
    if (shape === 0) return '<circle cx="24" cy="24" r="' + r + '" fill="' + c + '"/>';
    if (shape === 1) return '<polygon points="24,' + (24 - r) + ' ' + (24 - r) + ',' + (24 + r) + ' ' + (24 + r) + ',' + (24 + r) + '" fill="' + c + '"/>';
    if (shape === 2) return '<rect x="' + (24 - r) + '" y="' + (24 - r) + '" width="' + (2 * r) + '" height="' + (2 * r) + '" fill="' + c + '"/>';
    var t = Math.max(3, r / 3);
    return '<rect x="' + (24 - t) + '" y="' + (24 - r) + '" width="' + (2 * t) + '" height="' + (2 * r) + '" fill="' + c + '"/>' +
           '<rect x="' + (24 - r) + '" y="' + (24 - t) + '" width="' + (2 * r) + '" height="' + (2 * t) + '" fill="' + c + '"/>';
  }
  function drawAt(e){
    var b = cv.getBoundingClientRect();
    var p = e.touches ? e.touches[0] : e;
    var x = Math.floor((p.clientX - b.left) / b.width * G), y = Math.floor((p.clientY - b.top) / b.height * G);
    if (x < 0 || y < 0 || x >= G || y >= G) return;
    frame[y * G + x] = erase ? 0 : color + 1;
    paint(); queue();
  }
  var down = false;
  cv.addEventListener('pointerdown', function(e){ down = true; cv.setPointerCapture(e.pointerId); drawAt(e); });
  cv.addEventListener('pointermove', function(e){ if (down) drawAt(e); });
  cv.addEventListener('pointerup', function(){ down = false; });
  cv.addEventListener('pointercancel', function(){ down = false; });

  function queue(){ clearTimeout(timer); timer = setTimeout(run, 180); }
  function setStat(id, v, s){
    document.getElementById(id).textContent = v;
    if (s != null) document.getElementById(id + '-s').textContent = s;
  }
  function run(){
    if (busy) { again = true; return; }
    busy = true;
    var grid2d = [];
    for (var y = 0; y < G; y++) grid2d.push(frame.slice(y * G, y * G + G));
    var thr = parseFloat(document.getElementById('dm-thr').value);
    var pat = parseInt(document.getElementById('dm-pat').value, 10);
    Truffle.route(grid2d, color, thr, pat)
      .then(render)
      .catch(function(err){ console.error(err); })
      .then(function(){ busy = false; if (again) { again = false; run(); } });
  }
  function glyphAt(shape, col, cy, cx, size){   // 16×16 좌표 안에 재현 — 모델이 예측한 위치·크기로
    var c = 'var(' + ep.colors[col].var + ')', r = size + 2, y = cy + 0.5, x = cx + 0.5;
    if (shape === 0) return '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" fill="' + c + '"/>';
    if (shape === 1) return '<polygon points="' + x + ',' + (y - r) + ' ' + (x - r) + ',' + (y + r)
                          + ' ' + (x + r) + ',' + (y + r) + '" fill="' + c + '"/>';
    if (shape === 2) return '<rect x="' + (x - r) + '" y="' + (y - r) + '" width="' + (2 * r)
                          + '" height="' + (2 * r) + '" fill="' + c + '"/>';
    var t = Math.max(1, r / 3);
    return '<rect x="' + (x - t) + '" y="' + (y - r) + '" width="' + (2 * t) + '" height="' + (2 * r) + '" fill="' + c + '"/>'
         + '<rect x="' + (x - r) + '" y="' + (y - t) + '" width="' + (2 * r) + '" height="' + (2 * t) + '" fill="' + c + '"/>';
  }
  function render(res){
    var cq = document.querySelectorAll('.dm-cq');
    Array.prototype.forEach.call(cq, function(c){ c.className = 'dm-cq'; });
    document.getElementById('dm-recon').innerHTML = '';
    var det = document.getElementById('dm-det');
    if (res.empty) {
      det.className = 'dm-det none'; det.textContent = '아직 안 그렸다';
      setStat('dm-read', '—', 'full KV 대비'); setStat('dm-rounds', '—');
      setStat('dm-exact', '—', '출력 집합 = 정답 집합');
      document.getElementById('dm-exact').style.color = '';
      return;
    }
    det.className = 'dm-det';
    det.innerHTML = '<svg viewBox="0 0 48 48" style="width:18px;height:18px" aria-hidden="true">'
      + glyph(res.shape, res.color, 1) + '</svg>' + ep.colors[res.color].name + ' ' + res.shape_name
      + '<span class="iou">일치도 ' + res.iou.toFixed(2) + '</span>';
    setStat('dm-read', res.n_read + ' / 64', 'full KV 대비 ' + res.read_frac + ' %');
    setStat('dm-rounds', String(res.n_rounds));
    setStat('dm-exact', res.exact ? '맞음' : '틀림',
      res.exact ? '출력 = 정답 ' + res.answers.length + '개'
                : (res.cause ? '원인 ' + res.cause : '') + ' · 정답 ' + res.answers.length + ' · 출력 ' + res.output.length);
    document.getElementById('dm-exact').style.color = res.exact ? 'var(--good)' : 'var(--crit)';
    // 1) 캐시 점등 — 라운드마다 순서대로
    res.rounds.forEach(function(slots, r){
      setTimeout(function(){
        slots.forEach(function(s){
          if (!s.valid) return;
          var c = cq[s.room]; if (!c) return;
          c.classList.add('read');
          if (s.hit) c.classList.add(s.truth ? 'ans' : 'fp');
        });
      }, r * 170);
    });
    var after = res.rounds.length * 170;
    setTimeout(function(){
      res.answers.forEach(function(a){ if (res.output.indexOf(a) < 0) cq[a].classList.add('miss'); });
    }, after);

    // 2) 재현 — 찾은 순서대로 하나씩 그려 낸다
    var found = [];
    res.rounds.forEach(function(slots){
      slots.forEach(function(s){ if (s.valid && s.hit) found.push(s); });
    });
    var strip = document.getElementById('dm-recon');
    found.forEach(function(s, i){
      var el = document.createElement('div');
      el.className = 'dm-rc ' + (s.truth && s.ok ? 'ok' : 'no');
      el.title = '방 ' + s.room + ' — 재현 (' + s.cy + ',' + s.cx + ') 크기 ' + s.size
               + (s.truth ? (s.ok ? ' · 맞음' : ' · 위치·크기 틀림') : ' · 헛것');
      el.innerHTML = '<svg viewBox="0 0 16 16" style="width:100%;display:block" aria-hidden="true">'
                   + glyphAt(res.shape, res.color, s.cy, s.cx, s.size) + '</svg>'
                   + '<div class="rid">' + s.room + '</div>';
      strip.appendChild(el);
      setTimeout(function(){ el.classList.add('on'); }, after + 120 + i * 220);
    });
  }

  var msg = document.getElementById('dm-off-msg');
  Truffle.load(function(got, total){
    msg.textContent = '모델을 받는 중 — ' + (got / 1048576).toFixed(1) + ' / ' + (total / 1048576).toFixed(1) + ' MB (첫 방문만)';
  }).then(function(d){
    ep = d; ui.hidden = false; off.hidden = true;
    var cw = document.getElementById('dm-colors'), sw = document.getElementById('dm-samples');
    function drawSample(s){          // 본보기를 캔버스에 찍는다 — 입력이 아니라 그리기 보조다
      for (var i = 0; i < G * G; i++) frame[i] = 0;
      var r = 4, t = Math.max(1, Math.round(r / 3));
      for (var y = 0; y < G; y++) for (var x = 0; x < G; x++) {
        var dy = y - 8, dx = x - 8, on = false;
        if (s === 0) on = dy * dy + dx * dx <= r * r;
        else if (s === 1) on = (dy >= -r && dy <= r && 2 * Math.abs(dx) <= dy + r);
        else if (s === 2) on = Math.max(Math.abs(dy), Math.abs(dx)) <= r;
        else on = (Math.abs(dy) <= r && Math.abs(dx) <= t) || (Math.abs(dx) <= r && Math.abs(dy) <= t);
        if (on) frame[y * G + x] = color + 1;
      }
      paint(); queue();
    }
    function renderSamples(){        // 현재 색으로 네 모양을 보여 준다
      sw.innerHTML = '';
      d.shapes.forEach(function(nm, i){
        var b = document.createElement('button');
        b.className = 'dm-btn'; b.type = 'button';
        b.innerHTML = '<svg viewBox="0 0 48 48" style="width:17px;height:17px;vertical-align:-3px" aria-hidden="true">'
                    + glyph(i, color, 1) + '</svg> ' + nm;
        b.onclick = function(){ drawSample(i); };
        sw.appendChild(b);
      });
    }
    d.colors.forEach(function(c, i){
      var b = document.createElement('button');
      b.className = 'dm-sw' + (i === color ? ' on' : ''); b.type = 'button';
      b.style.background = 'var(' + c.var + ')'; b.title = c.name; b.setAttribute('aria-label', c.name);
      b.onclick = function(){
        color = i; erase = false;
        Array.prototype.forEach.call(cw.children, function(x, j){ x.classList.toggle('on', j === i); });
        document.getElementById('dm-erase').classList.remove('on');
        for (var p = 0; p < frame.length; p++) if (frame[p]) frame[p] = color + 1;   // 그린 것도 새 색으로
        renderSamples(); paint(); queue();
      };
      cw.appendChild(b);
    });
    renderSamples();
    var g = document.getElementById('dm-rooms');
    d.rooms.forEach(function(rm){
      var el = document.createElement('div');
      el.className = 'dm-cell';
      el.title = '방 ' + rm.id + ' · ' + d.colors[rm.color].name + d.shapes[rm.shape]
               + ' · (' + rm.cy + ',' + rm.cx + ') 크기 ' + rm.size;
      el.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true">'      // 재현 패널과 같은 좌표계
                   + glyphAt(rm.shape, rm.color, rm.cy, rm.cx, rm.size) + '</svg>';
      g.appendChild(el);
    });
    var cg = document.getElementById('dm-cache');
    for (var ci = 0; ci < d.n_rooms; ci++) {
      var q = document.createElement('div');
      q.className = 'dm-cq'; q.title = '캐시 그룹 ' + ci;
      cg.appendChild(q);
    }
    document.getElementById('dm-erase').onclick = function(){
      erase = !erase; this.classList.toggle('on', erase); };
    document.getElementById('dm-clear').onclick = function(){
      for (var i = 0; i < G * G; i++) frame[i] = 0; paint(); queue(); };
    ['dm-thr','dm-pat'].forEach(function(id){
      var el = document.getElementById(id);
      el.addEventListener('input', function(){
        var v = parseFloat(el.value);
        document.getElementById(id + '-v').textContent = id === 'dm-thr' ? (v >= 0 ? '+' : '') + v.toFixed(1) : String(v);
        var f = parseFloat(document.getElementById('dm-thr').value) === 3 && parseInt(document.getElementById('dm-pat').value, 10) === 2;
        document.getElementById('dm-dialnote').classList.toggle('warn', !f);
        queue();
      });
    });
    paint(); drawSample(1);          // 처음엔 삼각 본보기를 하나 넣어 둔다
  }).catch(function(err){
    console.error(err);
    msg.textContent = '모델을 불러오지 못했다 — ' + err.message + ' (문서의 나머지 파트는 그대로 읽힌다)';
  });
})();
