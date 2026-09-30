'use strict';
// Gradient-boosting core shared by the three Lecture 2B widgets.
// One second-order tree learner serves every loss:
//   leaf value  gamma = -G / (H + lambda)
//   split gain  = 1/2 [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda) ]
// with G, H the sums of first and second loss derivatives in a node.
// MSE (L = 1/2 (y-F)^2):  g = F - y, h = 1           -> gamma = mean residual (lambda = 0)
// BCE (F = log-odds):     g = p - y, h = p (1 - p)   (Newton) or h = 1 (gradient)
const GB = (() => {
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normal(r) {
    let u = 0;
    while (!u) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }

  // 1D regression: y = sin(2 pi x) + 0.6 x + N(0, 0.25^2)
  const truth1 = (x) => Math.sin(2 * Math.PI * x) + 0.6 * x;
  function regData(seed, n) {
    const r = rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) {
      const x = r();
      X.push([x]);
      y.push(truth1(x) + 0.25 * normal(r));
    }
    return { X, y };
  }

  // 2D classification on the unit square, labels y in {0, 1}, 6% label flips.
  const shapes = {
    disk: ([a, b]) => (a - 0.5) ** 2 + (b - 0.5) ** 2 < 0.3 ** 2,
    xor: ([a, b]) => (a - 0.5) * (b - 0.5) > 0,
    diagonal: ([a, b]) => b > a + 0.08 * Math.sin(8 * a),
  };
  function clsData(shape, seed, n, flip = 0.06) {
    const r = rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) {
      const x = [r(), r()];
      let label = shapes[shape](x) ? 1 : 0;
      if (r() < flip) label = 1 - label;
      X.push(x);
      y.push(label);
    }
    return { X, y };
  }

  const EPS = 1e-12;
  function fitTree(X, g, h, { maxDepth = 2, lambda = 0, minLeaf = 2 } = {}) {
    const d = X[0].length;
    function grow(idx, depth) {
      let G = 0, H = 0;
      for (const i of idx) { G += g[i]; H += h[i]; }
      const leaf = { value: -G / (H + lambda + EPS), n: idx.length, G, H };
      if (depth >= maxDepth || idx.length < 2 * minLeaf) return leaf;
      const base = (G * G) / (H + lambda + EPS);
      let best = null;
      for (let k = 0; k < d; k++) {
        const s = idx.slice().sort((a, b) => X[a][k] - X[b][k]);
        let GL = 0, HL = 0;
        for (let j = 0; j < s.length - 1; j++) {
          GL += g[s[j]]; HL += h[s[j]];
          const nl = j + 1, a = X[s[j]][k], b = X[s[j + 1]][k];
          if (a === b || nl < minLeaf || s.length - nl < minLeaf) continue;
          const GR = G - GL, HR = H - HL;
          const gain = 0.5 * ((GL * GL) / (HL + lambda + EPS) + (GR * GR) / (HR + lambda + EPS) - base);
          if (!best || gain > best.gain) best = { gain, feature: k, cut: (a + b) / 2, j, s };
        }
      }
      if (!best || best.gain <= 1e-12) return leaf;
      return {
        feature: best.feature, cut: best.cut, gain: best.gain, n: idx.length, G, H,
        left: grow(best.s.slice(0, best.j + 1), depth + 1),
        right: grow(best.s.slice(best.j + 1), depth + 1),
      };
    }
    return grow(X.map((_, i) => i), 0);
  }
  const predict = (t, x) => (t.left ? predict(x[t.feature] < t.cut ? t.left : t.right, x) : t.value);
  const leaves = (t) => (t.left ? [...leaves(t.left), ...leaves(t.right)] : [t]);

  const sigmoid = (F) => 1 / (1 + Math.exp(-F));
  const losses = {
    mse: {
      init: (y) => y.reduce((a, b) => a + b, 0) / y.length,
      grad: (y, F) => F - y,
      hess: () => 1,
      value: (y, F) => (y - F) ** 2, // reported as plain MSE
    },
    bce: {
      init: (y) => { const m = y.reduce((a, b) => a + b, 0) / y.length; return Math.log(m / (1 - m)); },
      grad: (y, F) => sigmoid(F) - y,
      hess: (y, F) => { const p = sigmoid(F); return p * (1 - p); },
      value: (y, F) => Math.log1p(Math.exp(-Math.abs(F))) + Math.max(F, 0) - y * F,
    },
  };

  // Train M rounds. Returns F0, the trees, and per-round losses on both samples.
  // opt.newton = false uses h = 1 (fit the negative gradient by least squares).
  function boost(train, val, lossName, { rounds = 100, eta = 0.3, maxDepth = 2, lambda = 0, newton = true, minLeaf = 2 } = {}) {
    const L = losses[lossName], F0 = L.init(train.y);
    const Ft = train.y.map(() => F0), Fv = val.y.map(() => F0);
    const mean = (d, F) => d.y.reduce((s, y, i) => s + L.value(y, F[i]), 0) / d.y.length;
    const trees = [], trainLoss = [mean(train, Ft)], valLoss = [mean(val, Fv)], before = [];
    for (let m = 0; m < rounds; m++) {
      const g = train.y.map((y, i) => L.grad(y, Ft[i]));
      const h = train.y.map((y, i) => (newton ? L.hess(y, Ft[i]) : 1));
      const t = fitTree(train.X, g, h, { maxDepth, lambda, minLeaf });
      before.push(Ft.slice());
      trees.push(t);
      train.X.forEach((x, i) => { Ft[i] += eta * predict(t, x); });
      val.X.forEach((x, i) => { Fv[i] += eta * predict(t, x); });
      trainLoss.push(mean(train, Ft));
      valLoss.push(mean(val, Fv));
    }
    const F = (x, m) => { let s = F0; for (let k = 0; k < m; k++) s += eta * predict(trees[k], x); return s; };
    return { F0, trees, eta, F, before, trainLoss, valLoss };
  }

  return { rng, normal, truth1, regData, clsData, fitTree, predict, leaves, sigmoid, losses, boost };
})();

// ---------- canvas helpers (same look as the Lecture 2A widgets) ----------
const COL = { gold: '#d99700', blue: '#253b53', accent: '#087bb9', red: '#c74842', grid: '#c3d0da', muted: '#617484', orange: '#d97a00' };
function makeCanvas(id) {
  const C = document.getElementById(id), ctx = C.getContext('2d');
  const P = {
    ctx, W: C.width, H: C.height,
    clear() { ctx.clearRect(0, 0, C.width, C.height); },
    text(s, x, y, size = 23, color = COL.blue, align = 'left') {
      ctx.fillStyle = color; ctx.font = `${size}px "Helvetica Neue",Arial`; ctx.textAlign = align; ctx.fillText(s, x, y); ctx.textAlign = 'left';
    },
    // Axis frame with data ranges xr = [x0, x1], yr = [y0, y1]; returns mapping helpers.
    frame(x, y, w, h, title, xlabel, ylabel, xr = [0, 1], yr = [0, 1], digits = 1) {
      P.text(title, x, y - 18, 25);
      ctx.strokeStyle = COL.grid; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
      const f = { x, y, w, h, xr, yr,
        X: (v) => x + ((v - xr[0]) / (xr[1] - xr[0])) * w,
        Y: (v) => y + h - ((v - yr[0]) / (yr[1] - yr[0])) * h };
      for (let i = 0; i <= 4; i++) {
        const t = i / 4, xv = xr[0] + t * (xr[1] - xr[0]), yv = yr[0] + t * (yr[1] - yr[0]);
        P.text(Number.isInteger(xv) ? String(xv) : xv.toFixed(2), x + t * w, y + h + 26, 17, COL.muted, 'center');
        P.text(Number.isInteger(yv) ? String(yv) : yv.toFixed(digits), x - 8, y + h - t * h + 6, 17, COL.muted, 'right');
      }
      P.text(xlabel, x + w / 2, y + h + 54, 20, COL.blue, 'center');
      ctx.save(); ctx.translate(x - 58, y + h / 2); ctx.rotate(-Math.PI / 2); P.text(ylabel, 0, 0, 20, COL.blue, 'center'); ctx.restore();
      return f;
    },
    clip(f, draw) { ctx.save(); ctx.beginPath(); ctx.rect(f.x, f.y, f.w, f.h); ctx.clip(); draw(); ctx.restore(); },
    line(f, pts, color, width = 3, dash = []) {
      P.clip(f, () => {
        ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath();
        pts.forEach(([a, b], i) => (i ? ctx.lineTo(f.X(a), f.Y(b)) : ctx.moveTo(f.X(a), f.Y(b))));
        ctx.stroke(); ctx.setLineDash([]);
      });
    },
    dot(f, a, b, color, r = 6, stroke = 'white') {
      ctx.beginPath(); ctx.arc(f.X(a), f.Y(b), r, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill();
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
    },
    // Colour map over the unit square: value(x) -> CSS colour.
    field(f, colour, n = 70) {
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        ctx.fillStyle = colour([(i + 0.5) / n, (j + 0.5) / n]);
        ctx.fillRect(f.x + (i * f.w) / n, f.y + ((n - j - 1) * f.h) / n, f.w / n + 0.6, f.h / n + 0.6);
      }
    },
  };
  return P;
}
// Blend between two RGB triplets.
function mix(a, b, t) { t = Math.max(0, Math.min(1, t)); return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`; }
const RGB = { sig: [255, 214, 122], bkg: [173, 199, 222], white: [247, 247, 247] };
// Diverging map: -1 -> background blue, 0 -> white, +1 -> signal gold.
const diverge = (s) => (s < 0 ? mix(RGB.white, RGB.bkg, -s) : mix(RGB.white, RGB.sig, s));
function stat(items) { document.getElementById('stats').innerHTML = items.map(([k, v]) => `<span>${k}<br><b>${v}</b></span>`).join(''); }
function syncOutputs() { document.querySelectorAll('input').forEach((e) => { const o = document.getElementById(e.id + '-value'); if (o) o.value = (+e.value).toFixed(e.dataset.digits ? +e.dataset.digits : 0); }); }
if (typeof module !== 'undefined') module.exports = GB;
