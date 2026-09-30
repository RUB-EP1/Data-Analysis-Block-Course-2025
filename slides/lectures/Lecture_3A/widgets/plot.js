'use strict';
// Small canvas toolkit shared by the Lecture 3 widgets. No dependencies.
// Colours follow the Lecture 2A widgets: gold = class 1, dark blue = class 0.
const PAL = {
  gold: '#d99700', blue: '#253b53', accent: '#087bb9', red: '#c74842',
  green: '#3c8d52', violet: '#7a4fb3', muted: '#617484', grid: '#c3d0da',
  light: '#e7f3fc', ink: '#172635'
};
const $ = id => document.getElementById(id);

// Deterministic random numbers, so every reload shows the same data.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) {
  const u = Math.max(r(), 1e-12), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function text(ctx, s, x, y, size = 22, color = PAL.ink, align = 'left', weight = '') {
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px "Helvetica Neue",Arial,sans-serif`;
  ctx.textAlign = align;
  ctx.fillText(s, x, y);
  ctx.textAlign = 'left';
}

// A rectangular data window mapped onto the canvas.
class Panel {
  constructor(ctx, x, y, w, h, xr, yr) {
    Object.assign(this, { ctx, x, y, w, h, xr, yr });
  }
  X(v) { return this.x + (v - this.xr[0]) / (this.xr[1] - this.xr[0]) * this.w; }
  Y(v) { return this.y + this.h - (v - this.yr[0]) / (this.yr[1] - this.yr[0]) * this.h; }
  inv(px, py) {
    return [this.xr[0] + (px - this.x) / this.w * (this.xr[1] - this.xr[0]),
            this.yr[0] + (this.y + this.h - py) / this.h * (this.yr[1] - this.yr[0])];
  }
  contains(px, py) { return px >= this.x && px <= this.x + this.w && py >= this.y && py <= this.y + this.h; }
  frame(title, xlabel, ylabel, xt, yt, fmt = v => String(+v.toFixed(2))) {
    const c = this.ctx;
    if (title) text(c, title, this.x, this.y - 16, 25, PAL.ink, 'left', '600');
    c.strokeStyle = PAL.grid; c.lineWidth = 1;
    c.strokeRect(this.x, this.y, this.w, this.h);
    (xt || []).forEach(v => text(c, fmt(v), this.X(v), this.y + this.h + 26, 17, PAL.muted, 'center'));
    (yt || []).forEach(v => text(c, fmt(v), this.x - 8, this.Y(v) + 6, 17, PAL.muted, 'right'));
    if (xlabel) text(c, xlabel, this.x + this.w / 2, this.y + this.h + 54, 20, PAL.ink, 'center');
    if (ylabel) {
      c.save(); c.translate(this.x - 52, this.y + this.h / 2); c.rotate(-Math.PI / 2);
      text(c, ylabel, 0, 0, 20, PAL.ink, 'center'); c.restore();
    }
  }
  clip(fn) {
    const c = this.ctx; c.save(); c.beginPath(); c.rect(this.x, this.y, this.w, this.h); c.clip(); fn(); c.restore();
  }
  line(pts, color = PAL.accent, width = 3, dash = []) {
    const c = this.ctx; c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash);
    c.beginPath();
    pts.forEach(([a, b], i) => { const X = this.X(a), Y = this.Y(b); i ? c.lineTo(X, Y) : c.moveTo(X, Y); });
    c.stroke(); c.setLineDash([]);
  }
  fn(f, color, width, dash, n = 400) {
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const x = this.xr[0] + i / n * (this.xr[1] - this.xr[0]);
      pts.push([x, f(x)]);
    }
    this.clip(() => this.line(pts, color, width, dash));
  }
  dot(x, y, color, r = 6, stroke = 'white', lw = 1.5) {
    const c = this.ctx; c.beginPath(); c.arc(this.X(x), this.Y(y), r, 0, 2 * Math.PI);
    c.fillStyle = color; c.fill();
    if (stroke) { c.strokeStyle = stroke; c.lineWidth = lw; c.stroke(); }
  }
  hline(y, color = PAL.grid, width = 1, dash = [6, 6]) { this.line([[this.xr[0], y], [this.xr[1], y]], color, width, dash); }
  vline(x, color = PAL.grid, width = 1, dash = [6, 6]) { this.line([[x, this.yr[0]], [x, this.yr[1]]], color, width, dash); }
  // Paint f(x, y) in [0, 1] as a colour field with n×n cells.
  field(f, n = 60, colour = probColour) {
    const c = this.ctx, cw = this.w / n, ch = this.h / n;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const [x, y] = this.inv(this.x + (i + .5) * cw, this.y + (j + .5) * ch);
      c.fillStyle = colour(f(x, y));
      c.fillRect(this.x + i * cw, this.y + j * ch, cw + .6, ch + .6);
    }
  }
}

function mix(a, b, t) {
  const p = s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16));
  const A = p(a), B = p(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
}
// 0 → pale blue, 0.5 → white, 1 → pale gold: the same classes as the points.
function probColour(p) {
  p = Math.min(1, Math.max(0, p));
  return p < .5 ? mix('#b9cde0', '#ffffff', p * 2) : mix('#ffffff', '#ffd98a', (p - .5) * 2);
}
// Signed values: negative blue, positive gold.
function signedColour(v, scale = 1) {
  const t = Math.max(-1, Math.min(1, v / scale));
  return t < 0 ? mix('#ffffff', '#3d6f99', -t) : mix('#ffffff', '#d99700', t);
}
// Perceptually ordered map for activation strength (dark → yellow).
function viridis(t) {
  t = Math.min(1, Math.max(0, t));
  const s = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'];
  const k = Math.min(3, Math.floor(t * 4));
  return mix(s[k], s[k + 1], t * 4 - k);
}

function arrow(ctx, x0, y0, x1, y1, color = PAL.ink, width = 2.5, head = 12) {
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(a - .4), y1 - head * Math.sin(a - .4));
  ctx.lineTo(x1 - head * Math.cos(a + .4), y1 - head * Math.sin(a + .4));
  ctx.closePath(); ctx.fill();
}

// Keep every <output id="X-value"> in sync with its <input id="X">.
function syncOutputs(fmt = {}) {
  document.querySelectorAll('input[type=range]').forEach(e => {
    const o = $(e.id + '-value');
    if (o) o.textContent = (fmt[e.id] || (v => v))(+e.value);
  });
}
function stats(items) {
  $('stats').innerHTML = items.map(([k, v]) => `<span>${k}<br><b>${v}</b></span>`).join('');
}
// Map the pointer to canvas coordinates regardless of CSS scaling.
function canvasPoint(canvas, ev) {
  const r = canvas.getBoundingClientRect();
  const s = Math.min(r.width / canvas.width, r.height / canvas.height);
  const ox = (r.width - canvas.width * s) / 2, oy = (r.height - canvas.height * s) / 2;
  return [(ev.clientX - r.left - ox) / s, (ev.clientY - r.top - oy) / s];
}
