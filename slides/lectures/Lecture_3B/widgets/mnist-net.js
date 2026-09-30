'use strict';
// Inference for the small MNIST CNN trained in scripts/make_mnist_data.jl.
//   28×28 → conv 3×3 (8) + ReLU → pool 2×2 → conv 3×3 (16) + ReLU → pool 2×2 → dense 400 → 10
// Images are Float32Array(784) in [0, 1], row-major: index y·28 + x.
// Also loads in Node (module.exports) for scripts/check_mnist_widget.mjs.

function b64bytes(s) {
  if (typeof atob === 'function') return Uint8Array.from(atob(s), c => c.charCodeAt(0));
  return new Uint8Array(Buffer.from(s, 'base64'));
}

// Split the flat weight vector of one checkpoint into its layers.
function mnistWeights(b64) {
  const f = new Float32Array(b64bytes(b64).buffer);
  let o = 0;
  const take = n => f.subarray(o, o += n);
  return { w1: take(8 * 9), b1: take(8), w2: take(16 * 8 * 9), b2: take(16), wd: take(10 * 400), bd: take(10) };
}

// Valid 3×3 convolution + ReLU. inp: nIn maps of n×n; w: [co][ci][ky][kx].
function conv3(inp, nIn, n, w, b, nOut) {
  const m = n - 2, out = new Float32Array(nOut * m * m);
  for (let co = 0; co < nOut; co++) for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) {
    let s = b[co];
    for (let ci = 0; ci < nIn; ci++) {
      const W = (co * nIn + ci) * 9, I = ci * n * n;
      for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) s += w[W + ky * 3 + kx] * inp[I + (y + ky) * n + x + kx];
    }
    out[co * m * m + y * m + x] = s > 0 ? s : 0;
  }
  return out;
}

// 2×2 max-pool with stride 2 (odd sizes drop the last row/column, as in Flux).
function pool2(inp, c, n) {
  const m = n >> 1, out = new Float32Array(c * m * m);
  for (let k = 0; k < c; k++) for (let y = 0; y < m; y++) for (let x = 0; x < m; x++) {
    const I = k * n * n + 2 * y * n + 2 * x;
    out[k * m * m + y * m + x] = Math.max(inp[I], inp[I + 1], inp[I + n], inp[I + n + 1]);
  }
  return out;
}

// All intermediate maps, for drawing, plus logits and probabilities.
function mnistForward(W, img) {
  const a1 = conv3(img, 1, 28, W.w1, W.b1, 8);   // 8 × 26 × 26
  const p1 = pool2(a1, 8, 26);                    // 8 × 13 × 13
  const a2 = conv3(p1, 8, 13, W.w2, W.b2, 16);   // 16 × 11 × 11
  const p2 = pool2(a2, 16, 11);                   // 16 × 5 × 5 = 400
  const logits = new Float32Array(10);
  for (let o = 0; o < 10; o++) {
    let s = W.bd[o];
    for (let i = 0; i < 400; i++) s += W.wd[o * 400 + i] * p2[i];
    logits[o] = s;
  }
  const mx = Math.max(...logits), e = logits.map(v => Math.exp(v - mx)), z = e.reduce((a, b) => a + b, 0);
  return { a1, p1, a2, p2, logits, probs: e.map(v => v / z) };
}

// Move an image dx pixels to the right, filling with zeros.
function shiftRight(img, dx) {
  const out = new Float32Array(784);
  for (let y = 0; y < 28; y++) for (let x = dx; x < 28; x++) if (x - dx >= 0) out[y * 28 + x] = img[y * 28 + x - dx];
  return out;
}

// MNIST preprocessing of a drawing (canvas alpha): crop to the bounding box, scale into
// 20 × 20, then centre by the centre of mass at (13.5, 13.5). Same as the Pluto DrawPad.
function mnistPreprocess(canvas) {
  const W = canvas.width, H = canvas.height, img = canvas.getContext('2d').getImageData(0, 0, W, H).data;
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (img[4 * (y * W + x) + 3] > 20) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 < 0) return null;
  const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h);
  const t = document.createElement('canvas'); t.width = t.height = 28;
  const tc = t.getContext('2d'); tc.imageSmoothingQuality = 'high';
  tc.drawImage(canvas, x0, y0, w, h, 14 - w * s / 2, 14 - h * s / 2, w * s, h * s);
  let d = tc.getImageData(0, 0, 28, 28).data, m = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) { const v = d[4 * i + 3]; m += v; cx += v * (i % 28); cy += v * Math.floor(i / 28); }
  const f = document.createElement('canvas'); f.width = f.height = 28;
  f.getContext('2d').drawImage(t, Math.round(13.5 - cx / m), Math.round(13.5 - cy / m));
  d = f.getContext('2d').getImageData(0, 0, 28, 28).data;
  return Float32Array.from({ length: 784 }, (_, i) => d[4 * i + 3] / 255);
}

if (typeof module !== 'undefined') module.exports = { mnistWeights, mnistForward, shiftRight, b64bytes };
