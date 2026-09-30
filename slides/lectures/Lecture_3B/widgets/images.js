'use strict';
// 28 × 28 test images (value 1 = ink, 0 = paper), drawn procedurally so no data files are needed.
const N28 = 28;

function segDist(px, py, [ax, ay], [bx, by]) {
  const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
  const t = L ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L)) : 0;
  return Math.hypot(px - ax - t * dx, py - ay - t * dy);
}
function strokes(paths, r = 1.25) {
  const im = new Float32Array(N28 * N28);
  for (let y = 0; y < N28; y++) for (let x = 0; x < N28; x++) {
    let d = Infinity;
    paths.forEach(p => { for (let i = 1; i < p.length; i++) d = Math.min(d, segDist(x, y, p[i - 1], p[i])); });
    im[y * N28 + x] = Math.max(0, Math.min(1, r + 0.5 - d));
  }
  return im;
}
function filled(inside) {
  const im = new Float32Array(N28 * N28);
  for (let y = 0; y < N28; y++) for (let x = 0; x < N28; x++) {
    // 4 × 4 supersampling for soft edges
    let s = 0;
    for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) s += inside(x - .375 + i * .25, y - .375 + j * .25) ? 1 : 0;
    im[y * N28 + x] = s / 16;
  }
  return im;
}
const IMAGES = {
  'digit 7': () => strokes([[[7, 6], [21, 6], [13, 23]], [[11, 14], [19, 14]]]),
  'digit 3': () => strokes([[[8, 6], [18, 6], [20, 9], [18, 13], [12, 13]], [[18, 13], [21, 17], [19, 21], [8, 22]]]),
  'square': () => filled((x, y) => x >= 7 && x <= 20 && y >= 7 && y <= 20),
  'triangle': () => filled((x, y) => y <= 22 && y >= 5 + 17 * Math.abs(x - 13.5) / 9.5 * 1.0 && Math.abs(x - 13.5) <= 9.5),
  'outline square': () => strokes([[[7, 7], [20, 7], [20, 20], [7, 20], [7, 7]]], 0.9),
  'plus': () => strokes([[[14, 5], [14, 22]], [[5, 14], [22, 14]]], 1.3),
  'ring': () => filled((x, y) => { const r = Math.hypot(x - 13.5, y - 13.5); return r >= 6 && r <= 9; }),
  'stripes': () => filled((x, y) => Math.floor((x + y) / 3) % 2 === 0),
  'blank': () => new Float32Array(N28 * N28)
};

// Paint on a 28 × 28 image with the mouse. `grid` maps canvas pixels to image cells.
function paintable(canvas, grid, image, redraw) {
  let down = false, erase = false;
  const paint = ev => {
    const [px, py] = canvasPoint(canvas, ev);
    const cx = (px - grid.x) / grid.cell, cy = (py - grid.y) / grid.cell;
    if (cx < -1 || cy < -1 || cx > N28 + 1 || cy > N28 + 1) return false;
    for (let y = Math.floor(cy) - 2; y <= Math.floor(cy) + 2; y++) for (let x = Math.floor(cx) - 2; x <= Math.floor(cx) + 2; x++) {
      if (x < 0 || y < 0 || x >= N28 || y >= N28) continue;
      const d = Math.hypot(x + .5 - cx, y + .5 - cy), v = Math.max(0, Math.min(1, 1.6 - d));
      const i = y * N28 + x, im = image();
      im[i] = erase ? Math.min(im[i], 1 - v) : Math.max(im[i], v);
    }
    redraw(); return true;
  };
  canvas.addEventListener('pointerdown', ev => {
    const [px, py] = canvasPoint(canvas, ev);
    if (px < grid.x || py < grid.y || px > grid.x + grid.cell * N28 || py > grid.y + grid.cell * N28) return;
    down = true; erase = ev.shiftKey || ev.button === 2; canvas.setPointerCapture(ev.pointerId); paint(ev);
  });
  canvas.addEventListener('pointermove', ev => { if (down) paint(ev); });
  canvas.addEventListener('pointerup', () => { down = false; });
  canvas.addEventListener('contextmenu', ev => ev.preventDefault());
}
