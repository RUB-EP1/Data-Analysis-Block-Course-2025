// Generates the static loss figures in ../figures. Run: node scripts/make_figures.js
const fs = require('fs'), path = require('path');
const OUT = path.join(__dirname, '..', 'figures');
const C = { mse: '#087bb9', mae: '#d97a00', huber: '#c74842', ink: '#172635', grid: '#c3d0da', muted: '#617484', gold: '#d99700', blue: '#253b53' };
function panel({ x0, y0, w, h, xr, yr, title, xlabel, ylabel, curves, xticks, yticks, notes = [], lx = 0 }) {
  const X = (v) => x0 + ((v - xr[0]) / (xr[1] - xr[0])) * w, Y = (v) => y0 + h - ((v - yr[0]) / (yr[1] - yr[0])) * h;
  let s = `<text x="${x0}" y="${y0 - 22}" font-size="34" fill="${C.ink}">${title}</text>`;
  s += `<rect x="${x0}" y="${y0}" width="${w}" height="${h}" fill="none" stroke="${C.grid}"/>`;
  s += `<clipPath id="c${x0}"><rect x="${x0}" y="${y0}" width="${w}" height="${h}"/></clipPath>`;
  for (const t of xticks) s += `<line x1="${X(t)}" x2="${X(t)}" y1="${y0 + h}" y2="${y0 + h + 8}" stroke="${C.muted}"/><text x="${X(t)}" y="${y0 + h + 38}" font-size="26" fill="${C.muted}" text-anchor="middle">${t}</text>`;
  for (const t of yticks) s += `<line x1="${x0 - 8}" x2="${x0}" y1="${Y(t)}" y2="${Y(t)}" stroke="${C.muted}"/><text x="${x0 - 14}" y="${Y(t) + 9}" font-size="26" fill="${C.muted}" text-anchor="end">${t}</text>`;
  if (yr[0] < 0) s += `<line x1="${x0}" x2="${x0 + w}" y1="${Y(0)}" y2="${Y(0)}" stroke="${C.grid}"/>`;
  s += `<text x="${x0 + w / 2}" y="${y0 + h + 80}" font-size="30" fill="${C.ink}" text-anchor="middle">${xlabel}</text>`;
  s += `<text transform="translate(${x0 - 70},${y0 + h / 2}) rotate(-90)" font-size="30" fill="${C.ink}" text-anchor="middle">${ylabel}</text>`;
  s += `<g clip-path="url(#c${x0})">`;
  for (const c of curves) {
    const n = 400, pts = [];
    for (let i = 0; i <= n; i++) { const v = xr[0] + (i / n) * (xr[1] - xr[0]); pts.push(`${X(v).toFixed(1)},${Y(c.f(v)).toFixed(1)}`); }
    s += `<polyline points="${pts.join(' ')}" fill="none" stroke="${c.color}" stroke-width="${c.width || 4.5}" ${c.dash ? `stroke-dasharray="${c.dash}"` : ''}/>`;
  }
  s += '</g>';
  curves.forEach((c, i) => { if (c.label) s += `<line x1="${x0 + lx + 24}" x2="${x0 + lx + 66}" y1="${y0 + 38 + i * 40}" y2="${y0 + 38 + i * 40}" stroke="${c.color}" stroke-width="5" ${c.dash ? `stroke-dasharray="${c.dash}"` : ''}/><text x="${x0 + lx + 78}" y="${y0 + 47 + i * 40}" font-size="27" fill="${C.ink}">${c.label}</text>`; });
  for (const nt of notes) s += `<text x="${X(nt.x)}" y="${Y(nt.y)}" font-size="26" fill="${nt.color || C.muted}" text-anchor="${nt.anchor || 'middle'}">${nt.text}</text>`;
  return s;
}
function svg(w, h, body) { return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" font-family="Helvetica Neue, Arial, sans-serif">${body}</svg>\n`; }

const d = 1;
const huber = (r) => (Math.abs(r) <= d ? 0.5 * r * r : d * (Math.abs(r) - d / 2));
fs.writeFileSync(path.join(OUT, 'huber.svg'), svg(1500, 620,
  panel({ x0: 110, y0: 60, w: 560, h: 440, xr: [-4, 4], yr: [0, 4.5], title: 'Loss versus residual r = y − F', lx: 150, xlabel: 'residual r', ylabel: 'loss', xticks: [-4, -2, -1, 0, 1, 2, 4], yticks: [0, 1, 2, 3, 4],
    curves: [{ f: (r) => 0.5 * r * r, color: C.mse, label: 'squared ½r²' }, { f: Math.abs, color: C.mae, label: 'absolute |r|' }, { f: huber, color: C.huber, label: 'Huber, δ = 1', width: 6 }] }) +
  panel({ x0: 880, y0: 60, w: 560, h: 440, xr: [-4, 4], yr: [-3, 3], title: 'Pseudo-residual −∂L/∂F', xlabel: 'residual r', ylabel: 'pull on the next tree', xticks: [-4, -2, -1, 0, 1, 2, 4], yticks: [-3, -1, 0, 1, 3],
    curves: [{ f: (r) => r, color: C.mse, label: 'squared: r' }, { f: Math.sign, color: C.mae, label: 'absolute: sign r' }, { f: (r) => Math.max(-d, Math.min(d, r)), color: C.huber, label: 'Huber: clip(r, ±δ)', width: 6 }],
    notes: [{ x: 3.9, y: -2.4, text: 'an outlier pulls at most δ', anchor: 'end', color: C.huber }] })));

const sp = (F) => Math.log1p(Math.exp(-Math.abs(F))) + Math.max(F, 0);
fs.writeFileSync(path.join(OUT, 'bce.svg'), svg(760, 640,
  panel({ x0: 110, y0: 60, w: 610, h: 460, xr: [-6, 6], yr: [0, 6], title: 'BCE loss of one event', xlabel: 'score F (log-odds)', ylabel: 'L(y, F)', xticks: [-6, -3, 0, 3, 6], yticks: [0, 2, 4, 6], lx: 150,
    curves: [{ f: (F) => sp(F) - F, color: C.gold, label: 'y = 1: log(1 + e^−F)' }, { f: sp, color: C.blue, label: 'y = 0: log(1 + e^F)' }],
    notes: [{ x: -5.7, y: 0.4, text: 'F = 0: p = ½, L = log 2', anchor: 'start' }] })));
console.log('wrote huber.svg, bce.svg');
