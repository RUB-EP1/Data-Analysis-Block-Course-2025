"""Editable SVG diagrams for Lecture 3B (no dependencies).

    python3 scripts/make_diagrams.py        # from the Lecture_3B folder
"""
from math import log2, sqrt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
INK, MUTED = "#172635", "#617484"
ACCENT, GOLD, RED = "#087bb9", "#d99700", "#c74842"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
KIND = {  # face, top, side, stroke
    "input": ("#e6ebf0", "#f3f6f8", "#d3dbe3", "#7d8f9e"),
    "conv": ("#cfe6f7", "#e7f3fc", "#b5d7ef", ACCENT),
    "pool": ("#f7d6d4", "#fbe9e8", "#efc0bd", RED),
    "fc": ("#d8efd9", "#ecf8ec", "#c0e3c2", "#3c8d52"),
    "kernel": ("#ffe4a3", "#fff1cf", "#ffd46e", GOLD),
}


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}" fill="{INK}">\n'
            '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>\n'
            + "\n".join(body) + "\n</svg>\n")


def text(x, y, s, size=22, anchor="middle", colour=INK, weight="normal", italic=False):
    it = ' font-style="italic"' if italic else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{colour}" font-weight="{weight}"{it}>{s}</text>'


def poly(pts, fill, stroke, sw=1.8, extra=""):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"{extra}/>'


def prism(x, y, t, s, d, kind="conv", sw=1.8):
    """A box with front face t wide and s tall at (x, y) (top-left), extruded by d up and to the right."""
    face, top, side, stroke = KIND[kind]
    return [poly([(x, y), (x + d, y - d), (x + t + d, y - d), (x + t, y)], top, stroke, sw),
            poly([(x + t, y), (x + t + d, y - d), (x + t + d, y + s - d), (x + t, y + s)], side, stroke, sw),
            poly([(x, y), (x + t, y), (x + t, y + s), (x, y + s)], face, stroke, sw)]


def line(x1, y1, x2, y2, stroke=INK, sw=2, dash=None, arrow=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arr)"' if arrow else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'


def alexnet():
    # name, spatial, channels, kind, operation label
    L = [("input", 224, 3, "input", "RGB image"),
         ("conv1", 55, 64, "conv", "11×11, stride 4"),
         ("pool", 27, 64, "pool", "3×3, stride 2"),
         ("conv2", 27, 192, "conv", "5×5"),
         ("pool", 13, 192, "pool", ""),
         ("conv3", 13, 384, "conv", "3×3"),
         ("conv4", 13, 256, "conv", "3×3"),
         ("conv5", 13, 256, "conv", "3×3"),
         ("pool", 6, 256, "pool", ""),
         ("fc6", 1, 4096, "fc", ""),
         ("fc7", 1, 4096, "fc", ""),
         ("fc8", 1, 1000, "fc", "softmax")]
    B, x, base = [], 30, 300
    for name, sp, ch, kind, op in L:
        if kind == "fc":
            t, s, d = 16, 60 + 26 * log2(ch / 500), 10
        else:
            s = 13 * sqrt(sp); t = 6 + 8 * log2(ch / 3 + 1); d = 0.5 * s
        y = base - s / 2 + d / 2
        B += prism(x, y, t, s, d, kind)
        cx = x + (t + d) / 2
        B.append(text(cx, base - 175, name, 22, weight="bold", colour=KIND[kind][3]))
        dims = f"{sp}×{sp}×{ch}" if kind != "fc" else f"{ch}"
        B.append(text(cx, base + 165, dims, 19, colour=INK))
        if op:
            B.append(text(cx, base + 192, op, 16, colour=MUTED))
        x += t + d + (38 if kind != "fc" else 34)
    # braces
    B += [line(30, 520, 1200, 520, MUTED, 2), text(615, 552, "5 convolutional layers (+ ReLU): features", 22, colour=MUTED),
          line(1290, 520, x - 20, 520, MUTED, 2), text((1290 + x - 20) / 2, 552, "3 dense layers: classifier", 22, colour=MUTED)]
    B += [text(x + 40, base + 8, "→ 1000 class", 22, anchor="start"), text(x + 40, base + 36, "probabilities", 22, anchor="start")]
    (FIG / "alexnet-architecture.svg").write_text(svg(x + 200, 575, B))


def conv_layer():
    B = []
    # input tensor: H × W × C_in, front face H × W, depth C_in to the back-right
    x0, y0, s, t, d = 60, 190, 300, 70, 110
    B += prism(x0, y0, s, s, d, "input")
    B += [text(x0 + s / 2, y0 + s + 42, "input  H × W × C<tspan dy='6' font-size='16'>in</tspan>", 24),
          text(x0 + s / 2, y0 - d - 22, "e.g. 224 × 224 × 3", 20, colour=MUTED)]
    # kernel sitting on the input
    kx, ky, ks, kd = x0 + 150, y0 + 70, 60, 110
    B += prism(kx, ky, ks, ks, kd, "kernel", 2.5)
    B.append(text(kx + 30, ky + ks + 28, "K × K × C<tspan dy='6' font-size='14'>in</tspan>", 19, colour="#9c6b00", weight="bold"))
    # output: stack of maps
    ox, oy, os_ = 760, 230, 230
    n = 7
    for i in range(n):
        dx = i * 22
        kind = "conv" if i != 3 else "kernel"
        B += prism(ox + dx, oy - dx, os_, os_, 0.1, kind, 1.6 if i != 3 else 2.5)
    # dot product arrow from kernel to one output cell in the highlighted map
    cx, cy = ox + 3 * 22 + 130, oy - 3 * 22 + 90
    B += [f'<rect x="{cx - 9}" y="{cy - 9}" width="18" height="18" fill="{GOLD}" stroke="{INK}" stroke-width="1.5"/>',
          line(kx + ks + kd * 0.5, ky - kd * 0.5 + 10, cx - 12, cy - 4, INK, 2.5, "7 5", True),
          text(560, 150, "dot product = one number", 22, colour=INK),
          text(560, 178, "(similarity of patch and kernel)", 19, colour=MUTED)]
    B += [text(ox + 190, oy + os_ + 60, "output  H′ × W′ × C<tspan dy='6' font-size='16'>out</tspan>", 24),
          text(ox + 190, oy + os_ + 92, "one activation map per kernel", 20, colour=MUTED),
          text(ox + 190, 60, "slide the kernel over all positions →", 20, colour=MUTED),
          text(ox + 190, 86, "one activation map", 20, colour=MUTED)]
    (FIG / "conv-layer.svg").write_text(svg(1140, 590, B))


if __name__ == "__main__":
    alexnet()
    conv_layer()
    print("written alexnet-architecture.svg, conv-layer.svg")
