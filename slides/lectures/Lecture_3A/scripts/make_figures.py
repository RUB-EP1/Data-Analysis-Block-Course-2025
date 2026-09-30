"""Figures for Lecture 3A. Needs numpy and matplotlib only.

    python scripts/make_figures.py          # from the Lecture_3A folder

Diagrams are written as plain SVG so their text stays editable; plots use
matplotlib with text kept as text (svg.fonttype = none).
"""
from math import erf, exp, log, sqrt, tanh, pi
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

INK, MUTED, GRID = "#172635", "#617484", "#c3d0da"
GOLD, BLUE, ACCENT, RED = "#d99700", "#253b53", "#087bb9", "#c74842"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "Helvetica Neue", "font.size": 15,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------------------------------------------------------------- plots
def activations():
    x = np.linspace(-5, 5, 801)
    sig = 1 / (1 + np.exp(-x))
    phi = np.array([0.5 * (1 + erf(v / sqrt(2))) for v in x])
    pdf = np.exp(-x ** 2 / 2) / sqrt(2 * pi)
    panels = [
        ("sigmoid σ(z)", sig, sig * (1 - sig), (-0.1, 1.1)),
        ("tanh(z)", np.tanh(x), 1 - np.tanh(x) ** 2, (-1.15, 1.15)),
        ("ReLU(z) = max(0, z)", np.maximum(0, x), (x > 0).astype(float), (-0.4, 3.2)),
        ("GELU(z) = z Φ(z)", x * phi, phi + x * pdf, (-0.4, 3.2)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, (title, f, d, yl) in zip(axes, panels):
        ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
        ax.plot(x, f, color=GOLD, lw=4, label="h(z)")
        ax.plot(x, d, color=INK, lw=2, ls="--", label="h′(z)")
        ax.set_title(title, color=INK, fontsize=17, loc="left")
        ax.set_ylim(*yl); ax.set_xlim(-5, 5); ax.set_xticks([-4, 0, 4])
        ax.set_xlabel("z")
    axes[0].legend(frameon=False, loc="upper left", fontsize=15)
    axes[0].annotate("max slope 1/4", xy=(0, 0.25), xytext=(1.2, 0.15), color=INK, fontsize=14,
                     arrowprops=dict(arrowstyle="->", color=MUTED))
    fig.tight_layout()
    fig.savefig(FIG / "activations.svg", transparent=True)
    plt.close(fig)


def imagenet(path):
    labels = ["2010", "2011", "2012\nAlexNet", "2013\nZFNet", "2014\nVGG", "2014\nGoogLeNet",
              "2015\nResNet", "2016", "2017\nSENet", "human"]
    err = [28.2, 25.8, 16.4, 11.7, 7.3, 6.7, 3.6, 3.0, 2.3, 5.1]
    depth = ["", "", "8 layers", "8", "19", "22", "152", "", "", ""]
    colours = [MUTED, MUTED, ACCENT] + [BLUE] * 6 + ["#b7c3cd"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars = ax.bar(range(len(err)), err, color=colours, width=0.68)
    for i, (b, e, dpt) in enumerate(zip(bars, err, depth)):
        ax.text(i, e + 0.6, f"{e:g}", ha="center", color=INK, fontsize=16, fontweight="bold")
        if dpt:
            ax.text(i, 0.8, dpt, ha="center", color="white", fontsize=12.5, rotation=90 if i > 2 else 0,
                    va="bottom")
    ax.axvspan(-0.5, 1.5, color="#f1f4f7", zorder=-1)
    ax.text(0.5, 31.0, "hand-designed\nfeatures", ha="center", color=MUTED, fontsize=14)
    ax.text(5.5, 31.8, "deep convolutional networks", ha="center", color=ACCENT, fontsize=14)
    ax.set_xticks(range(len(err)), labels, fontsize=13.5)
    ax.set_ylabel("top-5 error (%)")
    ax.set_ylim(0, 35)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(path, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------- SVG helpers
def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}" fill="{INK}">\n'
            '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/>'
            '</marker></defs>\n' + "\n".join(body) + "\n</svg>\n")


def circle(x, y, r, fill="#e9f5fd", stroke=ACCENT, sw=3):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def rect(x, y, w, h, fill="#e9f5fd", stroke=ACCENT, sw=3, rx=10):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def line(x1, y1, x2, y2, stroke=INK, sw=2.5, arrow=True, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arr)"' if arrow else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'


def label(x, y, s, size=28, anchor="middle", colour=INK, weight="normal", italic=False):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{colour}" '
            f'font-weight="{weight}"{st}>{s}</text>')


def edge_label(x1, y1, x2, y2, s, t=0.45, off=-20, **kw):
    """Label a point a fraction t along an edge, shifted perpendicular to it."""
    dx, dy = x2 - x1, y2 - y1
    n = (dx * dx + dy * dy) ** 0.5
    px, py = -dy / n, dx / n
    return label(x1 + t * dx + off * px, y1 + t * dy + off * py + 9, s, **kw)


def write(name, content):
    (FIG / name).write_text(content)


# ---------------------------------------------------------------- diagrams
def neuron():
    b = []
    ys = [110, 230, 410]
    names = ["x<tspan dy='8' font-size='20'>1</tspan>", "x<tspan dy='8' font-size='20'>2</tspan>",
             "x<tspan dy='8' font-size='20'>n</tspan>"]
    ws = ["w<tspan dy='8' font-size='20'>1</tspan>", "w<tspan dy='8' font-size='20'>2</tspan>",
          "w<tspan dy='8' font-size='20'>n</tspan>"]
    for y, n, w in zip(ys, names, ws):
        y2 = 262 + (y - 262) * 0.18
        b += [circle(90, y, 42), label(90, y + 10, n, 32, italic=True),
              line(134, y, 418, y2, sw=3),
              edge_label(134, y, 418, y2, w, t=0.5, off=-24, size=28, colour=ACCENT, italic=True)]
    b += [label(90, 330, "⋮", 40, colour=MUTED)]
    b += [circle(470, 262, 58, fill="#ffe4a3", stroke=GOLD),
          label(470, 280, "Σ", 54, weight="bold"),
          rect(430, 20, 80, 58, fill="#fff", stroke=MUTED, sw=2), label(470, 60, "b", 30, italic=True),
          line(470, 80, 470, 200, stroke=MUTED, dash="8 6"),
          line(530, 262, 640, 262, sw=3), label(585, 245, "z", 30, italic=True),
          rect(642, 212, 130, 100, fill="#fff", stroke=INK, sw=3), label(707, 273, "h(z)", 32, italic=True),
          line(774, 262, 880, 262, sw=3), label(860, 240, "y", 32, italic=True),
          label(90, 500, "inputs", 26, colour=MUTED), label(470, 360, "linear combination", 26, colour=MUTED),
          label(470, 392, "z = w·x + b", 26, colour=MUTED, italic=True),
          label(707, 360, "activation", 26, colour=MUTED)]
    write("neuron.svg", svg(900, 520, b))


def xor_network():
    b = []
    X = [(90, 140, "x<tspan dy='8' font-size='20'>1</tspan>"), (90, 380, "x<tspan dy='8' font-size='20'>2</tspan>")]
    H = [(420, 140, "OR", "−0.5"), (420, 380, "AND", "−1.5")]
    O = (760, 260)
    edges = [(0, 0, "+1"), (1, 0, "+1"), (0, 1, "+1"), (1, 1, "+1")]
    for i, j, w in edges:
        x1, y1, _ = X[i]; x2, y2, _, _ = H[j]
        y2e = y2 + (0 if i == j else (-14 if j == 0 else 14))
        b.append(line(x1 + 44, y1, x2 - 56, y2e, sw=3))
        b.append(edge_label(x1 + 44, y1, x2 - 56, y2e, w, t=0.22, off=-20 if i == j else (20 if i == 0 else -20),
                            size=24, colour=ACCENT, weight="bold"))
    for (x, y, n, bias) in H:
        b += [line(x + 56, y, O[0] - 58, O[1] + (y - O[1]) * 0.25, sw=3),
              circle(x, y, 54, fill="#ffe4a3", stroke=GOLD), label(x, y - 4, n, 26, weight="bold"),
              label(x, y + 26, f"b = {bias}", 20, colour=INK)]
    b += [label(600, 170, "+1", 26, colour=ACCENT, weight="bold"), label(600, 372, "−1", 26, colour=RED, weight="bold")]
    for (x, y, n) in X:
        b += [circle(x, y, 44), label(x, y + 10, n, 32, italic=True)]
    b += [circle(*O, 56, fill="#ffe4a3", stroke=GOLD), label(O[0], O[1] - 4, "XOR", 26, weight="bold"),
          label(O[0], O[1] + 26, "b = −0.5", 20), line(O[0] + 58, O[1], 880, O[1], sw=3),
          label(90, 490, "input", 24, colour=MUTED), label(420, 490, "hidden layer", 24, colour=MUTED),
          label(760, 490, "output", 24, colour=MUTED),
          label(450, 40, "every unit: step(Σ wᵢ xᵢ + b)", 24, colour=MUTED)]
    write("xor-network.svg", svg(900, 510, b))


def mlp():
    b = []
    cols = [(110, 3), (360, 5), (610, 5), (840, 2)]
    pos = [[(x, 260 + (j - (n - 1) / 2) * 92) for j in range(n)] for x, n in cols]
    for l in range(3):
        for j, (x2, y2) in enumerate(pos[l + 1]):
            for (x1, y1) in pos[l]:
                hot = l == 0 and j == 1
                b.append(line(x1 + 30, y1, x2 - 30, y2, stroke=ACCENT if hot else "#b7c3cd",
                              sw=3.5 if hot else 1.6, arrow=False))
    fills = ["#e9f5fd", "#ffe4a3", "#ffe4a3", "#d9e5ef"]
    strokes = [ACCENT, GOLD, GOLD, BLUE]
    for l, layer in enumerate(pos):
        for (x, y) in layer:
            b.append(circle(x, y, 28, fill=fills[l], stroke=strokes[l]))
    b += [label(110, 520, "input x", 25, colour=MUTED),
          label(360, 520, "hidden a⁽¹⁾", 25, colour=MUTED),
          label(610, 520, "hidden a⁽²⁾", 25, colour=MUTED),
          label(840, 520, "output ŷ", 25, colour=MUTED),
          label(235, 36, "one row of W⁽¹⁾", 24, colour=ACCENT)]
    write("mlp.svg", svg(940, 540, b))


def backprop_graph():
    x, w, bb, y = 1.5, 0.8, -0.2, 1
    u = w * x; z = u + bb; a = 1 / (1 + exp(-z)); L = -log(a)
    dz = a - y; du = dz; dw = du * x; dx = du * w; db = dz; da = -1 / a
    B = []
    # leaves
    leaves = [("x", 90, 100, x, dx), ("w", 90, 340, w, dw), ("b", 400, 440, bb, db)]
    for n, X, Y, v, g in leaves:
        B += [circle(X, Y, 40), label(X, Y + 11, n, 32, italic=True),
              label(X, Y - 54, f"{v:g}", 26, colour=ACCENT, weight="bold"),
              label(X, Y + 76, f"{g:+.3f}", 26, colour=RED, weight="bold")]
    ops = [("×", 330, 210), ("+", 560, 290), ("σ", 770, 290), ("−log", 980, 290)]
    for s_, X, Y in ops:
        B += [circle(X, Y, 48, fill="#fff", stroke=INK), label(X, Y + 12, s_, 34 if len(s_) == 1 else 26, weight="bold")]
    B += [line(130, 114, 286, 192), line(130, 326, 286, 232),
          line(376, 226, 514, 276), line(436, 414, 526, 326),
          line(608, 290, 720, 290), line(818, 290, 930, 290), line(1028, 290, 1130, 290)]
    vals = [(456, 222, f"u = {u:g}", f"{du:+.3f}"), (664, 272, f"z = {z:g}", f"{dz:+.3f}"),
            (874, 272, f"a = {a:.3f}", f"{da:+.3f}"), (1080, 272, f"L = {L:.3f}", "1")]
    for X, Y, v, g in vals:
        B += [label(X, Y - 10, v, 25, colour=ACCENT, weight="bold"), label(X, Y + 58, g, 25, colour=RED, weight="bold")]
    B += [label(1080, 370, "label y = 1", 22, colour=MUTED),
          label(700, 470, "forward values", 26, colour=ACCENT, anchor="start", weight="bold"),
          label(700, 506, "backward: ∂L/∂(·)", 26, colour=RED, anchor="start", weight="bold")]
    write("backprop-graph.svg", svg(1160, 550, B))
    return dict(u=u, z=z, a=a, L=L, dz=dz, dw=dw, db=db, dx=dx, da=da)


if __name__ == "__main__":
    activations()
    imagenet(FIG / "imagenet-errors.svg")
    neuron(); xor_network(); mlp()
    print(backprop_graph())
    print("written:", sorted(p.name for p in FIG.iterdir()))
