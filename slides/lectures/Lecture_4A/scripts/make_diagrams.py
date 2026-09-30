"""Editable SVG diagrams for Lecture 4A (no dependencies).

    python3 scripts/make_diagrams.py        # from the Lecture_4A folder
"""
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
INK, MUTED, LIGHT = "#172635", "#617484", "#c3d0da"
ACCENT, GOLD, RED, GREEN, VIOLET = "#087bb9", "#d99700", "#c74842", "#3c8d52", "#7a4fb3"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "Menlo, 'SF Mono', monospace"


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="{FONT}" fill="{INK}">\n'
            '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>\n'
            + "\n".join(body) + "\n</svg>\n")


def text(x, y, s, size=24, anchor="middle", colour=INK, weight="normal", family=None, italic=False):
    f = f' font-family="{family}"' if family else ""
    it = ' font-style="italic"' if italic else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{colour}" font-weight="{weight}"{f}{it}>{s}</text>'


def rect(x, y, w, h, fill="#e9f5fd", stroke=ACCENT, sw=2.5, rx=10, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def line(x1, y1, x2, y2, stroke=INK, sw=2.5, arrow=True, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arr)"' if arrow else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'


def path(d, stroke=INK, sw=2.5, arrow=True, dash=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arr)"' if arrow else ""
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{ds}{m}/>'


def cells(x, y, rows, cols, s, colour, fill_fn=None):
    """A small matrix of squares; fill_fn(i, j) -> opacity in [0, 1] or None for blank."""
    out = []
    for i in range(rows):
        for j in range(cols):
            a = 0.55 if fill_fn is None else fill_fn(i, j)
            if a is None:
                out.append(f'<rect x="{x + j * s:.1f}" y="{y + i * s:.1f}" width="{s - 2}" height="{s - 2}" fill="#eef1f4"/>')
            else:
                out.append(f'<rect x="{x + j * s:.1f}" y="{y + i * s:.1f}" width="{s - 2}" height="{s - 2}" fill="{colour}" fill-opacity="{a:.2f}"/>')
    return out


def write(name, content):
    (FIG / name).write_text(content)


# ------------------------------------------------------------------ the whole model
def pipeline():
    toks = ["The", " American", " flag", " is", " red", ",", " white", ",", " and"]
    ids = [464, 1605, 6056, 318, 2266, 11, 2330, 11, 290]
    B = []
    x0, dx, y0 = 40, 118, 60
    B.append(text(20, 32, "1. text → tokens", 22, "start", ACCENT, "bold"))
    for k, (t, i) in enumerate(zip(toks, ids)):
        x = x0 + k * dx
        B += [rect(x, y0, dx - 10, 44, "#fff1cf", GOLD, 2, 6), text(x + (dx - 10) / 2, y0 + 30, t.replace(" ", "·"), 21, family=MONO),
              text(x + (dx - 10) / 2, y0 + 72, str(i), 18, colour=MUTED, family=MONO)]
    B.append(text(20, 176, "2. token ids → embedding vectors (one row of W_E each, 768 numbers)", 22, "start", ACCENT, "bold"))
    for k in range(9):
        x = x0 + k * dx + (dx - 10) / 2 - 14
        B += cells(x, 192, 6, 1, 26, ACCENT, lambda i, j, k=k: 0.2 + 0.6 * (((k * 7 + i * 3) % 5) / 4))
    B += [rect(x0 - 10, 370, 9 * dx - 5, 140, "#f3f6f8", LIGHT, 2, 12),
          text(x0 + 9 * dx / 2 - 10, 402, "3. twelve transformer blocks, each: attention (tokens exchange information) + MLP (each token alone)", 21, colour=INK, weight="bold"),
          text(x0 + 9 * dx / 2 - 10, 434, "the matrix stays 9 × 768 all the way; each block adds its output to it: the residual stream", 20, colour=MUTED)]
    for k in range(9):
        x = x0 + k * dx + (dx - 10) / 2
        B.append(line(x, 350, x, 370, MUTED, 2))
    for k in range(8):
        x = x0 + k * dx + (dx - 10) / 2
        B.append(f'<circle cx="{x:.1f}" cy="480" r="7" fill="{ACCENT}" fill-opacity="0.35"/>')
        B.append(path(f"M{x + 8:.1f},480 C{x + 40:.1f},465 {x + dx - 40:.1f},465 {x + dx - 8:.1f},480", ACCENT, 1.5, arrow=False))
    xl = x0 + 8 * dx + (dx - 10) / 2
    B.append(f'<circle cx="{xl:.1f}" cy="480" r="9" fill="{GOLD}"/>')
    B.append(text(20, 566, "4. last row × unembedding matrix → 50 257 logits → softmax → probabilities", 22, "start", ACCENT, "bold"))
    probs = [(" blue", 0.93), (" green", 0.01), (" black", 0.01), (" yellow", 0.01)]
    for k, (t, p) in enumerate(probs):
        y = 590 + k * 34
        B += [text(xl - 30, y + 22, t.replace(" ", "·"), 20, "end", family=MONO),
              f'<rect x="{xl - 20:.1f}" y="{y + 5:.1f}" width="{max(3, 200 * p):.1f}" height="22" fill="{ACCENT}"/>',
              text(xl - 10 + max(3, 200 * p), y + 23, f"{p:.2f}", 18, "start", MUTED)]
    B.append(line(xl, 510, xl, 585, INK, 2.5))
    B += [text(20, 770, "5. sample a token, append it to the input, and run again: autoregressive generation", 22, "start", ACCENT, "bold"),
          rect(xl + 210, 592, 110, 44, "#fff1cf", GOLD, 2.5, 6), text(xl + 265, 622, "·blue", 21, family=MONO, weight="bold"),
          path(f"M{xl + 265:.1f},592 C{xl + 265:.1f},0 {x0 + 9 * dx + 60:.1f},20 {x0 + 9 * dx + 5:.1f},70", GOLD, 3, dash="9 6")]
    write("llm-pipeline.svg", svg(1440, 790, B))


# ------------------------------------------------------------------ one transformer block
def block():
    B = []
    cx, top, bot = 330, 40, 820
    B += [line(cx, bot, cx, top + 18, ACCENT, 7),
          text(cx, top, "to the next block", 20, colour=MUTED),
          text(cx, bot + 34, "x (one row per token, 768 numbers each)", 20, colour=MUTED),
          text(cx - 40, 470, "residual", 20, "end", ACCENT, "bold"), text(cx - 40, 496, "stream", 20, "end", ACCENT, "bold")]

    def branch(y_in, y_out, name, sub, colour, fill):
        bx = cx + 120
        out = [path(f"M{cx},{y_in} H{bx + 110}", colour, 2.5, arrow=False),
               line(bx + 110, y_in, bx + 110, y_in - 42, colour, 2.5),
               rect(bx, y_in - 110, 220, 64, "#fff", MUTED, 2, 8), text(bx + 110, y_in - 70, "LayerNorm", 22),
               line(bx + 110, y_in - 112, bx + 110, y_in - 142, colour, 2.5),
               rect(bx - 20, y_in - 242, 260, 96, fill, colour, 3, 10),
               text(bx + 110, y_in - 200, name, 25, weight="bold"), text(bx + 110, y_in - 168, sub, 18, colour=MUTED),
               path(f"M{bx + 110},{y_in - 244} V{y_out} H{cx + 24}", colour, 2.5),
               f'<circle cx="{cx}" cy="{y_out}" r="20" fill="#fff" stroke="{colour}" stroke-width="3"/>',
               text(cx, y_out + 9, "+", 30, weight="bold", colour=colour)]
        return out
    B += branch(790, 470, "Multi-head attention", "tokens exchange information", ACCENT, "#e9f5fd")
    B += branch(430, 120, "MLP", "768 → 3072 → 768, per token", GOLD, "#fff1cf")
    write("transformer-block.svg", svg(760, 880, B))


# ------------------------------------------------------------------ the attention computation with shapes
def attention_shapes():
    B, s = [], 22
    B += [text(80, 36, "X", 30, weight="bold", italic=True), text(80, 62, "T × 768", 18, colour=MUTED)]
    B += cells(20, 80, 7, 6, s, INK, lambda i, j: 0.15 + 0.5 * (((i * 5 + j * 3) % 7) / 6))
    for k, (n, c, y) in enumerate([("Q", ACCENT, 40), ("K", RED, 230), ("V", GREEN, 420)]):
        B += [line(160, 160, 316, y + 120, MUTED, 2),
              text(400, y + 100, n, 30, "start", c, "bold", italic=True),
              text(400, y + 128, f"= X W_{n}", 18, "start", MUTED, family=MONO),
              text(400, y + 154, "T × 64", 17, "start", MUTED)]
        B += cells(320, y + 50, 7, 3, s, c, lambda i, j, k=k: 0.2 + 0.6 * (((i * (k + 2) + j * 5) % 7) / 6))
    # scores
    B += [line(480, 170, 610, 250, ACCENT, 2), line(480, 360, 610, 300, RED, 2),
          text(700, 150, "scores  Q Kᵀ / √64", 24, weight="bold"), text(700, 178, "T × T: every query against every key", 17, colour=MUTED)]
    B += cells(620, 200, 7, 7, s, VIOLET, lambda i, j: 0.15 + 0.7 * (((i * 3 + j * 5) % 7) / 6))
    B += [line(790, 280, 860, 280, INK, 2.5),
          text(1010, 150, "mask + softmax → A", 24, weight="bold"), text(1010, 178, "future hidden; each row sums to 1", 17, colour=MUTED)]
    B += cells(930, 200, 7, 7, s, ACCENT, lambda i, j: None if j > i else (0.9 if j == i - 1 or (i == 0) else 0.25))
    B += [line(1010, 370, 1010, 450, INK, 2.5), line(420, 520, 900, 520, GREEN, 2),
          text(1010, 480, "output = A V", 24, weight="bold"), text(1010, 506, "T × 64: weighted average of values", 17, colour=MUTED)]
    B += cells(975, 525, 7, 3, s, GREEN, lambda i, j: 0.3 + 0.5 * (((i + j) % 4) / 3))
    B += [text(640, 700, "12 heads in parallel (different W_Q, W_K, W_V) → concatenate 12 × 64 = 768 → multiply by W_O", 22, colour=INK),
          text(640, 734, "parameters per layer: 4 × 768² ≈ 2.4 M", 20, colour=MUTED)]
    write("attention-shapes.svg", svg(1280, 760, B))


# ------------------------------------------------------------------ three families
def families():
    B, s = [], 26

    def mask(x, y, kind):
        f = {"full": lambda i, j: 0.6, "causal": lambda i, j: None if j > i else 0.6}[kind]
        return cells(x, y, 6, 6, s, ACCENT, f)
    cols = [(60, "Encoder only", "BERT (2018)", "sees the whole text in both directions", "full", "understanding: classify, embed, search"),
            (520, "Decoder only", "GPT, Llama, Claude (2018 →)", "each token sees only the past", "causal", "generation: next token, chat, code"),
            (980, "Encoder–decoder", "original Transformer (2017), T5", "decoder attends to encoder output", "cross", "translation, speech to text")]
    for x, title, ex, rule, kind, use in cols:
        B += [text(x + 180, 40, title, 28, weight="bold"), text(x + 180, 72, ex, 20, colour=MUTED)]
        if kind == "cross":
            B += cells(x + 20, 110, 6, 6, s, ACCENT, lambda i, j: 0.6)
            B += cells(x + 200, 110, 6, 6, s, GOLD, lambda i, j: None if j > i else 0.6)
            B += [text(x + 98, 290, "encoder: full", 17, colour=MUTED), text(x + 278, 290, "decoder: causal", 17, colour=MUTED),
                  line(x + 175, 190, x + 198, 190, INK, 2)]
        else:
            B += mask(x + 102, 110, kind)
            B.append(text(x + 180, 290, "attention mask: row = query, column = key", 16, colour=MUTED))
        B += [text(x + 180, 336, rule, 20), text(x + 180, 372, use, 20, colour=ACCENT, weight="bold")]
    write("families.svg", svg(1420, 400, B))


# ------------------------------------------------------------------ training targets
def training():
    toks = ["The", "·electron", "·has", "·negative", "·charge", "."]
    B, dx, x0 = [], 190, 170
    B += [text(20, 62, "input", 22, "start", MUTED), text(20, 222, "target", 22, "start", MUTED)]
    for k in range(5):
        x = x0 + k * dx
        B += [rect(x, 30, dx - 20, 48, "#fff1cf", GOLD, 2, 6), text(x + (dx - 20) / 2, 62, toks[k], 21, family=MONO),
              line(x + (dx - 20) / 2, 80, x + (dx - 20) / 2, 118, MUTED, 2),
              rect(x, 120, dx - 20, 44, "#e9f5fd", ACCENT, 2, 6), text(x + (dx - 20) / 2, 149, "p( · | ≤ here)", 18, colour=ACCENT),
              line(x + (dx - 20) / 2, 166, x + (dx - 20) / 2, 190, MUTED, 2),
              rect(x, 192, dx - 20, 48, "#fff", INK, 2, 6), text(x + (dx - 20) / 2, 224, toks[k + 1], 21, family=MONO),
              text(x + (dx - 20) / 2, 272, f"−log p({toks[k + 1]})", 17, colour=RED, family=MONO)]
    B += [text(x0 + 2.5 * dx - 10, 318, "loss = average over all positions: one text of T tokens gives T training examples, computed in one pass", 21, colour=INK)]
    write("training-targets.svg", svg(1150, 340, B))


# ------------------------------------------------------------------ KV cache
def kv_cache():
    B, s, n = [], 34, 7
    B += [text(40, 36, "Generating token 8: only the new row is computed", 24, "start", weight="bold")]
    x0, y0 = 150, 80
    for i in range(n + 1):
        for j in range(n + 1):
            if j > i:
                B.append(f'<rect x="{x0 + j * s}" y="{y0 + i * s}" width="{s - 3}" height="{s - 3}" fill="#eef1f4"/>')
            elif i == n:
                B.append(f'<rect x="{x0 + j * s}" y="{y0 + i * s}" width="{s - 3}" height="{s - 3}" fill="{RED}" fill-opacity="0.75"/>')
            else:
                B.append(f'<rect x="{x0 + j * s}" y="{y0 + i * s}" width="{s - 3}" height="{s - 3}" fill="{ACCENT}" fill-opacity="0.3"/>')
    B += [text(x0 - 12, y0 + n * s + 24, "new query", 18, "end", RED),
          text(x0 + 4 * s, y0 + (n + 1) * s + 34, "keys of all previous tokens", 18, colour=MUTED)]
    kx = 640
    B += [text(kx, 100, "KV cache", 24, "start", weight="bold"),
          text(kx, 132, "keys and values of earlier tokens never change:", 20, "start"),
          text(kx, 160, "store them instead of recomputing them.", 20, "start"),
          text(kx, 214, "cost per new token ∝ tokens so far (not their square)", 20, "start", ACCENT),
          text(kx, 268, "memory per token = 2 × layers × heads × d_head numbers", 20, "start"),
          text(kx, 300, "GPT-2 small: 2 × 12 × 12 × 64 = 18 432 numbers ≈ 37 kB (16-bit)", 20, "start", MUTED),
          text(kx, 332, "DeepSeek-R1 with standard attention: ~4 MB per token", 20, "start", MUTED),
          text(kx, 364, "with multi-head latent attention (MLA): ~70 kB per token", 20, "start", MUTED)]
    write("kv-cache.svg", svg(1300, 420, B))


if __name__ == "__main__":
    pipeline(); block(); attention_shapes(); families(); training(); kv_cache()
    print(sorted(p.name for p in FIG.glob("*.svg")))
