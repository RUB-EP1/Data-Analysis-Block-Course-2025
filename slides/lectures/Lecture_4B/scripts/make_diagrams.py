"""Editable SVG diagrams for Lecture 4B (no dependencies).

    python3 scripts/make_diagrams.py        # from the Lecture_4B folder
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


def text(x, y, s, size=24, anchor="middle", colour=INK, weight="normal", family=None):
    f = f' font-family="{family}"' if family else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{colour}" font-weight="{weight}"{f}>{s}</text>'


def rect(x, y, w, h, fill="#e9f5fd", stroke=ACCENT, sw=2.5, rx=12, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def path(d, stroke=INK, sw=2.5, arrow=True, dash=None):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arr)"' if arrow else ""
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"{ds}{m}/>'


def box(x, y, w, h, title, lines, colour, fill, size=22):
    out = [rect(x, y, w, h, fill, colour, 3), text(x + w / 2, y + 36, title, 25, colour=colour, weight="bold")]
    for k, l in enumerate(lines):
        out.append(text(x + w / 2, y + 72 + k * 30, l, size, colour=INK))
    return out


def stages():
    B = []
    data = [
        ("1. Pre-training", ["predict the next token", "on trillions of tokens", "of web text, books, code"], ACCENT, "#e9f5fd", "base model: continues text"),
        ("2. Instruction tuning", ["imitate written answers", "user → assistant pairs,", "10⁴ – 10⁶ examples"], GOLD, "#fff1cf", "follows the chat format"),
        ("3. Preference learning", ["people rank answers;", "RL towards preferred ones", "(RLHF, 2022)"], VIOLET, "#f1eafa", "helpful, polite, careful"),
        ("4. Verifiable rewards", ["reward only if the answer", "is correct: math, code tests", "(o1 2024, DeepSeek-R1 2025)"], GREEN, "#e6f3e8", "learns to think longer"),
    ]
    w, gap = 330, 46
    for k, (t, lines, c, f, result) in enumerate(data):
        x = 20 + k * (w + gap)
        B += box(x, 30, w, 190, t, lines, c, f)
        B.append(text(x + w / 2, 262, result, 21, colour=MUTED))
        if k < 3:
            B.append(path(f"M{x + w + 4},125 H{x + w + gap - 4}", INK, 3))
    B.append(text(20 + 0.5 * w, 300, "most of the compute", 19, colour=ACCENT, weight="bold"))
    B.append(text(20 + 2.5 * w + 2 * gap, 300, "small data, large change in behaviour", 19, colour=VIOLET, weight="bold"))
    (FIG / "training-stages.svg").write_text(svg(1520, 320, B))


def agent_loop():
    B = []
    B += box(40, 160, 330, 210, "Context", ["system prompt, tools,", "memory files,", "conversation so far,", "tool results"], ACCENT, "#e9f5fd", 21)
    B += box(510, 40, 330, 150, "Model", ["reads everything,", "writes text or a tool call"], VIOLET, "#f1eafa", 21)
    B += box(980, 160, 330, 210, "Harness", ["checks permissions,", "runs the tool (sandbox),", "validates: lint, tests,", "adds reminders"], GOLD, "#fff1cf", 21)
    B += box(510, 400, 330, 130, "Tool result", ["output, error, or", "“user declined”"], MUTED, "#f3f6f8", 21)
    B += [path("M370,210 C430,150 450,120 506,115", INK, 3), text(420, 128, "call", 20, colour=MUTED),
          path("M842,115 C900,120 930,150 976,210", INK, 3), text(935, 128, "tool call", 20, colour=MUTED),
          path("M1140,372 C1120,440 920,470 844,465", INK, 3), text(1050, 470, "executes", 20, colour=MUTED),
          path("M508,465 C420,470 250,440 205,374", INK, 3), text(300, 470, "appended", 20, colour=MUTED),
          path("M675,40 C675,0 1400,0 1440,90", GREEN, 3, dash="9 6"),
          text(1545, 118, "no tool call:", 21, "end", GREEN, "bold"), text(1545, 146, "answer to the user", 21, "end", GREEN)]
    (FIG / "agent-loop.svg").write_text(svg(1560, 560, B))


def harness():
    B = []
    cx, W, H = 720, 330, 140
    B += [rect(cx - 170, 305, 340, 150, "#f1eafa", VIOLET, 4, 18),
          text(cx, 372, "Model", 38, colour=VIOLET, weight="bold"), text(cx, 412, "text in → text out", 22, colour=MUTED)]
    parts = [
        (cx - W / 2, 30, "Permissions &amp; sandbox", ["ask / allow / deny,", "no secrets, no rm -rf"], MUTED, "#f3f6f8"),
        (40, 185, "Instructions &amp; memory", ["system prompt,", "CLAUDE.md, AGENTS.md, skills"], ACCENT, "#e9f5fd"),
        (1070, 185, "Tools", ["read, grep, edit, bash,", "web search, MCP servers"], GOLD, "#fff1cf"),
        (40, 455, "Context management", ["compaction, subagents,", "prompt caching"], GREEN, "#e6f3e8"),
        (1070, 455, "Validation", ["linters, tests, builds,", "screenshots"], RED, "#fbeceb"),
        (cx - W / 2, 610, "Interface", ["chat, CLI, desktop,", "IDE, cloud"], INK, "#eef1f4"),
    ]
    for x, y, t, lines, c, f in parts:
        B.insert(0, path(f"M{x + W / 2:.0f},{y + H / 2:.0f} L{cx},{380}", LIGHT, 3, arrow=False))
        B += box(x, y, W, H, t, lines, c, f, 21)
    B.insert(0, rect(15, 10, 1410, 755, "#fcfdfe", LIGHT, 2, 24, "10 8"))
    B.append(text(1405, 44, "harness", 26, "end", MUTED, "bold"))
    (FIG / "harness.svg").write_text(svg(1440, 775, B))


def rl_loop():
    B = []
    B += box(20, 150, 250, 150, "Prompt", ["a math problem,", "a coding task"], ACCENT, "#e9f5fd", 21)
    B += box(340, 150, 270, 150, "Model = policy", ["samples several", "answers"], VIOLET, "#f1eafa", 21)
    B += box(1110, 150, 290, 150, "Reward", ["program: correct? tests?", "or a reward model"], GREEN, "#e6f3e8", 21)
    answers = [("… so x = 42", "1"), ("… so x = 17", "0"), ("… wait, x = 42", "1"), ("… x = 40 + 2i", "0")]
    for k, (a, r) in enumerate(answers):
        y = 30 + k * 98
        B += [rect(690, y, 330, 70, "#f3f6f8", MUTED, 2), text(855, y + 44, a, 22, family=MONO)]
        B.append(path(f"M612,225 C650,225 650,{y + 35} 686,{y + 35}", LIGHT, 2.5))
        B.append(path(f"M1022,{y + 35} C1070,{y + 35} 1060,225 1106,225", LIGHT, 2.5))
        B.append(text(1440, y + 44, f"r = {r}", 23, "start", GREEN if r == "1" else RED, "bold"))
    B.append(path("M272,225 H336", INK, 3))
    B.append(path("M1255,302 C1240,470 520,470 475,304", GREEN, 3, dash="9 6"))
    B.append(text(865, 455, "update the weights: answers with reward above the group average become more likely", 22, colour=GREEN, weight="bold"))
    (FIG / "rl-loop.svg").write_text(svg(1560, 480, B))


if __name__ == "__main__":
    stages(); agent_loop(); harness(); rl_loop()
    print(sorted(p.name for p in FIG.glob("*.svg")))
