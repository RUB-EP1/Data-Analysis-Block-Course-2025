"""Measured data and figures for Lecture 4B.

Uses the Lecture 4A environment (GPT-2, tiktoken):

    ../Lecture_4A/.venv/bin/python scripts/make_data.py      # from the Lecture_4B folder

Outputs figures/*.svg and figures/results.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import tiktoken
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
torch.set_grad_enabled(False)
torch.manual_seed(0)
INK, MUTED, GRID = "#172635", "#617484", "#c3d0da"
GOLD, ACCENT, RED = "#d99700", "#087bb9", "#c74842"
plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "Helvetica Neue", "font.size": 15,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})
R = {}


def base_model():
    m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    t = GPT2TokenizerFast.from_pretrained("gpt2")
    out = {}
    for p in ["Can you explain what an electron is?", "Question: What is the capital of France?\nAnswer:"]:
        ids = t(p, return_tensors="pt").input_ids
        g = m.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=50256)
        out[p] = t.decode(g[0, ids.shape[1]:])
    R["base_model"] = out
    return m


def latency(m):
    """Prefill: one forward pass over N prompt tokens. Decode: one new token with the KV cache."""
    torch.set_num_threads(4)
    lengths = [16, 64, 128, 256, 384, 512, 768, 1000]
    pre, dec = [], []
    for n in lengths:
        ids = torch.randint(0, 50000, (1, n))
        m(ids)                                              # warm-up
        times = []
        for _ in range(5):                                  # minimum of five repeats: least disturbed by other load
            t0 = time.perf_counter(); out = m(ids, use_cache=True); times.append(time.perf_counter() - t0)
        pre.append(min(times))
        past, nxt = out.past_key_values, ids[:, -1:]
        times = []
        for _ in range(20):
            t0 = time.perf_counter(); m(nxt, past_key_values=past, use_cache=True); times.append(time.perf_counter() - t0)
        dec.append(min(times))
    R["latency"] = {"lengths": lengths, "prefill_s": pre, "decode_s_per_token": dec,
                    "prefill_tok_per_s": [n / p for n, p in zip(lengths, pre)], "decode_tok_per_s": [1 / d for d in dec]}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(lengths, np.array(pre) * 1000, "o-", color=ACCENT, lw=3, label="prefill: whole prompt, one pass")
    ax[0].plot(lengths, np.array(lengths) * np.array(dec) * 1000, "s--", color=RED, lw=2.5,
               label="generating the same number of tokens")
    ax[0].set_xlabel("tokens"); ax[0].set_ylabel("time (ms)"); ax[0].set_yscale("log")
    ax[0].legend(frameon=False, fontsize=13)
    ax[0].set_title("Reading is parallel, writing is sequential", loc="left", color=INK)
    ax[1].bar(["prefill\n(1000-token prompt)", "decode\n(one token at a time)"],
              [lengths[-1] / pre[-1], 1 / dec[-1]], color=[ACCENT, RED], width=0.5)
    for i, v in enumerate([lengths[-1] / pre[-1], 1 / dec[-1]]):
        ax[1].text(i, v * 1.03, f"{v:,.0f} tokens/s", ha="center", fontsize=14, color=INK)
    ax[1].set_title("Throughput, GPT-2 small on a laptop CPU", loc="left", color=INK)
    ax[1].set_ylim(0, max(lengths[-1] / pre[-1], 1 / dec[-1]) * 1.2)
    ax[1].set_ylabel("tokens per second")
    ax[1].tick_params(axis="x", length=0)
    fig.tight_layout(); fig.savefig(FIG / "prefill-decode.svg", transparent=True); plt.close(fig)


def context_windows():
    models = [  # (year as decimal, tokens, label)
        (2019.1, 1024, "GPT-2"), (2020.4, 2048, "GPT-3"), (2022.9, 4096, "ChatGPT (GPT-3.5)"),
        (2023.2, 32768, "GPT-4 (32k)"), (2023.4, 100_000, "Claude (100k)"), (2023.9, 200_000, "Claude 2.1"),
        (2024.1, 1_000_000, "Gemini 1.5 Pro"), (2026.3, 1_000_000, "Claude Opus 5, Sonnet 5"),
    ]
    enc = tiktoken.get_encoding("o200k_base")
    qmd = "\n".join(f.read_text() for f in sorted((HERE.parent).glob("Lecture_*/Lecture_*.qmd")))
    refs = {"all slides of this course": len(enc.encode(qmd, disallowed_special=()))}
    R["context_windows"] = models
    R["token_references"] = refs
    fig, ax = plt.subplots(figsize=(12, 5.4))
    x, y = [m[0] for m in models], [m[1] for m in models]
    ax.plot(x, y, "o-", color=ACCENT, lw=3, ms=9)
    for xi, yi, lab in models:
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(-8, 10), ha="right", fontsize=13, color=INK)
    ax.set_yscale("log"); ax.set_ylim(500, 5e6); ax.set_xlim(2018.6, 2026.8)
    ax.set_ylabel("context window (tokens)")
    for val, lab in [(refs["all slides of this course"], "all slides of this course"), (100_000, "a 300-page book")]:
        ax.axhline(val, color=GOLD, lw=1.5, ls=":")
        ax.text(2018.7, val * 1.12, f"{lab} ≈ {val / 1000:.0f} k tokens", color="#9c6b00", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "context-windows.svg", transparent=True); plt.close(fig)


if __name__ == "__main__":
    m = base_model()
    latency(m)
    context_windows()
    (FIG / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(json.dumps({k: R[k] for k in ["base_model", "token_references"]}, indent=1, ensure_ascii=False))
    L = R["latency"]
    print("prefill tok/s", [round(v) for v in L["prefill_tok_per_s"]]); print("decode tok/s", [round(v) for v in L["decode_tok_per_s"]])
