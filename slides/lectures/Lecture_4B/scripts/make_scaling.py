"""Scaling figures for Lecture 4B: METR time horizons and pre-training data sizes.

    ../Lecture_4A/.venv/bin/python scripts/make_scaling.py      # from the Lecture_4B folder

figures/metr-horizons.csv: METR Time Horizon 1.1, 50% horizons in minutes with 95% CI,
copied from the data embedded in https://metr.org/time-horizons/ (page updated 2026-05-08).
"""
import csv
from datetime import date
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"
INK, MUTED, GRID = "#172635", "#617484", "#c3d0da"
GOLD, ACCENT, RED = "#d99700", "#087bb9", "#c74842"
plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "Helvetica Neue", "font.size": 15,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def year(s):
    d = date.fromisoformat(s)
    return d.year + (d.timetuple().tm_yday - 0.5) / 365.25


def metr():
    rows = list(csv.DictReader(open(FIG / "metr-horizons.csv")))
    x = np.array([year(r["release_date"]) for r in rows])
    y = np.array([float(r["p50_minutes"]) for r in rows])
    lo = np.array([float(r["ci_low"]) for r in rows])
    hi = np.array([float(r["ci_high"]) for r in rows])
    front = np.array([r["frontier"] == "1" for r in rows])

    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_yscale("log")
    for val, lab in [(480, "1 work day"), (2400, "1 work week")]:
        ax.axhline(val, color=GRID, lw=1, ls=":")
        ax.text(2018.75, val * 1.12, lab, color=MUTED, fontsize=13)
    ax.axhspan(960, 2e4, color="#f3f5f7", zorder=0)
    ax.text(2018.75, 2e4 / 1.6, "above 16 h: beyond the task suite, unreliable", color=MUTED, fontsize=12)

    # exponential fit to the frontier models: log2(horizon) linear in time
    k, b = np.polyfit(x[front], np.log2(y[front]), 1)
    xs = np.linspace(2019, 2026.6, 50)
    ax.plot(xs, 2 ** (k * xs + b), color=GOLD, lw=2.5, ls="--", zorder=1)
    months = 12 / k
    ax.text(2020.9, 2 ** (k * 2021.6 + b) * 3.2, f"doubling every {months:.0f} months", color="#9c6b00",
            fontsize=15, rotation=np.degrees(np.arctan(k * np.log10(2) * 0.62)), ha="center")

    ax.errorbar(x[~front], y[~front], yerr=[y[~front] - lo[~front], hi[~front] - y[~front]], fmt="o",
                color=GRID, ms=7, lw=1.2, zorder=2)
    ax.errorbar(x[front], y[front], yerr=[y[front] - lo[front], hi[front] - y[front]], fmt="o",
                color=ACCENT, ms=9, lw=1.5, capsize=0, zorder=3)
    show = {"GPT-2": (8, -4), "GPT-3 (davinci-002)": (8, -4), "GPT-3.5": (8, -4), "GPT-4": (-8, 6),
            "GPT-4o": (8, -8), "o1": (-8, 8), "Claude 3.7 Sonnet": (-10, 6), "o3": (-8, 10),
            "Claude Opus 4.5": (-12, 4), "Claude Opus 4.6": (-14, -2), "Claude Mythos Preview": (-12, 14)}
    for r, xi, yi in zip(rows, x, y):
        if r["model"] in show:
            dx, dy = show[r["model"]]
            ax.annotate(r["model"], (xi, yi), textcoords="offset points", xytext=(dx, dy),
                        ha="left" if dx > 0 else "right", fontsize=13, color=INK)
    ax.set_xlim(2018.7, 2026.8); ax.set_ylim(0.01, 2e4)
    ax.set_yticks([0.1, 1, 10, 60, 600, 6000])
    ax.set_yticklabels(["6 s", "1 min", "10 min", "1 h", "10 h", "100 h"])
    ax.set_ylabel("task length (human expert time)\nsucceeded at 50%")
    ax.set_xlabel("model release")
    fig.tight_layout(); fig.savefig(FIG / "metr-horizons.svg", transparent=True); plt.close(fig)
    print(f"doubling time of the frontier fit: {months:.1f} months")


def training_tokens():
    data = [  # (year, tokens, label)  published pre-training token counts
        (2020.4, 3.0e11, "GPT-3"), (2022.2, 1.4e12, "Chinchilla"), (2023.2, 1.4e12, "Llama 1"),
        (2023.55, 2.0e12, "Llama 2"), (2024.3, 1.5e13, "Llama 3"), (2024.97, 1.48e13, "DeepSeek-V3"),
        (2025.3, 3.6e13, "Qwen3"),
    ]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.set_yscale("log")
    x, y = [d[0] for d in data], [d[1] for d in data]
    ax.plot(x, y, "o", color=ACCENT, ms=10)
    for xi, yi, lab in data:
        off = (8, -22) if lab == "DeepSeek-V3" else (-8, 8)
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=off, ha="left" if off[0] > 0 else "right",
                    fontsize=14, color=INK)
    for val, lab in [(5e9, "English Wikipedia ≈ 5 B"), (1.5e13, "FineWeb (2024): 15 T from Common Crawl")]:
        ax.axhline(val, color=GOLD, lw=1.5, ls=":")
        ax.text(2019.6, val * 1.25, lab, color="#9c6b00", fontsize=13)
    ax.set_xlim(2019.5, 2025.9); ax.set_ylim(2e9, 1.5e14)
    ax.set_yticks([1e10, 1e11, 1e12, 1e13, 1e14])
    ax.set_yticklabels(["10 B", "100 B", "1 T", "10 T", "100 T"])
    ax.set_ylabel("pre-training tokens")
    fig.tight_layout(); fig.savefig(FIG / "training-tokens.svg", transparent=True); plt.close(fig)


if __name__ == "__main__":
    metr()
    training_tokens()
