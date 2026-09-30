"""Data and figures for Lecture 4A, computed with GPT-2 small (124 M parameters).

    .venv/bin/python scripts/make_data.py          # from the Lecture_4A folder

The first run downloads GPT-2 (~550 MB) from the Hugging Face hub and the
tiktoken vocabularies. Everything else is deterministic.

Outputs
  widgets/data_attention.js   attention patterns of all 144 heads for a few sentences
  widgets/data_embed.js       word embeddings (int8-quantized) with a 2D t-SNE map
  widgets/data_sampler.js     next-token logits and sampled continuations
  figures/*.svg               tokenizer comparison, logit lens, induction loss, attention grid
  figures/results.json        numbers quoted on the slides
"""
import base64
import json
from pathlib import Path

import numpy as np
import torch
import tiktoken
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).resolve().parent.parent
FIG, WID = HERE / "figures", HERE / "widgets"
FIG.mkdir(exist_ok=True)
torch.set_grad_enabled(False)
torch.manual_seed(0)

INK, MUTED, GRID = "#172635", "#617484", "#c3d0da"
GOLD, BLUE, ACCENT, RED, GREEN = "#d99700", "#253b53", "#087bb9", "#c74842", "#3c8d52"
plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "Helvetica Neue", "font.size": 15,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

MODEL = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
TOK = GPT2TokenizerFast.from_pretrained("gpt2")
BOS = TOK.eos_token_id          # GPT-2 uses <|endoftext|> as its only special token
RESULTS = {}


def show(tok):
    return tok.replace("\n", "↵")


def js(name, obj):
    (WID / name).write_text(f"window.{name.split('.')[0].upper()}=" + json.dumps(obj, separators=(",", ":")) + ";\n")


# ------------------------------------------------------------------ model facts
def parameters():
    d, L, V, N = 768, 12, 50257, 1024
    parts = {
        "token embedding (50257 × 768)": V * d,
        "position embedding (1024 × 768)": N * d,
        "attention, 12 layers × (4·768² + biases)": L * (4 * d * d + 4 * d),
        "MLP, 12 layers × (2·768·3072 + biases)": L * (2 * d * 4 * d + 4 * d + d),
        "layer norms": (2 * L + 1) * 2 * d,
    }
    total = sum(p.numel() for p in MODEL.parameters())
    assert total == sum(parts.values()), (total, sum(parts.values()))
    RESULTS["gpt2_parameters"] = {"total": total, **parts}


# ------------------------------------------------------------------ tokenizers
def tokenizers():
    texts = {
        "English": "The electron and the proton attract each other, and the energy of the atom is quantized.",
        "German": "Das Elektron und das Proton ziehen sich an, und die Energie des Atoms ist quantisiert.",
        "Russian": "Электрон и протон притягиваются друг к другу, а энергия атома квантована.",
        "Chinese": "电子和质子相互吸引，原子的能量是量子化的。",
        "Python": "def energy(mass, p):\n    return (mass**2 + p**2) ** 0.5\n",
    }
    encs = [("GPT-2 (2019), 50 257", "gpt2"), ("GPT-4 (2023), 100 277", "cl100k_base"), ("GPT-4o (2024), 200 019", "o200k_base")]
    counts = {lab: [len(tiktoken.get_encoding(e).encode(s)) for s in texts.values()] for lab, e in encs}
    RESULTS["tokens_per_sentence"] = {"texts": texts, "counts": counts}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.26
    for k, ((lab, _), c) in enumerate(zip(encs, [MUTED, ACCENT, GOLD])):
        vals = counts[lab]
        bars = ax.bar(np.arange(len(texts)) + (k - 1) * w, vals, w, color=c, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", fontsize=12, color=INK)
    ax.set_xticks(range(len(texts)), list(texts), fontsize=15)
    ax.set_ylabel("tokens for the same sentence")
    ax.legend(frameon=False, fontsize=13, title="tokenizer, vocabulary size", title_fontsize=13)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout(); fig.savefig(FIG / "tokenizers.svg", transparent=True); plt.close(fig)

    enc2, enc4 = tiktoken.get_encoding("gpt2"), tiktoken.get_encoding("o200k_base")
    RESULTS["token_examples"] = {
        s: {"gpt2": [enc2.decode([i]) for i in enc2.encode(s)], "o200k": [enc4.decode([i]) for i in enc4.encode(s)]}
        for s in [" strawberry", "strawberry", " Bochum", " Heisenberg", " Energieerhaltungssatz",
                  " 12345678", " 2026", " SolidGoldMagikarp", " transformer", " Transformer"]}


# ------------------------------------------------------------------ attention patterns
def attention():
    sentences = [
        ("flag", "The American flag is red, white, and"),
        ("names", "When Mary and John went to the store, John gave a drink to"),
        ("repeat", " quantum lamp river seven tiger cloud violin" * 2),
        ("physics", "The electron was accelerated in the magnetic field, and the electron"),
    ]
    out = []
    for key, s in sentences:
        ids = [BOS] + TOK.encode(s)
        res = MODEL(torch.tensor([ids]), output_attentions=True)
        A = torch.stack(res.attentions)[:, 0]                     # 12 × 12 × T × T
        p = res.logits[0].softmax(-1)
        nxt = [TOK.decode([int(i)]) for i in p[-1].topk(5).indices]
        loss = [None] + [float(-torch.log(p[i - 1, ids[i]])) for i in range(1, len(ids))]
        toks = ["⟨bos⟩"] + [show(TOK.decode([i])) for i in ids[1:]]
        # store the lower triangle only, rounded to 1/200
        pats = [[[[int(round(float(A[l, h, i, j]) * 200)) for j in range(i + 1)] for i in range(len(ids))]
                 for h in range(12)] for l in range(12)]
        out.append({"key": key, "text": s, "tokens": toks, "patterns": pats,
                    "next": nxt, "nextp": [round(float(v), 3) for v in p[-1].topk(5).values],
                    "loss": [None if v is None else round(v, 2) for v in loss]})
        RESULTS[f"attention_{key}"] = {"next": list(zip(nxt, out[-1]["nextp"]))}
    notes = {"4.11": "previous token", "5.5": "induction", "6.9": "induction", "5.1": "induction",
             "7.10": "induction", "9.9": "name mover", "9.6": "name mover", "10.7": "negative name mover"}
    js("data_attention.js", {"sentences": out, "notes": notes})

    # 12 × 12 grid of patterns for the flag sentence (Welch Labs guide, figure 8.1)
    s = out[0]; T = len(s["tokens"])
    fig, axes = plt.subplots(12, 12, figsize=(12, 12))
    for l in range(12):
        for h in range(12):
            M = np.zeros((T, T))
            for i, row in enumerate(s["patterns"][l][h]):
                M[i, :len(row)] = np.array(row) / 200
            ax = axes[l, h]
            ax.imshow(M[1:, 1:], cmap="viridis", vmin=0, vmax=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if h == 0: ax.set_ylabel(str(l), rotation=0, labelpad=12, fontsize=12, color=MUTED)
            if l == 0: ax.set_title(str(h), fontsize=12, color=MUTED)
    fig.subplots_adjust(0.03, 0.01, 0.99, 0.97, 0.08, 0.08)
    fig.savefig(FIG / "gpt2-attention-grid.png", dpi=80); plt.close(fig)

    # per-token loss on the repeated sequence
    r = out[2]; n = (len(r["tokens"]) - 1) // 2
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(1, len(r["tokens"]))
    ax.bar(x, r["loss"][1:], color=[MUTED] * n + [GOLD] * n)
    ax.set_xticks(x, r["tokens"][1:], rotation=60, fontsize=12)
    ax.set_ylabel("loss −log p(token)")
    top = max(r["loss"][1:])
    ax.set_ylim(0, top * 1.25)
    ax.text(n / 2 + 0.5, top * 1.12, "first time: unpredictable", ha="center", color=MUTED, fontsize=15)
    ax.text(n * 1.5 + 0.5, top * 1.12, "second time: copied", ha="center", color="#9c6b00", fontsize=15)
    fig.tight_layout(); fig.savefig(FIG / "induction-loss.svg", transparent=True); plt.close(fig)
    RESULTS["induction_loss"] = {"first": float(np.mean(r["loss"][2:n + 1])), "second": float(np.mean(r["loss"][n + 2:]))}


# ------------------------------------------------------------------ logit lens
def logit_lens(prompt="The electron has a negative", target=" charge"):
    ids = [BOS] + TOK.encode(prompt)
    res = MODEL(torch.tensor([ids]), output_hidden_states=True)
    hs = res.hidden_states                                        # 13 × (1, T, 768)
    tid = TOK.encode(target)[0]
    toks = [show(TOK.decode([i])) for i in ids[1:]]
    rows, probs = [], []
    for l, h in enumerate(hs):
        z = MODEL.transformer.ln_f(h[0]) if l < 12 else h[0]    # the last hidden state already has ln_f applied
        p = MODEL.lm_head(z).softmax(-1)
        rows.append([show(TOK.decode([int(p[i].argmax())])) for i in range(1, len(ids))])
        probs.append([float(p[i].max()) for i in range(1, len(ids))])
        if l == 12:
            pass
    ptarget = []
    for l, h in enumerate(hs):
        z = MODEL.transformer.ln_f(h[0]) if l < 12 else h[0]
        ptarget.append(float(MODEL.lm_head(z[-1]).softmax(-1)[tid]))
    RESULTS["logit_lens"] = {"prompt": prompt, "target": target, "p_target_by_layer": ptarget, "top_last": [r[-1] for r in rows]}
    fig, ax = plt.subplots(figsize=(12, 7.2))
    P = np.array(probs)
    ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for l in range(P.shape[0]):
        for i in range(P.shape[1]):
            ax.text(i, l, rows[l][i].strip() or "␣", ha="center", va="center", fontsize=11.5,
                    color="white" if P[l, i] > 0.55 else INK)
    ax.set_xticks(range(len(toks)), toks, fontsize=14)
    ax.set_yticks(range(13), ["embed"] + [f"layer {l}" for l in range(1, 13)], fontsize=12)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_xlabel("input position (each cell: most likely NEXT token read out from the residual stream)", fontsize=13)
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "logit-lens.svg", transparent=True); plt.close(fig)


# ------------------------------------------------------------------ embeddings
def embeddings():
    groups = {
        "physics": "electron proton neutron photon atom nucleus quark energy mass charge force field wave particle laser magnet gravity momentum spin plasma",
        "countries": "France Germany Italy Spain Japan China Russia Canada Brazil Egypt India Poland Sweden Greece Mexico",
        "capitals": "Paris Berlin Rome Madrid Tokyo Beijing Moscow Ottawa Cairo Delhi Warsaw Stockholm Athens London Vienna",
        "people": "man woman king queen boy girl father mother son daughter brother sister husband wife prince princess uncle aunt",
        "numbers": "one two three four five six seven eight nine ten hundred thousand million",
        "days & months": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July",
        "colours": "red blue green yellow black white orange purple pink brown gray",
        "animals": "dog cat horse cow pig sheep lion tiger bear wolf mouse rabbit bird fish snake",
        "verbs": "run walk swim eat drink sleep read write speak think see hear jump fly",
        "past tense": "ran walked swam ate drank slept wrote spoke thought saw heard jumped flew",
        "science": "physics chemistry biology mathematics geology astronomy medicine",
    }
    E = MODEL.transformer.wte.weight
    En = E / E.norm(dim=1, keepdim=True)
    words, cats, ids = [], [], []
    for c, ws in groups.items():
        for w in ws.split():
            i = TOK.encode(" " + w)
            if len(i) == 1:
                words.append(w); cats.append(c); ids.append(i[0])
    V = E[ids].numpy()
    xy = TSNE(n_components=2, perplexity=12, random_state=0, init="pca").fit_transform(V)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    # nearest neighbours over the full vocabulary (whole words that start with a space)
    ok = torch.tensor([TOK.decode([i]).startswith(" ") and TOK.decode([i])[1:].isalpha() for i in range(E.shape[0])])
    nn = []
    for i in ids:
        s = En @ En[i]; s[~ok] = -2; s[i] = -2
        nn.append([TOK.decode([int(j)])[1:] for j in s.topk(6).indices])
    scale = np.abs(V).max(1, keepdims=True) / 127
    q = np.round(V / scale).astype(np.int8)
    js("data_embed.js", {
        "words": words, "cats": cats, "catNames": list(groups), "xy": np.round(xy, 4).tolist(),
        "scale": np.round(scale[:, 0], 6).tolist(), "dim": int(V.shape[1]),
        "vec": base64.b64encode(q.tobytes()).decode(), "nn": nn,
    })
    RESULTS["embedding_words"] = len(words)


# ------------------------------------------------------------------ next-token sampling
def sampler():
    prompts = [
        "The American flag is red, white, and",
        "The capital of France is",
        "The electron and the",
        "Once upon a",
        "In particle physics, the Higgs boson",
        "My favourite thing about Bochum is",
    ]
    data = []
    for k, s in enumerate(prompts):
        ids = TOK.encode(s)
        logits = MODEL(torch.tensor([ids])).logits[0, -1]
        v, i = logits.topk(200)
        full = logits.softmax(-1)
        # The other ~50 000 logits, as a histogram: enough to renormalize exactly at any temperature.
        rest = logits.clone(); rest[i] = -1e9
        rest = rest[rest > -1e8].numpy()
        edges = np.linspace(rest.min(), rest.max(), 121)
        counts, _ = np.histogram(rest, edges)
        centres = 0.5 * (edges[1:] + edges[:-1])
        gens = {}
        for m, T in enumerate([0.0, 0.7, 1.0, 1.5]):
            torch.manual_seed(100 * k + m)
            out = MODEL.generate(torch.tensor([ids]), max_new_tokens=28, do_sample=T > 0, temperature=T if T > 0 else None,
                                 top_k=0, pad_token_id=BOS)
            gens[str(T)] = TOK.decode(out[0, len(ids):])
        data.append({"prompt": s, "tokens": [TOK.decode([int(j)]) for j in i], "logits": [round(float(x), 3) for x in v],
                     "restLogit": [round(float(c), 3) for c in centres], "restCount": counts.tolist(),
                     "p": [round(float(full[j]), 4) for j in i[:5]], "covered": round(float(full[i].sum()), 4), "gens": gens})
    js("data_sampler.js", {"prompts": data})
    RESULTS["sampler"] = {d["prompt"]: {"top": list(zip(d["tokens"][:5], d["p"])),
                                        "covered_by_top200": d["covered"], "gens": d["gens"]} for d in data}


if __name__ == "__main__":
    parameters()
    tokenizers()
    attention()
    logit_lens()
    embeddings()
    sampler()
    (FIG / "results.json").write_text(json.dumps(RESULTS, indent=1, ensure_ascii=False))
    print(json.dumps({k: RESULTS[k] for k in ["gpt2_parameters", "induction_loss", "logit_lens"]}, indent=1, ensure_ascii=False))
