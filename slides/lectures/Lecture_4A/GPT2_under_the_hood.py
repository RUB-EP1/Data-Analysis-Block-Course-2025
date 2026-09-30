import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GPT-2 under the hood

    **Lecture 4A companion.** We open GPT-2 small (124 M parameters, 2019), the last large OpenAI model
    with public weights and still a faithful miniature of today's chat models, and follow a prompt through it:
    tokens → embeddings → attention → residual stream → next-token probabilities. At the end we write a
    tiny GPT from scratch and train it on the text of this course.

    Install once (Python 3.10+), from the `Lecture_4A` folder:

    ```bash
    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/marimo edit GPT2_under_the_hood.py
    ```

    The first run downloads GPT-2 (~550 MB) from the Hugging Face hub. Everything runs on a laptop CPU.
    """)
    return


@app.cell
def _():
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import tiktoken
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    torch.set_grad_enabled(False)
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    return F, GPT2LMHeadModel, GPT2TokenizerFast, Path, np, nn, plt, tiktoken, time, torch


@app.cell
def _(GPT2LMHeadModel, GPT2TokenizerFast):
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    BOS = tok.eos_token_id
    print(model)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    return BOS, model, tok


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Tokenization

    Type anything. Each coloured chip is one token; the number under it is its id. Compare GPT-2's
    50 257-token vocabulary with the 100 k and 200 k vocabularies of later OpenAI models
    (`·` marks a space, which belongs to the token that follows it).
    """)
    return


@app.cell
def _(mo):
    tok_text = mo.ui.text_area(
        value="The electron has a negative charge. Das Elektron ist negativ geladen. Электрон заряжен отрицательно. 12345678 strawberry",
        full_width=True, rows=3)
    tok_text
    return (tok_text,)


@app.cell
def _(mo, tiktoken, tok_text):
    def chips(enc_name):
        enc = tiktoken.get_encoding(enc_name)
        ids = enc.encode(tok_text.value)
        spans = []
        for i in ids:
            s = enc.decode([i]).replace(" ", "·").replace("\n", "↵").replace("<", "&lt;")
            hue = (i * 47) % 360
            spans.append(f"<span style='display:inline-block;margin:2px;padding:1px 4px;border-radius:4px;"
                         f"background:hsl({hue} 70% 88%);font-family:monospace'>{s}<br>"
                         f"<small style='color:#617484'>{i}</small></span>")
        return mo.vstack([mo.md(f"**{enc_name}** — {len(ids)} tokens for {len(tok_text.value)} characters"),
                          mo.Html("".join(spans))])

    mo.vstack([chips("gpt2"), chips("cl100k_base"), chips("o200k_base")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The next token

    The model returns one score (logit) for each of the 50 257 tokens. Softmax with temperature $T$ turns
    them into probabilities; generation samples a token, appends it, and repeats.
    """)
    return


@app.cell
def _(mo):
    prompt = mo.ui.text(value="The electron has a negative", full_width=True, label="Prompt")
    temperature = mo.ui.slider(0.0, 2.0, step=0.05, value=0.8, label="Temperature", show_value=True)
    n_new = mo.ui.slider(5, 80, step=5, value=40, label="New tokens", show_value=True)
    generate = mo.ui.run_button(label="Generate")
    mo.vstack([prompt, mo.hstack([temperature, n_new, generate], justify="start", gap=2)])
    return generate, n_new, prompt, temperature


@app.cell
def _(model, plt, prompt, temperature, tok, torch):
    ids = tok.encode(prompt.value)
    logits = model(torch.tensor([ids])).logits[0, -1]
    _T = max(temperature.value, 1e-4)
    probs = (logits / _T).softmax(-1)
    _v, _i = probs.topk(15)
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.barh([repr(tok.decode([int(j)])) for j in _i][::-1], _v.numpy()[::-1], color="#087bb9")
    _ax.set_xlabel("probability"); _ax.set_title(f"next token after “{prompt.value}”  (T = {temperature.value})", loc="left")
    _ax.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    _fig
    return (ids,)


@app.cell
def _(BOS, generate, ids, mo, model, n_new, temperature, tok, torch):
    mo.stop(not generate.value, mo.md("*Press Generate.*"))
    _out = model.generate(torch.tensor([ids]), max_new_tokens=n_new.value, do_sample=temperature.value > 0,
                          temperature=temperature.value if temperature.value > 0 else None, top_k=0, pad_token_id=BOS)
    mo.md(f"**{tok.decode(ids)}**{tok.decode(_out[0, len(ids):])}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Attention patterns

    GPT-2 has 12 layers with 12 heads each. Every head computes
    $A = \mathrm{softmax}\!\left(QK^\top/\sqrt{64} + \text{mask}\right)$: row = query token, column = key token.
    Try layer 5, head 5 on a text that repeats itself (an *induction head*), or layer 4, head 11
    (it looks at the previous token).
    """)
    return


@app.cell
def _(mo):
    att_text = mo.ui.text(value=" quantum lamp river seven tiger cloud violin quantum lamp river seven tiger cloud violin",
                          full_width=True, label="Text")
    layer = mo.ui.slider(0, 11, value=5, label="Layer", show_value=True)
    head = mo.ui.slider(0, 11, value=5, label="Head", show_value=True)
    mo.vstack([att_text, mo.hstack([layer, head], justify="start", gap=2)])
    return att_text, head, layer


@app.cell
def _(BOS, att_text, head, layer, model, plt, tok, torch):
    _ids = [BOS] + tok.encode(att_text.value)
    _A = model(torch.tensor([_ids]), output_attentions=True).attentions[layer.value][0, head.value].numpy()
    _labels = ["⟨bos⟩"] + [tok.decode([i]) for i in _ids[1:]]
    _fig, _ax = plt.subplots(figsize=(6.5, 6.5))
    _ax.imshow(_A, cmap="viridis", vmin=0, vmax=0.8)
    _ax.set_xticks(range(len(_labels)), _labels, rotation=90, fontsize=8)
    _ax.set_yticks(range(len(_labels)), _labels, fontsize=8)
    _ax.set_title(f"layer {layer.value}, head {head.value}", loc="left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Logit lens: watching the prediction form

    The residual stream after every layer is a 768-number vector per token. Applying the final layer norm and
    the unembedding matrix to it *early* shows what the model would predict if it stopped there
    (nostalgebraist, 2020).
    """)
    return


@app.cell
def _(BOS, model, np, plt, prompt, tok, torch):
    _ids = [BOS] + tok.encode(prompt.value)
    _hs = model(torch.tensor([_ids]), output_hidden_states=True).hidden_states
    _rows, _P = [], []
    for _l, _h in enumerate(_hs):
        _z = model.transformer.ln_f(_h[0]) if _l < 12 else _h[0]
        _p = model.lm_head(_z).softmax(-1)
        _rows.append([tok.decode([int(_p[i].argmax())]) for i in range(1, len(_ids))])
        _P.append([float(_p[i].max()) for i in range(1, len(_ids))])
    _P = np.array(_P)
    _fig, _ax = plt.subplots(figsize=(1.3 * _P.shape[1] + 2, 6))
    _ax.imshow(_P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for _l in range(_P.shape[0]):
        for _i in range(_P.shape[1]):
            _ax.text(_i, _l, repr(_rows[_l][_i])[1:-1], ha="center", va="center", fontsize=8,
                     color="white" if _P[_l, _i] > 0.55 else "black")
    _ax.set_xticks(range(len(_ids) - 1), [tok.decode([i]) for i in _ids[1:]])
    _ax.set_yticks(range(13), ["embed"] + [f"layer {l}" for l in range(1, 13)])
    _ax.xaxis.tick_top()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The embedding space

    Row $i$ of the 50 257 × 768 embedding matrix is the vector of token $i$. Similar tokens have similar
    vectors, and differences between vectors can carry meaning:
    `king − man + woman` lands near `queen`.
    """)
    return


@app.cell
def _(mo):
    word_b = mo.ui.text(value="king", label="b")
    word_a = mo.ui.text(value="man", label="− a")
    word_c = mo.ui.text(value="woman", label="+ c")
    mo.hstack([word_b, word_a, word_c], justify="start", gap=1)
    return word_a, word_b, word_c


@app.cell
def _(mo, model, tok, word_a, word_b, word_c):
    E = model.transformer.wte.weight
    En = E / E.norm(dim=1, keepdim=True)

    def token_id(w):
        _ids = tok.encode(" " + w.strip())
        return _ids[0] if len(_ids) == 1 else None

    _ia, _ib, _ic = (token_id(w.value) for w in (word_a, word_b, word_c))
    if None in (_ia, _ib, _ic):
        _out = mo.md("Each word must be a **single** GPT-2 token (with a leading space). Try another word.")
    else:
        _v = E[_ib] - E[_ia] + E[_ic]
        _s = En @ (_v / _v.norm())
        for _j in (_ia, _ib, _ic):
            _s[_j] = -1
        _best = [(tok.decode([int(j)]), float(_s[j])) for j in _s.topk(8).indices]
        _near = [tok.decode([int(j)]) for j in (En @ En[_ib]).topk(9).indices[1:]]
        _out = mo.md(f"**{word_b.value} − {word_a.value} + {word_c.value}** ≈ "
                     + ", ".join(f"`{w.strip()}` ({s:.2f})" for w, s in _best)
                     + f"\n\nNearest tokens to **{word_b.value}**: " + ", ".join(f"`{w}`" for w in _near))
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Why a KV cache?

    Generating token $n+1$ needs the keys and values of all $n$ previous tokens. They never change, so the
    model can store them instead of recomputing the whole prefix at every step.
    """)
    return


@app.cell
def _(mo):
    run_kv = mo.ui.run_button(label="Time 200 new tokens with and without the cache")
    run_kv
    return (run_kv,)


@app.cell
def _(BOS, mo, model, run_kv, time, tok, torch):
    mo.stop(not run_kv.value)
    _ids = torch.tensor([tok.encode("The electron has a negative")])
    _t = {}
    for _cache in (True, False):
        _t0 = time.perf_counter()
        model.generate(_ids, max_new_tokens=200, do_sample=False, use_cache=_cache, pad_token_id=BOS)
        _t[_cache] = time.perf_counter() - _t0
    mo.md(f"with cache: **{_t[True]:.1f} s**, without cache: **{_t[False]:.1f} s** "
          f"({_t[False] / _t[True]:.1f}× slower). The gap grows with the length of the text.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. A tiny GPT from scratch

    Everything above fits in a few dozen lines. Below is a character-level GPT: token + position embeddings,
    a stack of blocks (causal multi-head attention + MLP, each with layer norm and a residual connection),
    and an unembedding. We train it on the source files of this course's slides, with the next-character
    cross-entropy loss and Adam from Lecture 3A. About a minute on a laptop CPU.

    Watch the validation curve: the course text is small (~150 k characters), so after roughly 700 steps the
    0.8 M-parameter model starts memorizing it. Training loss keeps falling, validation loss turns up —
    the overfitting of Lecture 3A. Real language models see trillions of tokens and rarely repeat any.
    """)
    return


@app.cell
def _(F, nn, torch):
    class CausalSelfAttention(nn.Module):
        def __init__(self, d, n_heads, context):
            super().__init__()
            self.h, self.dk = n_heads, d // n_heads
            self.qkv = nn.Linear(d, 3 * d)          # W_Q, W_K, W_V for all heads at once
            self.out = nn.Linear(d, d)              # W_O
            self.register_buffer("mask", torch.tril(torch.ones(context, context)).bool())

        def forward(self, x):                        # x: batch × T × d
            B, T, d = x.shape
            q, k, v = self.qkv(x).split(d, dim=-1)
            q, k, v = (t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v))  # B × h × T × dk
            scores = q @ k.transpose(-2, -1) / self.dk ** 0.5                            # B × h × T × T
            scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))              # no peeking ahead
            A = scores.softmax(-1)
            y = (A @ v).transpose(1, 2).reshape(B, T, d)                                 # concatenate heads
            return self.out(y)

    class Block(nn.Module):
        def __init__(self, d, n_heads, context):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn = CausalSelfAttention(d, n_heads, context)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x):
            x = x + self.attn(self.ln1(x))           # tokens exchange information
            x = x + self.mlp(self.ln2(x))            # each token thinks on its own
            return x

    class TinyGPT(nn.Module):
        def __init__(self, vocab, d=128, n_heads=4, n_layers=4, context=128):
            super().__init__()
            self.context = context
            self.tok_emb = nn.Embedding(vocab, d)
            self.pos_emb = nn.Embedding(context, d)
            self.blocks = nn.Sequential(*[Block(d, n_heads, context) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d)
            self.unembed = nn.Linear(d, vocab, bias=False)

        def forward(self, idx):                      # idx: batch × T integers
            T = idx.shape[1]
            x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))
            return self.unembed(self.ln_f(self.blocks(x)))   # logits: batch × T × vocab

        def generate(self, idx, n, temperature=0.8):
            for _ in range(n):
                logits = self(idx[:, -self.context:])[:, -1] / temperature
                idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
            return idx

    return (TinyGPT,)


@app.cell
def _(Path, mo):
    _root = Path(mo.notebook_dir()).resolve().parent          # the lectures/ folder
    _files = sorted(_root.glob("Lecture_*/Lecture_*.qmd"))
    corpus = "\n".join(f.read_text(encoding="utf-8") for f in _files)
    chars = sorted(set(corpus))
    mo.md(f"Training text: **{len(_files)}** lecture sources, **{len(corpus):,}** characters, "
          f"**{len(chars)}** distinct characters (our vocabulary).")
    return chars, corpus


@app.cell
def _(mo):
    steps = mo.ui.slider(200, 3000, step=100, value=1000, label="Training steps", show_value=True)
    run_train = mo.ui.run_button(label="Train the tiny GPT")
    mo.hstack([steps, run_train], justify="start", gap=2)
    return run_train, steps


@app.cell
def _(TinyGPT, chars, corpus, mo, plt, run_train, steps, torch):
    mo.stop(not run_train.value, mo.md("*Press Train (about a minute).*"))
    torch.manual_seed(0)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in corpus])
    n_train = int(0.9 * len(data))
    gpt = TinyGPT(len(chars))
    opt = torch.optim.AdamW(gpt.parameters(), lr=2e-3)

    def batch(split, B=32):
        src = data[:n_train] if split == "train" else data[n_train:]
        i = torch.randint(len(src) - gpt.context - 1, (B,))
        x = torch.stack([src[j:j + gpt.context] for j in i])
        return x, torch.stack([src[j + 1:j + gpt.context + 1] for j in i])   # targets = inputs shifted by one

    history = []
    with torch.enable_grad():
        for _s in mo.status.progress_bar(range(steps.value), title="training"):
            _x, _y = batch("train")
            _loss = torch.nn.functional.cross_entropy(gpt(_x).flatten(0, 1), _y.flatten())
            opt.zero_grad(); _loss.backward(); opt.step()
            if _s % 50 == 0:
                with torch.no_grad():
                    _xv, _yv = batch("val")
                    _lv = torch.nn.functional.cross_entropy(gpt(_xv).flatten(0, 1), _yv.flatten())
                history.append((_s, _loss.item(), _lv.item()))
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.plot([h[0] for h in history], [h[1] for h in history], label="train")
    _ax.plot([h[0] for h in history], [h[2] for h in history], label="validation")
    _ax.axhline(torch.log(torch.tensor(len(chars))).item(), color="grey", ls=":", label="uniform guess")
    _ax.set_xlabel("step"); _ax.set_ylabel("cross-entropy (nats per character)"); _ax.legend(frameon=False)
    _fig.tight_layout()
    mo.vstack([mo.md(f"{sum(p.numel() for p in gpt.parameters()):,} parameters"), _fig])
    return gpt, stoi


@app.cell
def _(chars, gpt, mo, stoi, torch):
    _start = "## Attention"
    _idx = torch.tensor([[stoi[c] for c in _start]])
    torch.manual_seed(1)
    _out = gpt.generate(_idx, 600, temperature=0.8)[0]
    mo.md("Sample from the tiny GPT (it has seen nothing but these slides):\n\n```\n"
          + "".join(chars[i] for i in _out) + "\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercises

    1. Find a word that GPT-2 splits into many tokens but the o200k tokenizer keeps whole. Why does the vocabulary matter for cost and for spelling tasks?
    2. Set the temperature to 0 and generate 80 tokens. What goes wrong? Why does sampling help?
    3. Find a previous-token head and an induction head in section 3. Why does an induction head need a previous-token head in an *earlier* layer?
    4. In the logit lens, at which layer does the correct answer first appear for your own prompt?
    5. Count the parameters of the tiny GPT by hand: embeddings, attention (4 d² + 4d per block), MLP (8 d² + 5d per block), layer norms.
    6. Remove the causal mask in `CausalSelfAttention` and retrain. The training loss falls much faster — why is the model useless for generation?
    7. Replace the learned position embedding with nothing at all. What happens to the samples, and why?
    """)
    return


if __name__ == "__main__":
    app.run()
