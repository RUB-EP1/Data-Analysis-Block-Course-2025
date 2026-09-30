# Lecture 4A: Transformers

`Lecture_4A.qmd` is the native Quarto source (32 slides): generative models and
next-token prediction, tokenization (BPE), embeddings and positions, attention
(query, key, value; scaled dot-product; masks; multi-head), real GPT-2 attention
heads and induction heads, the transformer block, LayerNorm and residual stream, the logit
lens, encoder/decoder families, training, sampling (temperature, top-k, top-p),
parameter counting, the KV cache, and transformers beyond text.

Sources: `Lecture-4_previous.pdf` (generative-models opening, p(x | prompt),
tokenizer, latent space) and chapters 2, 3, 5, 7 and 8 of *The Welch Labs
Illustrated Guide to AI*. The autoencoder and normalizing-flow part of the old
Lecture 4 is not included.

## Present

From the repository root:

```sh
make -C slides html DECK=Lecture_4A
quarto preview slides/lectures/Lecture_4A/Lecture_4A.qmd --port 4195
make -C slides pdf DECK=Lecture_4A
```

## Environment (notebook and data)

From this folder:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The first use of GPT-2 downloads ~550 MB from the Hugging Face hub.

## Marimo notebook: GPT-2 under the hood

```sh
.venv/bin/marimo edit GPT2_under_the_hood.py
```

Tokenizers side by side, next-token probabilities and generation, any attention
head, the logit lens, embedding analogies, KV-cache timing, and a tiny GPT
(~40 lines, 0.8 M parameters) trained in about a minute on the `.qmd` sources of
this course. Static export: `notebooks/exports/GPT2_under_the_hood.html`.

## Five interactive widgets

1. **bpe.html** — byte-pair encoding learned live from an editable training text.
2. **embed.html** — t-SNE map of 151 GPT-2 token embeddings, nearest neighbours,
   and analogies computed in the full 768 dimensions.
3. **attention.html** — hand-built queries, keys and values: which *bank*?
   Includes the causal mask.
4. **heads.html** — real attention patterns of all 144 GPT-2 heads for four texts,
   with known heads marked (previous token, induction, name movers).
5. **sampler.html** — real GPT-2 next-token distributions with temperature,
   top-k and top-p, plus sampled continuations.

Widgets 2, 4 and 5 read `widgets/data_*.js`, produced by `scripts/make_data.py`.
Posters: `zsh scripts/make_posters.zsh` (Google Chrome).

## Figures

- `scripts/make_data.py` (venv): widget data, tokenizer comparison, logit lens,
  induction loss, 144-head grid, and `figures/results.json` with every number
  quoted on the slides (the GPT-2 parameter breakdown is checked by an assertion).
- `scripts/make_diagrams.py` (no dependencies): editable SVGs of the pipeline,
  transformer block, attention shapes, model families, training targets and KV cache.
- `figures/crocodile-turtle.jpg` is the generated image of the old Lecture 4
  (also used in Lecture 2B).

## Notes

- GPT-2 attention and logit-lens figures prepend `<|endoftext|>` as a start token.
- Head 10.7 is labelled a *negative* name mover, following Wang et al. (2022).
- Continuations are generated with fixed seeds; other seeds give other texts.
