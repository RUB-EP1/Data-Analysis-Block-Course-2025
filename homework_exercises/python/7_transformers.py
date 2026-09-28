# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "torch>=2.2", "matplotlib>=3.8", "tokenizers>=0.20"]
# ///

import marimo

__generated_with = "0.24.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from torch import nn
    import matplotlib.pyplot as plt
    import time
    import copy

    return copy, mo, nn, plt, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Transformers: from attention to a classifier and a tokenizer

    This homework follows up on the attendance exercise "Transformers by Hand". There,
    you computed attention, shapes and parameter counts on paper. Here, you implement
    and run the same ideas in PyTorch, then train a small tokenizer and connect its
    vocabulary size back to the embedding-matrix parameter count from Exercise 1.1
    of the attendance sheet.

    Work through the sections in order; later cells deliberately wait until their
    prerequisites work. No language-model download or HEP files are needed. CPU is
    the default and sufficient throughout.

    | Section | Topic |
    |---|---|
    | 1 | Tokens, shapes, prediction task |
    | 2 | Scaled dot-product attention, implemented by hand |
    | 3 | `torch.nn.MultiheadAttention` |
    | 4 | Pooling, encoder, permutation invariance |
    | 5 | Training and comparison |
    | 6 | Training and running your own tokenizer |

    The `NotImplementedError` functions are your tasks. Complete them in order.

    **Our task for Sections 1-5:** each example is a padded set of `(key, value)`
    pairs. Predict whether the value belonging to the **largest key** is positive.
    Both labels are equally probable in the data-generating process.

    A token is one pair, an example is one set, and a batch contains many sets.
    There is no physical ordering here, which is exactly why Exercise 1.3 and
    Exercise 2.1 on the attendance sheet matter: an encoder can classify a whole
    input; a decoder with causal masking instead supports next-token prediction.
    """)
    return


@app.cell
def _(torch):
    def make_sample(n, seed, max_tokens=8):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(n, max_tokens, 2, generator=generator)
        lengths = torch.randint(3, max_tokens + 1, (n,), generator=generator)
        padding = torch.arange(max_tokens)[None, :] >= lengths[:, None]
        winner = x[:, :, 0].masked_fill(padding, -torch.inf).argmax(dim=1)
        y = (x[torch.arange(n), winner, 1] > 0).float()
        x = x.masked_fill(padding[:, :, None], 0)
        return x, padding, y

    train_data = make_sample(2048, 11)
    val_data = make_sample(512, 12)
    print("x, padding, y:", [tuple(t.shape) for t in train_data])
    print("Training signal fraction:", train_data[2].mean().item())
    return train_data, val_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Write attention

    For one head, $A=\mathrm{softmax}(QK^T/\sqrt{d_k})$, $Z=AV$.
    Softmax runs over **keys**, separately for each query. Compare with your
    by-hand calculation in Exercise 2.1 of the attendance sheet before coding:
    the shapes here are the batched version of that same calculation.

    Implement `attention`: `q` is `(B,Lq,D)`, `k` is `(B,Lk,D)`,
    `v` is `(B,Lk,Dv)` and `padding` is `(B,Lk)`.
    `True` means ignore that key. Use `masked_fill` with negative infinity
    **before** softmax. Assume at least one valid key per example.
    Return `(output, weights)`. Do not call a ready-made attention function yet.

    Discuss: why divide by $\sqrt{d_k}$? What happens if every key is masked?
    Is this a causal mask? Which information should a classifier be allowed to see?
    """)
    return


@app.function
def attention(q, k, v, padding):
    """Return output (B,Lq,Dv), weights (B,Lq,Lk); True masks a key."""
    raise NotImplementedError("Exercise 2: scores, mask, softmax, weighted values")


@app.cell
def _(mo, torch):
    try:
        _q = torch.zeros(1, 2, 3)
        _k = torch.zeros(1, 3, 3)
        _v = torch.tensor([[[2.], [4.], [100.]]])
        _mask = torch.tensor([[False, False, True]])
        _out, _weights = attention(_q, _k, _v, _mask)
        assert _out.shape == (1, 2, 1)
        torch.testing.assert_close(_out, torch.full((1, 2, 1), 3.))
        torch.testing.assert_close(_weights.sum(-1), torch.ones(1, 2))
        assert (_weights[..., 2] == 0).all()
        attention_ready = True
        print("Uniform-score and padding checks passed. Add a non-uniform example.")
    except NotImplementedError:
        attention_ready = False
    mo.md("Attention checks passed." if attention_ready else "Complete Exercise 2 to unlock the comparison.")
    return (attention_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Explore the library

    Read [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
    Find `batch_first`, `key_padding_mask`, `need_weights`, and
    `average_attn_weights`. Predict the returned shapes for two heads.
    Why must the embedding dimension be divisible by the number of heads?
    (You answered this by hand in Exercise 2.3 of the attendance sheet.)

    The cell below checks your function against a single head with identity
    projections. A general multi-head layer also learns Q/K/V and output
    projections, so its output will not equal attention on the raw input.
    Change the exploration layer to four heads and inspect another head.
    Remove the padding mask and explain the difference. These are **untrained**
    weights; a heatmap alone is not an explanation of a trained prediction.
    """)
    return


@app.cell
def _(attention_ready, mo, nn, plt, torch, train_data):
    mo.stop(not attention_ready, mo.md("Waiting for Exercise 2."))
    torch.manual_seed(20)
    _x, _padding, _ = train_data
    _x, _padding = _x[:1], _padding[:1]
    _identity = nn.MultiheadAttention(2, 1, dropout=0., bias=False, batch_first=True)
    with torch.no_grad():
        _identity.in_proj_weight.copy_(torch.eye(2).repeat(3, 1))
        _identity.out_proj.weight.copy_(torch.eye(2))
        _actual, _w = _identity(_x, _x, _x, key_padding_mask=_padding)
        _expected, _a = attention(_x, _x, _x, _padding)
    torch.testing.assert_close(_actual, _expected)
    torch.testing.assert_close(_w, _a)
    _embed = nn.Linear(2, 16)(_x)
    _layer = nn.MultiheadAttention(16, 2, batch_first=True, dropout=0.)
    _z, _heads = _layer(_embed, _embed, _embed, key_padding_mask=_padding,
                        need_weights=True, average_attn_weights=False)
    print("Output / per-head weights:", _z.shape, _heads.shape)
    _fig, _ax = plt.subplots()
    _im = _ax.imshow(_heads[0, 0].detach().numpy(), vmin=0)
    _ax.set(xlabel="Key index", ylabel="Query index", title="Untrained head 0")
    _fig.colorbar(_im, ax=_ax)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. From tokens to one prediction

    Implement `masked_mean(x, padding)` for `(B,L,D)` → `(B,D)`.
    Exclude padded tokens from both numerator and denominator; assume each
    example contains at least one real token. A key-padding mask does not force
    the outputs at padded **query** positions to zero, so pooling needs a mask too.

    Inspect [TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).
    Locate its feed-forward network, residual connections, normalisation and
    dropout — this is the same block you labelled by hand in Exercise 3.1 of the
    attendance sheet.

    Predict: will reordering the valid tokens change this model's prediction?
    Run the supplied check in evaluation mode. Explain why positional embeddings
    would be useful for a sentence but are unnecessary for this task (Exercise 1.3).
    """)
    return


@app.function
def masked_mean(x, padding):
    """Average valid token vectors; return shape (B,D)."""
    raise NotImplementedError("Exercise 4: mask, sum, divide by valid count")


@app.cell
def _(mo, torch):
    try:
        _x = torch.tensor([[[2., 4.], [4., 8.], [900., 900.]]])
        _result = masked_mean(_x, torch.tensor([[False, False, True]]))
        torch.testing.assert_close(_result, torch.tensor([[3., 6.]]))
        pooling_ready = True
    except NotImplementedError:
        pooling_ready = False
    mo.md("Pooling check passed." if pooling_ready else "Complete masked_mean.")
    return (pooling_ready,)


@app.cell
def _(nn):
    class SetClassifier(nn.Module):
        def __init__(self, use_attention=True):
            super().__init__()
            self.embed = nn.Linear(2, 32)
            self.encoder = (
                nn.TransformerEncoderLayer(32, 2, dim_feedforward=64,
                                           dropout=0., batch_first=True)
                if use_attention else None
            )
            self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x, padding):
            h = self.embed(x)
            if self.encoder is not None:
                h = self.encoder(h, src_key_padding_mask=padding)
            return self.head(masked_mean(h, padding)).squeeze(-1)

    return (SetClassifier,)


@app.cell
def _(SetClassifier, mo, pooling_ready, torch, train_data):
    mo.stop(not pooling_ready, mo.md("Waiting for pooling."))
    torch.manual_seed(30)
    _model = SetClassifier().eval()
    _x, _p, _ = train_data
    _order = torch.randperm(_x.shape[1])
    with torch.no_grad():
        _original = _model(_x[:4], _p[:4])
        _permuted = _model(_x[:4, _order], _p[:4, _order])
    torch.testing.assert_close(_original, _permuted, atol=1e-6, rtol=1e-5)
    print("Joint token/mask permutation leaves predictions unchanged.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Train, then investigate

    Implement `accuracy(logits, labels)`. A logit of zero corresponds to
    probability 0.5. Return a Python float. The training loop is supplied.
    `BCEWithLogitsLoss` takes raw logits: do not apply sigmoid first.

    Train once with the default settings. Compare with the supplied mean-pooling
    baseline, which has no token interaction before pooling. It is a deliberately
    simple baseline, not a parameter-matched comparison. Inspect parameter counts
    and compare them with your hand calculation in Exercise 3.3 of the attendance
    sheet — note the model here uses different `d_model`/`d_ff`/`h`, so the numbers
    will differ, but the same formula applies.

    Change **one** of the epoch count, learning rate or number of heads; record
    the change and its validation result. Do not expect a guaranteed accuracy.

    Deliver: loss curves, validation accuracies, wall time and a two-sentence
    interpretation. Could a handcrafted rule solve this synthetic task exactly?
    Why does that not make the learning exercise meaningless? Validation is used
    for exploration here; it is not an unbiased final test after repeated tuning.
    """)
    return


@app.function
def accuracy(logits, labels):
    """Fraction correctly classified at logit threshold zero, as a float."""
    raise NotImplementedError("Exercise 5: threshold and compare")


@app.cell
def _(mo):
    run_training = mo.ui.run_button(label="Train both small models (CPU)")
    run_training
    return (run_training,)


@app.cell
def _(
    SetClassifier,
    copy,
    mo,
    nn,
    pooling_ready,
    run_training,
    time,
    torch,
    train_data,
    val_data,
):
    mo.stop(not pooling_ready or not run_training.value, mo.md("Complete pooling and accuracy, then click Train."))
    try:
        assert accuracy(torch.tensor([-1., 1.]), torch.tensor([0., 1.])) == 1.
        assert accuracy(torch.tensor([1., -1.]), torch.tensor([0., 1.])) == 0.
    except NotImplementedError:
        mo.stop(True, mo.md("Complete accuracy, then click Train again."))
    epochs = 12  # Start small; measure before increasing.
    learning_rate = 3e-3
    torch.set_num_threads(2)
    training_results = {}
    for _name, _use_attention in [("Mean baseline", False), ("Transformer", True)]:
        torch.manual_seed(42)
        _model = SetClassifier(_use_attention)
        _optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate)
        _loss_fn = nn.BCEWithLogitsLoss()
        _x, _padding, _y = train_data
        _vx, _vp, _vy = val_data
        _history = []
        _best_loss = float("inf")
        _best_state = None
        _start = time.perf_counter()
        for _epoch in range(epochs):
            _model.train()
            _order = torch.randperm(len(_y))
            _loss_sum = 0.
            for _idx in _order.split(128):
                _optimizer.zero_grad()
                _loss = _loss_fn(_model(_x[_idx], _padding[_idx]), _y[_idx])
                _loss.backward()
                _optimizer.step()
                _loss_sum += _loss.item() * len(_idx)
            _model.eval()
            with torch.no_grad():
                _vl = _loss_fn(_model(_vx, _vp), _vy).item()
            _history.append((_loss_sum / len(_y), _vl))
            if _vl < _best_loss:
                _best_loss = _vl
                _best_state = copy.deepcopy(_model.state_dict())
        _model.load_state_dict(_best_state)
        _model.eval()
        with torch.no_grad():
            _acc = accuracy(_model(_vx, _vp), _vy)
        training_results[_name] = {"history": _history, "accuracy": _acc,
            "seconds": time.perf_counter() - _start,
            "parameters": sum(p.numel() for p in _model.parameters())}
        print(_name, {k: v for k, v in training_results[_name].items() if k != "history"})
    return (training_results,)


@app.cell
def _(plt, training_results):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    for _ax, (_name, _result) in zip(_axes, training_results.items()):
        _ax.plot(range(1, len(_result["history"]) + 1), _result["history"])
        _ax.set(title=_name, xlabel="Epoch", ylabel="Binary cross entropy")
        _ax.legend(["Train (during updates)", "Validation (end of epoch)"])
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Training and running your own tokenizer

    Sections 1-5 fed continuous `(key, value)` pairs directly into the network:
    there was no vocabulary and no embedding lookup. Real text input needs a
    **tokenizer** first, to turn raw text into a sequence of integer indices
    that an embedding matrix (Exercise 1.1 of the attendance sheet) can look up.

    We train a small [byte-level BPE](https://huggingface.co/docs/tokenizers/quicktour)
    (Byte-Pair Encoding) tokenizer from scratch on a tiny in-memory corpus, so no
    internet access or pretrained files are needed. BPE starts from individual
    characters/bytes and iteratively merges the most frequent adjacent pair into
    a new subword token, up to a target vocabulary size.
    """)
    return


@app.cell
def _():
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "a cat and a dog are pets",
        "transformers use self-attention",
        "attention is computed between queries and keys",
        "the tokenizer splits text into subword tokens",
        "byte pair encoding merges frequent character pairs",
        "positional encodings add order information to embeddings",
        "the encoder is bidirectional and the decoder is causal",
        "multi-head attention splits the model dimension across heads",
    ]
    return (corpus,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 6.1 - Train a BPE tokenizer

    Implement `train_bpe_tokenizer(corpus, vocab_size)`:

    1. Create a `tokenizers.Tokenizer` with a `tokenizers.models.BPE()` model.
    2. Set its pre-tokenizer to `tokenizers.pre_tokenizers.Whitespace()` (split on
       whitespace/punctuation boundaries before BPE merges characters within each piece).
    3. Create a `tokenizers.trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]"])`.
    4. Call `tokenizer.train_from_iterator(corpus, trainer)`.
    5. Return the trained `tokenizer`.

    Use a small `vocab_size` (e.g. 80) so that BPE only has time to learn a few
    merges beyond individual characters — this makes the subword splitting in
    Exercise 6.2 visible and interesting.
    """)
    return


@app.function
def train_bpe_tokenizer(corpus, vocab_size):
    """Train and return a byte-level BPE tokenizers.Tokenizer on `corpus`."""
    raise NotImplementedError("Exercise 6.1: build model, pre-tokenizer, trainer; train")


@app.cell
def _(corpus, mo):
    try:
        tokenizer = train_bpe_tokenizer(corpus, vocab_size=80)
        tokenizer_ready = True
        print("Trained vocabulary size:", tokenizer.get_vocab_size())
    except NotImplementedError:
        tokenizer_ready = False
        tokenizer = None
    mo.md("Tokenizer trained." if tokenizer_ready else "Complete Exercise 6.1 to train the tokenizer.")
    return tokenizer, tokenizer_ready


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 6.2 - Run the tokenizer

    Encode a sentence that was **not** in the training corpus, e.g.
    `"the transformer decoder generates text"`. Print

    - the list of resulting subword tokens (`encoding.tokens`),
    - the list of resulting integer ids (`encoding.ids`),
    - the number of words in the sentence versus the number of tokens produced.

    Which words got split into more than one subword token? Check how often
    each word (or a word containing it) occurs in `corpus` — do frequently
    occurring words survive as single tokens more often than rare ones, even
    if a rare word did appear as a whole word somewhere in the corpus? With
    only `vocab_size=80`, BPE has very few merge steps to spend, so it spends
    them on the *most frequent* adjacent pairs first. Relate this to why real
    tokenizers (trained on much larger corpora) still occasionally split rare
    or unseen words into several pieces, e.g. `"tokenization" -> "token" + "ization"`.
    """)
    return


@app.cell
def _(mo, tokenizer, tokenizer_ready):
    mo.stop(not tokenizer_ready, mo.md("Waiting for Exercise 6.1."))
    _sentence = "the transformer decoder generates text"
    _encoding = tokenizer.encode(_sentence)
    print("Tokens:", _encoding.tokens)
    print("Ids:", _encoding.ids)
    print("Words:", len(_sentence.split()), "-> Tokens:", len(_encoding.tokens))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 6.3 - Subword splits per word

    Implement `tokens_per_word(tokenizer, words)`: for each word in `words`,
    encode it on its own and return a dictionary mapping the word to the number
    of tokens it was split into.

    Apply it to `["cat", "dog", "attention", "tokenizer", "self-attention",
    "encodings"]`. Which words are unaffected (map to 1 token) and which are
    split? For each word, count how many times it (or, for "self-attention",
    its parts) occurs in `corpus`. Does token count correlate with frequency
    of occurrence, even among words that all appeared literally in the
    corpus at least once?
    """)
    return


@app.function
def tokens_per_word(tokenizer, words):
    """Return {word: number of subword tokens} for each word in `words`."""
    raise NotImplementedError("Exercise 6.3: encode each word, count tokens")


@app.cell
def _(mo, tokenizer, tokenizer_ready):
    mo.stop(not tokenizer_ready, mo.md("Waiting for Exercise 6.1."))
    try:
        _words = ["cat", "dog", "attention", "tokenizer", "self-attention", "encodings"]
        _counts = tokens_per_word(tokenizer, _words)
        for _word, _count in _counts.items():
            print(f"{_word!r}: {_count} token(s)")
    except NotImplementedError:
        mo.md("Complete Exercise 6.3.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 6.4 - From vocabulary size to embedding parameters

    Recall Exercise 1.1 of the attendance sheet: an embedding matrix has shape
    `(vocab_size, d)` and thus `vocab_size * d` trainable parameters.

    Using `tokenizer.get_vocab_size()` from Exercise 6.1, compute the number of
    parameters an embedding layer would need for this tokenizer at `d=128`,
    `d=512` and `d=768` (a small, a medium and a GPT-2-sized embedding width).

    A real subword tokenizer trained on a large corpus typically has a
    vocabulary of 30,000-100,000 tokens rather than the ~80 used here. Recompute
    the three parameter counts for `vocab_size=50000` and compare the scale of
    just the embedding matrix with the ~3.1M parameters of a single encoder
    layer you calculated in Exercise 3.3 of the attendance sheet.
    """)
    return


@app.cell
def _(mo, tokenizer, tokenizer_ready):
    mo.stop(not tokenizer_ready, mo.md("Waiting for Exercise 6.1."))
    # Write here: compute vocab_size * d for d in (128, 512, 768),
    # for both tokenizer.get_vocab_size() and vocab_size=50000.
    return


if __name__ == "__main__":
    app.run()
