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
    # Transformers: from attention to a classifier and a tokenizer (solution)

    This homework follows up on the attendance exercise "Transformers by Hand". There,
    you computed attention, shapes and parameter counts on paper. Here, you implement
    and run the same ideas in PyTorch, then train a small tokenizer and connect its
    vocabulary size back to the embedding-matrix parameter count from Exercise 1.1
    of the attendance sheet.

    | Section | Topic |
    |---|---|
    | 1 | Tokens, shapes, prediction task |
    | 2 | Scaled dot-product attention, implemented by hand |
    | 3 | `torch.nn.MultiheadAttention` |
    | 4 | Pooling, encoder, permutation invariance |
    | 5 | Training and comparison |
    | 6 | Training and running your own tokenizer |

    **Task for Sections 1-5:** each example is a padded set of `(key, value)`
    pairs. Predict whether the value belonging to the **largest key** is positive.
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
    ## 2. Attention

    $A=\mathrm{softmax}(QK^T/\sqrt{d_k})$, $Z=AV$, softmax over keys.
    """)
    return


@app.function
def attention(q, k, v, padding):
    """Return output (B,Lq,Dv), weights (B,Lq,Lk); True masks a key."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d_k**0.5
    scores = scores.masked_fill(padding[:, None, :], float("-inf"))
    weights = scores.softmax(dim=-1)
    output = weights @ v
    return output, weights


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
        print("Uniform-score and padding checks passed.")
    except NotImplementedError:
        attention_ready = False
    mo.md("Attention checks passed." if attention_ready else "Complete Exercise 2 to unlock the comparison.")
    return (attention_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. `torch.nn.MultiheadAttention`

    Sanity-checked below against a single head with identity projections; a
    general multi-head layer also learns Q/K/V and output projections, so its
    output will differ from raw attention on the input once those are non-trivial.
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
    ## 4. Pooling, encoder, permutation invariance

    `masked_mean` excludes padded tokens from both numerator and denominator.
    A `TransformerEncoderLayer` (self-attention + feed-forward, each with a
    residual connection and layer norm) preserves permutation equivariance, so
    a masked mean over it is permutation **invariant**: reordering the valid
    tokens (and their mask together) does not change the pooled prediction.
    """)
    return


@app.function
def masked_mean(x, padding):
    """Average valid token vectors; return shape (B,D)."""
    valid = (~padding).unsqueeze(-1).float()
    total = (x * valid).sum(dim=1)
    count = valid.sum(dim=1).clamp(min=1.0)
    return total / count


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

    `accuracy` thresholds raw logits at zero (equivalent to probability 0.5)
    since `BCEWithLogitsLoss` is trained on logits, not probabilities.
    """)
    return


@app.function
def accuracy(logits, labels):
    """Fraction correctly classified at logit threshold zero, as a float."""
    predictions = (logits > 0).float()
    return (predictions == labels).float().mean().item()


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

    We train a small byte-level BPE tokenizer from scratch on a tiny in-memory
    corpus (no internet access or pretrained files needed), then look at how it
    splits words it has and hasn't seen.
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
    """)
    return


@app.function
def train_bpe_tokenizer(corpus, vocab_size):
    """Train and return a byte-level BPE tokenizers.Tokenizer on `corpus`."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]"])
    tokenizer.train_from_iterator(corpus, trainer)
    return tokenizer


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

    `"the transformer decoder generates text"` was not in the training corpus.
    Only `"the"` survives as one token; `"decoder"` and `"text"` (each present
    but only **once** in the corpus, as whole words) still fragment, because
    BPE merges the *most frequent* adjacent pair first, and with a vocab budget
    of only 80 there are not enough merge steps to fully reassemble words that
    occurred rarely. `"the"` occurs 8 times in the corpus and gets priority.
    This is the same reason real tokenizers split rare words like
    "tokenization" into "token" + "ization": frequency in the training corpus,
    not mere presence, determines which words end up as single tokens.
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
    """)
    return


@app.function
def tokens_per_word(tokenizer, words):
    """Return {word: number of subword tokens} for each word in `words`."""
    return {word: len(tokenizer.encode(word).tokens) for word in words}


@app.cell
def _(mo, tokenizer, tokenizer_ready):
    mo.stop(not tokenizer_ready, mo.md("Waiting for Exercise 6.1."))
    _words = ["cat", "dog", "attention", "tokenizer", "self-attention", "encodings"]
    _counts = tokens_per_word(tokenizer, _words)
    for _word, _count in _counts.items():
        print(f"{_word!r}: {_count} token(s)")
    # "cat" (2x), "dog" (2x) and "attention" (3x, counting the "self-attention"
    # occurrence) come back as a single token, because BPE spent its limited
    # merge budget on these frequent pairs first. "tokenizer" appears only
    # once in the corpus, so despite occurring as a whole word it still
    # fragments; "self-attention" and "encodings" are rarer/compound forms
    # and fragment even more. Frequency in the corpus, not mere presence,
    # decides which words survive as a single token.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 6.4 - From vocabulary size to embedding parameters
    """)
    return


@app.cell
def _(mo, tokenizer, tokenizer_ready):
    mo.stop(not tokenizer_ready, mo.md("Waiting for Exercise 6.1."))
    _trained_vocab = tokenizer.get_vocab_size()
    _realistic_vocab = 50000
    for _vocab_size, _label in [(_trained_vocab, "this tokenizer"), (_realistic_vocab, "realistic subword vocab")]:
        for _d in (128, 512, 768):
            print(f"{_label} (vocab={_vocab_size}), d={_d}: {_vocab_size * _d:,} embedding parameters")
    # At vocab_size=50000, d=512 gives 25.6M embedding parameters alone -
    # roughly 8x the ~3.1M weight parameters of a single encoder layer
    # (Exercise 3.3 of the attendance sheet). The embedding matrix is often
    # one of the largest single parameter blocks in a small transformer.
    return


if __name__ == "__main__":
    app.run()
