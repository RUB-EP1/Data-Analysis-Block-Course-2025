import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", css_file="notebook.css")


@app.cell
def _():
    import marimo as mo
    import torch
    return mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    # PyTorch: gradient and parameter update
    Use the same scalar loss. Changing a slider creates a fresh parameter,
    computes its gradient, and performs **one** optimizer step.
    """)
    return


@app.cell
def _(mo):
    start = mo.ui.slider(-1.5, 2.0, step=0.1, value=1.0, label="Initial w", show_value=True)
    rate = mo.ui.slider(0.0, 0.5, step=0.01, value=0.01, label="Learning rate", show_value=True)
    mo.hstack([start, rate])
    return rate, start


@app.cell
def _():
    def loss(w):
        a = 2.0 * w
        return 0.5 * (a**2 + a - 5.0)**2
    return (loss,)


@app.cell
def _(loss, rate, start, torch):
    parameter = torch.tensor(start.value, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=rate.value)
    optimizer.zero_grad(set_to_none=True)
    before = loss(parameter)
    before.backward()
    backward_gradient = parameter.grad.item()
    optimizer.step()
    after = loss(parameter).item()
    updated_weight = parameter.item()
    before_value = before.item()
    return after, backward_gradient, before_value, updated_weight


@app.cell
def _(loss, start, torch):
    primal = torch.tensor(start.value, dtype=torch.float64)
    _, tangent = torch.func.jvp(loss, (primal,), (torch.ones_like(primal),))
    _, pullback = torch.func.vjp(loss, primal)
    (reverse,) = pullback(torch.ones_like(primal))
    assert torch.allclose(tangent, reverse)
    return reverse, tangent


@app.cell
def _(after, backward_gradient, before_value, mo, reverse, tangent, updated_weight):
    mo.md(f"""
    | Quantity | Result |
    |:--|--:|
    | Loss before update | {before_value:.6g} |
    | Gradient from `.backward()` | {backward_gradient:.6g} |
    | Forward mode: `torch.func.jvp` | {tangent.item():.6g} |
    | Reverse mode: `torch.func.vjp` | {reverse.item():.6g} |
    | Updated weight | {updated_weight:.6g} |
    | Loss after update | **{after:.6g}** |

    The loss **{'decreased' if after < before_value else 'increased' if after > before_value else 'stayed the same'}**.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Try:** increase the learning rate to $0.5$. Does a correct gradient guarantee a better step?

    `.backward()` accumulates into `.grad`; `zero_grad()` clears it before a new training step.
    `torch.func.jvp` and `torch.func.vjp` return derivatives directly.

    [PyTorch function transforms](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html)
    """)
    return


if __name__ == "__main__":
    app.run()
