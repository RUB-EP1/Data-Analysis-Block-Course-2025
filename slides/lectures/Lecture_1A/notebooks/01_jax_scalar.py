import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", css_file="notebook.css")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    return jax, jnp, mo


@app.cell
def _(mo):
    mo.md(r"""
    # One loss, two autodiff modes
    The same example as the lecture: $a=2w$, $c=a^2+a$, $L=\frac12(c-5)^2$.
    Move the weight and compare the two derivatives.
    """)
    return


@app.cell
def _(mo):
    weight = mo.ui.slider(-1.5, 2.0, step=0.1, value=1.0, label="Weight w", show_value=True)
    weight
    return (weight,)


@app.cell
def _():
    def loss(w):
        a = 2.0 * w
        c = a**2 + a
        return 0.5 * (c - 5.0)**2
    return (loss,)


@app.cell
def _(jax, jnp, loss, weight):
    w = jnp.array(weight.value, dtype=jnp.float64)
    value, forward = jax.jvp(loss, (w,), (jnp.ones_like(w),))
    _, pullback = jax.vjp(loss, w)
    (backward,) = pullback(jnp.array(1.0))
    gradient = jax.grad(loss)(w)
    analytic = ((2*w)**2 + 2*w - 5) * (8*w + 2)
    assert jnp.allclose(jnp.array([forward, backward, gradient]), analytic)
    return analytic, backward, forward, gradient, value, w


@app.cell
def _(backward, forward, gradient, mo, value):
    mo.md(f"""
    | Computation | Result |
    |:--|--:|
    | Loss | {float(value):.6g} |
    | Forward mode: `jax.jvp`, input tangent 1 | {float(forward):.6g} |
    | Reverse mode: `jax.vjp`, output seed 1 | {float(backward):.6g} |
    | Scalar gradient: `jax.grad` | {float(gradient):.6g} |

    Both modes agree. Their computational advantage depends on input and output dimensions.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Try:** set $w=0.9$. The loss becomes $0.0008$ and the gradient becomes $0.368$.

    [JAX JVP documentation](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html)
    · [JAX VJP documentation](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html)
    """)
    return


if __name__ == "__main__":
    app.run()
