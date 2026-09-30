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
    # A direction in, a sensitivity back
    $F(x_1,x_2)=(x_1x_2,\ x_1^2+x_2)$ at $x=(2,3)$.
    Forward mode computes $Jv$; reverse mode computes $J^T u$.
    """)
    return


@app.cell
def _(mo):
    direction = mo.ui.dropdown({"e₁ = (1, 0)": 0, "e₂ = (0, 1)": 1}, value="e₁ = (1, 0)", label="Input direction v")
    seed = mo.ui.dropdown({"First output: u = (1, 0)": 0, "Second output: u = (0, 1)": 1}, value="First output: u = (1, 0)", label="Output seed u")
    mo.hstack([direction, seed])
    return direction, seed


@app.cell
def _(jnp):
    def F(x):
        return jnp.array([x[0] * x[1], x[0]**2 + x[1]])
    return (F,)


@app.cell
def _(F, direction, jax, jnp, seed):
    x = jnp.array([2.0, 3.0])
    v = jnp.eye(2)[direction.value]
    u = jnp.eye(2)[seed.value]
    y, Jv = jax.jvp(F, (x,), (v,))
    _, pullback = jax.vjp(F, x)
    (JTu,) = pullback(u)
    J_forward = jax.jacfwd(F)(x)
    J_reverse = jax.jacrev(F)(x)
    expected = jnp.array([[3.0, 2.0], [4.0, 1.0]])
    assert jnp.allclose(J_forward, expected)
    assert jnp.allclose(J_reverse, expected)
    assert jnp.allclose(Jv, expected @ v)
    assert jnp.allclose(JTu, expected.T @ u)
    return J_forward, J_reverse, JTu, Jv, u, v, x, y


@app.cell
def _(JTu, Jv, mo, u, v, y):
    mo.md(r"""
    $$J=\begin{pmatrix}3&2\\4&1\end{pmatrix}$$
    """ + f"""
    | Quantity | Result |
    |:--|:--|
    | $F(x)$ | `{y.tolist()}` |
    | Input tangent $v$ | `{v.tolist()}` |
    | $Jv$ | **`{Jv.tolist()}`** |
    | Output sensitivity $u$ | `{u.tolist()}` |
    | $J^T u$ | **`{JTu.tolist()}`** |

    A basis input direction selects a **column**. A basis output seed selects a **row** (returned as a column vector).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Try:** switch the input direction and the output seed independently.
    `jacfwd` and `jacrev` assemble the same full Jacobian using different modes.

    [JAX forward- and reverse-mode guide](https://docs.jax.dev/en/latest/jacobian-vector-products.html)
    """)
    return


if __name__ == "__main__":
    app.run()
