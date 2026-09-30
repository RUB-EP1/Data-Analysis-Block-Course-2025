"""Execute the teaching snippets and independently check their derivatives."""
from pathlib import Path
import re
import jax
import jax.numpy as jnp
import torch

ROOT = Path(__file__).resolve().parent.parent
scope = {}
source = (ROOT/'lectures/Lecture_1A/blocks/python-ecosystem.qmd').read_text()
for snippet in re.findall(r'```python\n(.*?)\n```', source, re.S):
    # The first snippet introduces an API before defining F; check it below.
    if snippet.startswith('y, pullback'):
        continue
    exec(snippet, scope)
assert abs(scope['w'].item()-.9) < 1e-12
assert abs(scope['loss'](scope['w']).item()-.0008) < 1e-12

loss = scope['loss']
for weight in [-1.3, -.5, .0, .9, 1., 1.7]:
    expected = ((2*weight)**2+2*weight-5)*(8*weight+2)
    assert abs(float(jax.grad(loss)(weight))-expected) < 1e-10
    tensor = torch.tensor(weight, dtype=torch.float64)
    _, jvp = torch.func.jvp(loss, (tensor,), (torch.ones_like(tensor),))
    _, pullback = torch.func.vjp(loss, tensor)
    (vjp,) = pullback(torch.ones_like(tensor))
    assert abs(jvp.item()-expected) < 1e-10
    assert abs(vjp.item()-expected) < 1e-10

F = scope['F']
x = jnp.array([2., 3.])
J = jnp.array([[3., 2.], [4., 1.]])
for seed in [jnp.array([1., 0.]), jnp.array([0., 1.]), jnp.array([.4, -.7])]:
    _, tangent = jax.jvp(F, (x,), (seed,))
    _, pullback = jax.vjp(F, x)
    (cotangent,) = pullback(seed)
    assert jnp.allclose(tangent, J@seed)
    assert jnp.allclose(cotangent, J.T@seed)
assert jnp.allclose(jax.jacfwd(F)(x), jax.jacrev(F)(x))
print('All displayed code executed; scalar gradients and vector products verified.')
