# Python autodiff examples in marimo

The three notebooks are executable Python files with reactive controls:

1. `01_jax_scalar.py`: the lecture loss using JAX `jvp`, `vjp`, and `grad`.
2. `02_jax_vector.py`: input directions, output seeds, and full Jacobians.
3. `03_pytorch_training.py`: PyTorch transforms and one optimizer step.

The slides use screenshots of these notebooks after execution. The files in
`exports/` are static HTML snapshots; their controls do not execute Python.

## Run locally

The shared environment is `.venv-lectures`. From the `slides/` directory,
open the Lecture 1A notebook collection with:

```sh
.venv-lectures/bin/marimo run lectures/Lecture_1A/notebooks --headless --port 42761
```

Visit `http://127.0.0.1:42761` and select an example. To edit code:

```sh
.venv-lectures/bin/marimo edit lectures/Lecture_1A/notebooks/01_jax_scalar.py
```

For a fresh environment, use Python 3.12 and install the tested packages:

```sh
python3.12 -m venv .venv-lectures
.venv-lectures/bin/python -m pip install -r lectures/Lecture_1A/notebooks/requirements-lock.txt
```

Tested here with marimo 0.24.2, JAX 0.11.2, and PyTorch 2.14.0 on CPU.
The full lock file records the exact environment used for execution.

## Verify and export

```sh
.venv-lectures/bin/marimo check --strict lectures/Lecture_1A/notebooks/0*.py
.venv-lectures/bin/python scripts/verify_python_examples.py
.venv-lectures/bin/marimo export html notebooks/01_jax_scalar.py \
  -o notebooks/exports/01_jax_scalar.html --no-include-code
```

Repeat the export command for the other two examples after changing them.

The PyTorch notebook constructs a fresh parameter on each reactive execution,
so its sliders always describe one update from the displayed initial value.
The JAX notebooks enable 64-bit values to make numerical comparisons clear.
