# Lecture 3A: Neural Networks

`Lecture_3A.qmd` is the native Quarto source (28 slides). It replaces the first
half of the imported screenshot deck in `../Lecture_3/` and the neural-network
material that was in its 3B part: history, perceptron, XOR, multilayer
perceptrons, activation functions, universal approximation, output layers and
losses, gradient descent, backpropagation, mini-batches, momentum, AdaGrad,
RMSProp, Adam, initialization and regularization. Convolutional networks are in
`../Lecture_3B/`.

## Present

From the repository root:

```sh
make -C slides html DECK=Lecture_3A
quarto preview slides/lectures/Lecture_3A/Lecture_3A.qmd --port 4193
make -C slides pdf DECK=Lecture_3A
```

## Five interactive widgets

All run locally in the browser, with no server and no data files. They share
`widgets/plot.js` (canvas helpers) and `widgets/lab.css` (the Lecture 2A widget style).

1. **neuron.html** — one neuron on AND, OR, NAND and XOR; perceptron rule or
   sigmoid gradient descent; the decision line with its normal vector.
2. **uat.html** — universal approximation built from sigmoid steps or ReLU kinks,
   with constructed (not trained) weights.
3. **lr.html** — one-dimensional gradient descent: smooth, zig-zag and divergent
   learning rates; a two-minimum loss.
4. **optimizers.html** — GD, momentum, RMSProp and Adam on a narrow valley, the
   Rosenbrock function and a two-basin surface, with optional gradient noise.
5. **playground.html** — a small network trained in the browser (mini-batch Adam,
   hand-written backpropagation) with per-unit activation maps, in the spirit of
   playground.tensorflow.org.

Each widget accepts `?demo`, which pre-runs a representative scenario. Posters for
the PDF export are screenshots of the demo state:

```sh
zsh scripts/make_posters.zsh            # all widgets (needs Google Chrome)
zsh scripts/make_posters.zsh lr.html    # one widget
```

## Live Julia slide (Pluto)

The slide "Your turn: build a shape from ReLUs" (after the universal-approximation
widget) embeds two cells of `scripts/lecture-3B-UAT.jl`, the 12 weight sliders and
the plot, through Pluto's isolated-cell view. Julia runs the network (Flux) and the
plot; the slide only shows it. Before the lecture (about a minute to load Flux):

```sh
make -C slides start-julia-server DECK=Lecture_3A
make -C slides open-html DECK=Lecture_3A
```

The notebooks of a deck are listed in `pluto.toml` with fixed notebook ids;
`slides/scripts/serve_pluto.jl` opens them on port 1243 with the secret off (the
iframe cannot know it; Pluto listens on 127.0.0.1 only) and never writes the `.jl`
files. The slide URL in `Lecture_3A.qmd` uses that id and the two cell ids. While the
server is down, the widget shows the start command over the poster and attaches by
itself once the server answers. The PDF uses `widgets/relu-shapes-poster.png`; to
re-capture it with a freshly started server (sliders at zero):

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
  --force-device-scale-factor=1.3 --window-size=847,630 --virtual-time-budget=15000 \
  --screenshot="$PWD/widgets/relu-shapes-poster.png" \
  "http://localhost:1243/edit?id=3a0a7000-0000-4000-8000-00000000a7a1&isolated_cell_id=6a0ba0ea-6c59-41ae-b706-cca9ba4d7c32&isolated_cell_id=edaeea1e-fc79-4ed6-b29b-06184decba2b"
```

After clicking a slider, the arrow keys stay inside the iframe (it is a different
origin, so they cannot be forwarded to Reveal): click the slide text to navigate again.

## Figures

`scripts/make_figures.py` (needs numpy and matplotlib) writes the activation-function
plot, the ImageNet error chart, and the editable SVG diagrams: neuron, XOR network,
MLP and the backpropagation computational graph with its worked numbers. The
McCulloch–Pitts photo, the *Perceptrons* cover and the Nobel medal are the original
lecture's images, extracted from `previous_lectures/Lecture-3.key`.

```sh
python3 scripts/make_figures.py          # e.g. with ../Lecture_3B/.venv/bin/python
```

## Corrections relative to the original slides

- Momentum is an exponentially weighted sum of **all** past gradients, not the
  current plus β times the previous gradient.
- The slide titled "Adagrad" showed RMSProp; both are now given.
- Adam's default β₂ is 0.999 (Kingma & Ba 2014), not 0.99.
- The Tobermory detail (1967) is omitted; the remaining history is standard.
