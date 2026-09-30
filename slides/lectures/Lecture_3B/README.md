# Lecture 3B: Convolutional Neural Networks

`Lecture_3B.qmd` is the native Quarto source (27 slides). The storyline follows
chapter 5 ("AlexNet", from p. 187) of *The Welch Labs Illustrated Guide to AI*: a
kernel is a similarity template; AlexNet's first-layer kernels are edge and colour
detectors; activation maps are stacked and processed again ("stack and repeat");
deeper layers respond to corners and eventually to faces; the last layers form an
embedding space; and the 2012 breakthrough was one of scale. The MNIST, ImageNet
and AlexNet material of the imported `../Lecture_3/` deck is included.

## Present

From the repository root:

```sh
make -C slides html DECK=Lecture_3B
quarto preview slides/lectures/Lecture_3B/Lecture_3B.qmd --port 4194
make -C slides pdf DECK=Lecture_3B
```

## Environment (notebook and figures)

From this folder:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The first use of AlexNet downloads the torchvision weights (~240 MB) to
`~/.cache/torch`. Test images are the public-domain / CC0 samples bundled with
scikit-image.

## Marimo notebook: AlexNet under the hood

```sh
.venv/bin/marimo edit AlexNet_under_the_hood.py     # or: marimo run …
```

Sections: predictions for a sample or an uploaded photo; the 64 first-layer
kernels and their activation maps; activation maps of any layer and channel;
occlusion sensitivity; feature visualization by gradient ascent on the input;
the fc7 embedding space; exercises. Everything runs on a laptop CPU.
`notebooks/exports/AlexNet_under_the_hood.html` is a static export
(`marimo export html AlexNet_under_the_hood.py -o notebooks/exports/AlexNet_under_the_hood.html`).

## Four interactive widgets

1. **conv.html** — slide a 3 × 3 kernel over a drawable 28 × 28 image; preset or
   editable kernels, stride, padding, ReLU; the nine products at each position.
2. **stack.html** — conv → ReLU → max-pool → conv: four hand-set edge kernels,
   then four 3 × 3 × 4 corner kernels that combine two edge maps; hover a corner
   unit to see its 8 × 8 receptive field.
3. **mnist.html** ("Train a network, then draw for it") — draw a digit for a trained
   CNN (2 × conv 3×3 + pool, dense; 5 258 weights). Pick the weights after 0, 1, 2 or
   3 epochs; the widget shows the preprocessed input, all feature maps, the class
   probabilities and the test accuracy per epoch.
4. **mnist-shift.html** ("Shift the digits") — a test digit moved to the right, the
   CNN's prediction for it, and accuracy against shift for the CNN and a dense
   784–128–10 network (2 000 test digits).

They share `widgets/plot.js` and `widgets/lab.css`; conv and stack use
`widgets/images.js` (procedural test images). The MNIST widgets only run inference:
`widgets/mnist-net.js` is the forward pass and `widgets/data_mnist.js` (144 KB) holds
the weights, accuracies and shift curves, written by

```sh
julia --project=slides/lectures/Lecture_3B/pluto slides/lectures/Lecture_3B/scripts/make_mnist_data.jl
node slides/lectures/Lecture_3B/scripts/check_mnist_widget.mjs   # JS logits = Flux logits
```

(about half a minute on a laptop CPU; MNIST from `~/.cache/mnist/MNIST/raw`).
Posters: `zsh scripts/make_posters.zsh`.

## Live Julia (Pluto)

`pluto/mnist-draw.jl` trains the same networks live in Flux, one epoch per click; it is
no longer embedded in the slides. `pluto/mnist-trees.jl` has the boosted trees of the
shift table. Serve both with `make -C slides start-julia-server DECK=Lecture_3B`; see
`pluto/README.md`.

## Figures

- `scripts/make_figures.py` (the venv above): all AlexNet figures, computed from the
  pretrained torchvision weights — first-layer kernels, the kernel-template
  experiment, conv1 maps of a cat, conv2 maps of a square (channel 94 = corners),
  conv5 channel 174 (faces), feature visualizations, occlusion, embedding
  similarity, test-image predictions — plus the MNIST sample grid (downloads the
  MNIST test set, ~11 MB). Key numbers are saved in `figures/results.json`.
- `scripts/make_diagrams.py` (no dependencies): the AlexNet architecture and the
  convolution-layer diagram as editable SVG.
- `figures/imagenet-errors.svg` is shared with Lecture 3A (generated there).

## Notes on the AlexNet version

All figures and the notebook use the torchvision AlexNet (64-192-384-256-256
kernels, no local response normalization, 61.1 M parameters). The 2012 network
had 96-256-384-384-256 kernels split over two GPUs. The book's face unit (map
186) and corner unit (map 104) were found on the author's webcam images; with
the scikit-image test images the corresponding units here are conv5 channel 174
and conv2 channel 94.
