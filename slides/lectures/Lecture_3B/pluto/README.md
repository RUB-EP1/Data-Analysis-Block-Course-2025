# Pluto notebooks for Lecture 3B

Both notebooks read MNIST from the local raw files in `~/.cache/mnist/MNIST/raw`
(downloaded by `../scripts/make_figures.py`), so nothing is downloaded during the
lecture. Each notebook is self-contained: the reader, image shifting and the
`DrawPad()` input are in its "Appendix: helpers" cells. Both use the environment in
this folder (`Project.toml`, `Manifest.toml`; JLBoost comes from
github.com/JuliaHEP/JLBoost.jl). First setup:

```sh
julia --project=slides/lectures/Lecture_3B/pluto -e 'using Pkg; Pkg.instantiate()'
```

Serve both for the slides (port 1243, fixed notebook ids in `../pluto.toml`):

```sh
make -C slides start-julia-server DECK=Lecture_3B
```

## mnist-draw.jl — train, then draw (in the slides)

Formerly embedded in the slides; they now use the in-browser widgets `widgets/mnist.html` and `widgets/mnist-shift.html`, whose weights come from `../scripts/make_mnist_data.jl`.

A dense network (784–128–10) or a small CNN, trained with the "Train one epoch"
button (about 0.3 s per epoch for the dense network and 3 s for the CNN on a laptop
CPU, after a first epoch that includes compilation; 96 % after two epochs). Draw a
digit on the pad: it is cropped, scaled into 20 × 20 and centred by its centre of mass,
as MNIST digits are, and the bar chart shows the network's probabilities.

## mnist-trees.jl — boosted trees on pixels (not in the slides)

Opened as a full notebook, from the same server
(`http://localhost:1243/edit?id=3b0a7000-0000-4000-8000-00000074ee02`), to show that
training takes time.

Ten one-versus-rest JLBoost boosters (N = 10 000 images, 20 rounds, depth 4) reach
92.4 % test accuracy. Training takes about 1.5 minutes on 8 cores, so the result is
cached in `cache/` (git-ignored); the notebook trains once if the cache is missing,
and the "Retrain" button refits. Shows the split gain per pixel, the first tree as
if/else questions on pixels, the drawing pad, and accuracy on shifted digits.

## Shifted test digits (2 000 test images, 3 epochs for the networks)

| shift (pixels) | 0 | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|
| dense 784–128–10 | 95 % | 92 % | 71 % | 49 % | 31 % | 12 % |
| CNN | 97 % | 96 % | 92 % | 78 % | 58 % | 22 % |
| boosted trees | 90 % | 86 % | 69 % | 51 % | 33 % | 14 % |

Trees and dense networks both treat each pixel as a fixed input and fail in the same
way; the CNN shares its kernels across positions and pools, so it degrades much later.
