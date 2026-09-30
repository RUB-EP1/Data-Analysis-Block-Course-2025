# Lecture 2B: Gradient Boosting

`Lecture_2B.qmd` is the native Quarto/Reveal.js source, in the style of Lecture 2A.
It replaces `Lecture-2B_previous.pdf` (kept as reference) and covers all of its topics.

```sh
make -C slides html DECK=Lecture_2B
quarto preview slides/lectures/Lecture_2B/Lecture_2B.qmd --port 4193
make -C slides pdf DECK=Lecture_2B
```

## Storyline (27 slides)

AdaBoost recap → crocodile & turtle (step size η) → notation → regression with MSE →
additive model and algorithm → trees as piecewise-constant functions →
**leaf values** (argmin per leaf = mean residual) → **split search** (gain from
running sums) → widget 1 → **boosting loop** widget 2 → pseudo-residuals →
second-order (Newton/XGBoost) leaf values and split gain → general algorithm →
classification with **BCE** (log-odds, g = p − y, h = p(1 − p), worked example) →
MSE-vs-BCE summary → widget 3 → regularization, Huber, other losses (deviance,
exponential ↔ AdaBoost) → libraries/parameter names → impact → playground.

## Three interactive widgets (`widgets/`)

1. `split.html` — round-1 stump on residuals: cut slider, leaf values, gain curve, λ.
2. `regression.html` — MSE boosting step by step: model, residuals + tree m, train/validation MSE; η, depth.
3. `classification.html` — BCE boosting in 2D: p(x) map, contribution of tree m, loss curves; dataset, η, depth, Newton vs gradient leaves.

All three use `widgets/gb.js`, a small second-order tree learner
(leaf `−G/(H+λ)`, gain `½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−G²/(H+λ)]`). Data are generated
in the browser from fixed seeds, so no Julia/Python server is needed. Posters
(`*-poster.png`) are headless-Chrome screenshots used in PDF export; regenerate them
after changing a widget.

## Figures

`figures/huber.svg` and `figures/bce.svg` come from `node scripts/make_figures.js`.
`figures/crocodile-turtle.jpg` is extracted from the previous lecture PDF.

## Julia playground

[Gradient boosting playground](notebooks/README.md): an interactive Pluto adaptation of Alex Rogozhnikov's demo using JuliaHEP/JLBoost.jl.

```sh
julia slides/lectures/Lecture_2B/notebooks/launch.jl
```

## Conventions

- Regression loss `½(y−F)²`; residual `r = y − F = −∂L/∂F`.
- Classification: `y ∈ {0,1}`, score `F` = log-odds, `p = σ(F)`, BCE `log(1+e^F) − yF`
  (the old slide showed the log-likelihood with the opposite sign).
- Leaf values are `γ`; the XGBoost split cost is called `τ` here to avoid the clash
  with `γ` in the old slides (XGBoost's own name is `gamma`).
- Pseudo-residual carries the minus sign: `r = −∂L/∂F`.
