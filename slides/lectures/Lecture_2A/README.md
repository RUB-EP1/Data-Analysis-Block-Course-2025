# Lecture 2A: Decision Trees

This is the third course lecture, following Lecture 1A and Lecture 1B.
`Lecture_2A.qmd` is the native editable Quarto source. The visual style matches
Lecture 1B. The archived Keynote remains unchanged.

## Present

From the repository root:

```sh
make -C slides html DECK=Lecture_2A
quarto preview slides/lectures/Lecture_2A/Lecture_2A.qmd --port 4192
make -C slides pdf DECK=Lecture_2A
```

The 21 slides cover every topic in the 13-slide original, including the original
sample and quiz. Text, equations, tables, and the tree diagram are editable.
Original scientific plots remain image assets. New scientific figures come from
Julia calculations; no slide uses a full-slide screenshot.

## Four interactive widgets

1. **Cuts:** efficiency, purity, and the ROC operating point, with adjustable class prevalence.
2. **Split gain:** Gini versus entropy and a search for the best threshold.
3. **Depth and bagging:** training/validation points, tree depth, and ensemble size.
4. **AdaBoost:** incoming event weights, individual stumps, and the cumulative vote.

Widgets run locally without a Julia server or network connection. They read
`widgets/data.js`, generated from Julia. Static posters preserve their initial
states in PDF exports. Static notebook exports may require access to Pluto's
frontend CDN; live notebooks run locally through the launcher below.

## Python / Marimo CartPole notebook

From this lecture folder, create the environment and install dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Open the notebook in the Marimo editor:

```sh
.venv/bin/marimo edit CartPole_decision_trees_and_bagging.py
```

For the app view, replace `edit` with `run`. To use `python` and `marimo`
directly in your terminal, first run `source .venv/bin/activate`.

## Julia notebooks

See [notebooks/README.md](notebooks/README.md). The environment and manifest are
local to this lecture. The regression notebook is a repaired working copy of
`Data-Analysis-Block-Course-2026/homework_solutions/julia_(old)/3_decision_trees.jl`.
The archived exercise and solution files remain unchanged.

The two classification notebooks provide self-contained versions of the cuts,
impurity, classification, bagging, and boosting material in the original lecture.
They use synthetic data and do not need the external ROOT datasets required by
the older physics exercises.

## Mathematical conventions

- Half-Gini `p(1-p)` follows the source lecture. Standard binary Gini is twice this.
- Split gain weights both children by their event counts or nonnegative weights.
- AdaBoost uses ±1 labels, alpha = 0.5 log((1-error)/error), and the symmetric
  exponential weight update consistently.
- Bagging widgets use full-size bootstrap samples and both features at each split.
  Feature subsampling is discussed separately as a random-forest option.
- Seeds 21/22 provide independent training/validation data; labels have 8% flips.
- New results are newly calculated examples, not the numerical values in old screenshots.

## Source coverage

Original 1–2: title/tree; 3: cuts + widget; 4–6: objectives, Gini, split + widget;
7–8: evaluation/overtraining + widget; 9: bagging; 10: boosting + widget;
11: closing; 12–13: original sample and quiz. Julia regression and notebook links
are additions. Original scientific plot assets were extracted from a Keynote PDF export.
