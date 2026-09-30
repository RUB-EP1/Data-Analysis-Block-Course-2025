# Lecture 2A Julia notebooks

Tested with Julia 1.11.6 and the checked-in Manifest.toml. Run from the repository root:

```sh
julia --project=slides/lectures/Lecture_2A/notebooks -e 'using Pkg; Pkg.instantiate()'
julia --project=slides/lectures/Lecture_2A/notebooks slides/lectures/Lecture_2A/notebooks/launch.jl
```

The launcher opens Pluto on the classification notebook. Use Pluto's file picker
to open `cuts-and-impurity.jl` or `regression-trees.jl` from this directory.
The notebooks activate their environment relative to their own source location.
First setup downloads packages; subsequent runs reuse the pinned environment.

- `cuts-and-impurity.jl`: sampled distributions, ROC, purity, weighted impurity gain.
- `classification-trees.jl`: tree depth, bootstrap bagging, AdaBoost, validation accuracy.
- `regression-trees.jl`: repaired copy of the user's regression-tree/forest solution,
  including 1D and 2D functions and Huber loss. Placeholder code in Markdown remains
  as the original exercises; executable solution cells are complete.

The old regression example calls its holdout data “test”. During interactive
hyperparameter exploration, treat that sample as **validation**. An unbiased final
test score requires another untouched sample.

## Rebuild and verify

```sh
julia --project=slides/lectures/Lecture_2A/notebooks slides/lectures/Lecture_2A/notebooks/check_notebooks.jl
julia --project=slides/lectures/Lecture_2A/notebooks slides/lectures/Lecture_2A/notebooks/export_material.jl
julia --project=slides/lectures/Lecture_2A/notebooks slides/lectures/Lecture_2A/notebooks/export_notebooks.jl
```

`check_notebooks.jl` checks impurity, perfect splits, empty splits, perfect stumps,
weight normalization, and AdaBoost's post-update error, then runs all notebooks
as scripts. `export_notebooks.jl` additionally executes all cells in Pluto's actual
reactive engine, rejects cell errors, and writes static HTML pages to `exports/`.
Static exports display completed calculations; sliders require the live Pluto session.
Pluto's static HTML frontend loads from a CDN.

The cuts and classification notebooks define their teaching functions in visible Pluto
cells, including CART, bagging, and binary AdaBoost. `TreeLab.jl` is retained for
the figure/widget exporter; the notebooks do not import it.
The regression notebook uses the pinned DecisionTree.jl package. The widget scripts
interpret exported Julia trees rather than retraining a second implementation.

## Repairs to the regression notebook

- Replaced temporary environments and unnecessary HighEnergyTools downloads with a local project.
- Prevented zero-sized samples, empty train/holdout splits, and empty forests.
- Seeded data and estimators for reproducibility.
- Grouped model creation and fitting in one Pluto cell so downstream predictions react to refits.
- Reordered source cells for ordinary Julia execution.
- Used scatter plots for irregular 2D samples instead of treating them as a gridded surface.
- Evaluated the 2D comparison on a Cartesian grid, not only the diagonal x = y.
- Corrected explanations of feature selection, bias/variance, and ensemble guarantees.
