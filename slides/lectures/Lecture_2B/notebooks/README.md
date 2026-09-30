# Gradient boosting playground

A Pluto adaptation of [Alex Rogozhnikov’s playground](https://arogozhnikov.github.io/2016/07/05/gradient_boosting_playground.html), with every tree fitted by [JuliaHEP/JLBoost.jl](https://github.com/JuliaHEP/JLBoost.jl).

From the course repository root:

```sh
julia slides/lectures/Lecture_2B/notebooks/launch.jl
```

The launcher installs the pinned environment and opens the notebook in Pluto. Julia 1.11 is recommended. First launch requires internet access. The Manifest pins the JuliaHEP fork to commit `06d99241888811722923b49dd845be5901bfdb21`; the registry package is not substituted.

Features: six synthetic datasets, reproducible samples and label noise, dataset rotation, tree depth, learning rate, tree count, subsampling, per-tree rotations, Newton/gradient updates, an ensemble-stage slider, gradient-sized points, training/validation loss, and individual tree contribution maps.

`gradient-boosting-playground.jl` contains all implementation code in visible, explained Pluto cells: probabilities and loss, the gradient fitting rule, coordinate rotation, data generation, and the boosting loop. It calls JLBoost for one round at a time with explicit accumulated margins. Each tree’s rotation is reused at prediction time. No alternate tree learner is implemented. Gradient mode supplies logistic first derivatives and unit curvature through the LossFunctions interface. Newton mode uses JLBoost's logistic loss, affecting split selection as well as leaf scores. The generated data and this optimization differ from the original JavaScript demo; this is not a numerical replica. Gradient inspection uses a stage slider instead of hover.

Run verification and regenerate the static preview:

```sh
julia --project=slides/lectures/Lecture_2B/notebooks slides/lectures/Lecture_2B/notebooks/check_notebook.jl
```

Checks cover learning, ensemble summation, rotated predictions, reproducibility, both update modes, and execution in Pluto's reactive engine. `exports/gradient-boosting-playground.html` is a static preview; controls require live Pluto.
