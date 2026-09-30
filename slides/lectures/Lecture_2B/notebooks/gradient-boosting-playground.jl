### A Pluto.jl notebook ###
# v0.20.28

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ e2d17b38-37fb-4d9a-a7b0-64c5b38bf67c
begin
    import Pkg
    Pkg.activate(@__DIR__)
    using PlutoUI, Plots, Random, Statistics, DataFrames, JLBoost
    import LossFunctions
end

# ╔═╡ 1b465b7a-95ea-43d6-88af-571d8af7e1de
md"""# Lecture 2B · Gradient boosting playground
Inspired by [Alex Rogozhnikov’s playground](https://arogozhnikov.github.io/2016/07/05/gradient_boosting_playground.html).
Every tree is trained by [JuliaHEP/JLBoost.jl](https://github.com/JuliaHEP/JLBoost.jl).
Choose a dataset, grow an ensemble, then scrub through its history.
Blue = class 0; orange = class 1. The independent holdout is called **validation** because we use it to tune the model.
"""

# ╔═╡ 7a3c4cb1-0d1f-4491-a394-cc30a7781b24
md"""## 1. From a margin to a probability
A boosted classifier adds tree outputs to a real-valued **margin** ``F(x)``.
The sigmoid converts it to ``p(y=1|x)``. We measure the average binary log loss,
``ℓ(y,F)=log(1+e^F)-yF``.

The equivalent formulas below avoid overflow for large positive or negative margins.
At the initial margin zero, both classes have probability 1/2 and the loss is ``log(2)``.
"""

# ╔═╡ 3ab32098-21bb-4368-9074-c1ba11cdc3b4
begin
    sigmoid(z) = z >= 0 ? 1 / (1 + exp(-z)) : exp(z) / (1 + exp(z))
    logloss(y, f) = mean(max(z, 0) - t*z + log1p(exp(-abs(z))) for (t,z) in zip(y,f))
end

# ╔═╡ 20115761-400b-417d-b330-92515427c0aa
md"""## 2. Gradient or Newton updates
For logistic loss, the gradient is ``g=p-y`` and the curvature is ``h=p(1-p)``.
JLBoost's built-in `LogitLogLoss` supplies both for Newton boosting. A leaf gets
``-∑g/(∑h+λ)`` before multiplication by the learning rate.

To demonstrate ordinary gradient boosting, we define `GradientOnly`: it supplies
that same gradient but sets curvature to 1. JLBoost then fits the negative gradients
``y-p`` using squared-error-style split gains and leaf averages (with the small regularizer).
This is a fitting rule, not the true second derivative of logistic loss; evaluation still uses `logloss`.
"""

# ╔═╡ f3b0da4e-a24e-448e-91db-0899f2649f45
begin
    struct GradientOnly <: LossFunctions.SupervisedLoss end
    LossFunctions.deriv(::GradientOnly, y::Number, f::Number) = sigmoid(f) - y
    LossFunctions.deriv2(::GradientOnly, y::Number, f::Number) = 1.0
end

# ╔═╡ ce7661db-4448-4890-9128-4893ab0b2c43
md"""## 3. Coordinates and synthetic data
`coordinates` rotates the two input features by an angle in radians and returns the
column table JLBoost expects. A rotated tree must use this same transformation at prediction time.
"""

# ╔═╡ 4a0525ad-6c21-462d-b14f-5696e5cb4c24
function coordinates(X, angle)
    c, s = cos(angle), sin(angle)
    DataFrame(x1=c .* X[:,1] .- s .* X[:,2], x2=s .* X[:,1] .+ c .* X[:,2])
end

# ╔═╡ 7cd9405a-88bf-433b-9e95-a2ede1790ed9
md"""`dataset` samples points uniformly in a square and assigns binary labels using
one of six geometric rules. `noise` is the probability of flipping each label.
A local random seed makes the sample reproducible. We rotate the coordinates after
assigning labels, so the whole classification problem turns together.
Training and validation use separate seeds.
"""

# ╔═╡ 2b555b86-9cbc-4bbc-8b11-3163a7b232fe
function dataset(kind, n, seed; angle=0, noise=0.05)
    rng = MersenneTwister(seed)
    X = 2rand(rng, n, 2) .- 1
    x, y = X[:,1], X[:,2]
    labels = if kind == "XOR"
        (x .* y) .> 0
    elseif kind == "Circle"
        (x.^2 .+ y.^2) .< 0.5
    elseif kind == "Rings"
        sin.(3pi .* sqrt.(x.^2 .+ y.^2)) .> 0
    elseif kind == "Stripes"
        sin.(3pi .* x) .> 0
    elseif kind == "Spiral"
        sin.(2 .* atan.(y,x) .+ 7 .* sqrt.(x.^2 .+ y.^2)) .> 0
    else
        y .> 0.35 .* sin.(3pi .* x)
    end
    labels = xor.(labels, rand(rng,n) .< noise)
    rotated = coordinates(X, angle)
    (X=Matrix(rotated), y=Float64.(labels))
end

# ╔═╡ 2d6bf854-df60-47c4-bec2-01f77a7d0226
md"""## 4. Build the ensemble one tree at a time
The first column of each history matrix stores ``F_0=0``. At step `k`:

1. Optionally choose a fresh rotation and create the training table.
2. Pass the current training margins to `jlboost` as the warm start and request **one tree**.
3. Add its predictions to the training, validation, and plotting-grid margins.
4. Save the model and rotation, then evaluate the loss at every stored stage.

JLBoost handles splitting, leaf scores, subsampling, and the learning rate `eta`.
Its returned predictions already contain `eta`, so we do not multiply by it again.
Only the training sample is fitted; validation labels are used for the loss curve.
Keeping all stages lets the inspection slider move through the ensemble without refitting.
"""

# ╔═╡ c27b0963-d0c5-4c6f-a94e-aceb00e2d999
function fit_path(train, validation, grid; rounds=40, depth=2, eta=0.2,
                  subsample=1.0, rotate=false, newton=true, seed=123)
    # JLBoost samples with the task-local RNG; keep refits reproducible.
    Random.seed!(seed)
    rng = MersenneTwister(seed+1)
    ft = zeros(length(train.y), rounds+1)
    fv = zeros(length(validation.y), rounds+1)
    fg = zeros(size(grid,1), rounds+1)
    models = Any[]
    angles = Float64[]
    for k in 1:rounds
        angle = rotate ? 2pi*rand(rng) : 0.0
        data = coordinates(train.X,angle)
        data.target = train.y
        loss = newton ? JLBoost.LogitLogLoss() : GradientOnly()
        # One round with explicit accumulated margins also permits a fresh rotation.
        model = jlboost(data, :target, [:x1,:x2], ft[:,k], loss;
            nrounds=1, max_depth=depth, eta=eta, subsample=subsample,
            lambda=1e-6, min_child_weight=1.0)
        ft[:,k+1] = ft[:,k] + predict(model,data)
        fv[:,k+1] = fv[:,k] + predict(model,coordinates(validation.X,angle))
        fg[:,k+1] = fg[:,k] + predict(model,coordinates(grid,angle))
        push!(models,model); push!(angles,angle)
    end
    (;models,angles,train=ft,validation=fv,grid=fg,
      train_loss=[logloss(train.y,ft[:,k]) for k in axes(ft,2)],
      validation_loss=[logloss(validation.y,fv[:,k]) for k in axes(fv,2)])
end

# ╔═╡ b96e2990-5724-4568-ab2b-402a9ac878d9
md"""
Dataset $(@bind kind Select(["XOR", "Circle", "Rings", "Stripes", "Spiral", "Wave"]))

Tree depth $(@bind depth Slider(1:6; default=2, show_value=true)) · learning rate $(@bind eta Slider([0.03,0.1,0.2,0.5,1.0]; default=0.2, show_value=true))

Trees $(@bind rounds Slider(10:10:150; default=40, show_value=true)) · subsample $(@bind subsample Slider(0.2:0.1:1.0; default=1.0, show_value=true))

Rotate dataset (degrees) $(@bind degrees Slider(0:15:180; default=0, show_value=true)) · random rotation per tree $(@bind rotate CheckBox(default=false))

Newton updates $(@bind newton CheckBox(default=true)) · label noise $(@bind noise Slider(0:0.025:0.2; default=0.05, show_value=true))

Data seed $(@bind seed NumberField(1:10000; default=42))
"""

# ╔═╡ 0ca78c2d-0751-4d52-9a3b-2e8711b98148
begin
    train = dataset(kind, 400, seed; angle=deg2rad(degrees), noise=noise)
    validation = dataset(kind, 1500, seed+10001; angle=deg2rad(degrees), noise=noise)
    axis_grid = range(-1.45,1.45; length=71)
    grid = hcat([x for y in axis_grid for x in axis_grid], [y for y in axis_grid for x in axis_grid])
end

# ╔═╡ 18791bb0-9b07-4d9a-be58-acd6ef74c843
path = fit_path(train, validation, grid; rounds, depth, eta, subsample, rotate, newton, seed=seed+20001)

# ╔═╡ a204cb42-afc7-4c83-8127-24bd189dfef0
md"""### Inspect the ensemble
Number of trees included $(@bind stage Slider(0:rounds; default=min(10,rounds), show_value=true))

Scale training points by negative-gradient magnitude $(@bind gradients CheckBox(default=true))

This slider reuses the fitted trees; it does not retrain the ensemble.
"""

# ╔═╡ 33eca583-2a71-4cc5-bcbd-b03cb72d727a
let
    palette = cgrad(["#2676b8", "#f7f7f7", "#ed8b23"])
    prob = reshape(sigmoid.(path.grid[:,stage+1]),length(axis_grid),length(axis_grid))'
    p = heatmap(axis_grid,axis_grid,prob; c=palette,clims=(0,1),aspect_ratio=1,
        title="P(orange) after $stage trees",xlabel="x₁",ylabel="x₂",colorbar_title="probability")
    residual = train.y - sigmoid.(path.train[:,stage+1])
    sizes = gradients ? 2 .+ 7 .* sqrt.(abs.(residual)) : fill(4.0,length(train.y))
    scatter!(p,train.X[:,1],train.X[:,2];marker_z=train.y,c=palette,clims=(0,1),
        markersize=sizes,markerstrokewidth=0.4,markerstrokecolor=:white,label="")
    l = plot(0:rounds,path.train_loss;label="training",lw=2,color="#2676b8",
        xlabel="trees",ylabel="binary log loss",title="Learning curves")
    plot!(l,0:rounds,path.validation_loss;label="validation",lw=2,color="#ed8b23")
    vline!(l,[stage];color=:gray,ls=:dash,label="selected stage")
    plot(p,l;layout=(1,2),size=(1000,420),margin=5Plots.mm)
end

# ╔═╡ a5304571-ba8b-424f-85ed-341c9e7f1381
md"""**At $(stage) trees:** training loss **$(round(path.train_loss[stage+1];digits=4))** · validation loss **$(round(path.validation_loss[stage+1];digits=4))**.

Point size represents ``|y-p|`` after the selected stage. Large points are poorly predicted; orange points request a positive change to the margin, blue points a negative one.
"""

# ╔═╡ 9eb5cc67-6505-4fca-88ac-896f0f1e73d6
let
    indices = unique(vcat(collect(1:min(5,rounds)), [max(1,stage)]))
    contributions = [path.grid[:,k+1] - path.grid[:,k] for k in indices]
    limit = max(maximum(maximum(abs,v) for v in contributions),1e-8)
    panels = [heatmap(axis_grid,axis_grid,reshape(v,length(axis_grid),length(axis_grid))';
        c=cgrad(["#2676b8","#ffffff","#ed8b23"]),clims=(-limit,limit),aspect_ratio=1,
        title="Tree $k · η × output",xlabel="x₁",ylabel="x₂",colorbar=false)
        for (k,v) in zip(indices,contributions)]
    plot(panels...;layout=(2,3),size=(1000,610))
end

# ╔═╡ 38d7930b-cc4e-42c1-b11a-89093d229584
md"""### What is being added?
The raw margin starts at ``F_0(x)=0`` and the probability is ``p=1/(1+e^{-F})``.
Each step adds ``F_m=F_{m-1}+ηt_m``. The panels above share a color scale and include the learning rate.

With **Newton updates**, JLBoost uses logistic gradients and Hessians for split gains and leaf values.
With the box off, we give JLBoost the same logistic gradient and unit curvature, producing ordinary gradient boosting on the residuals. This switch changes both split selection and leaf values, so it is an adaptation of Alex’s demo, not a numerical replica of its algorithm.
A tiny leaf regularizer (``λ=10^{-6}``) and minimum child weight 1 keep the fits well behaved.
Random tree rotations change coordinates before fitting and predicting each tree.

### Try it
- Turn noise off and fit XOR with depth 2, then depth 1. Why do stumps struggle?
- Fit stripes with stumps, then rotate the dataset by 45°.
- Increase depth and train longer on noisy data. Find where validation loss rises while training loss falls.
- Compare Newton and ordinary updates at the same learning rate.
- Fit the spiral, then enable random rotations and subsampling.

The synthetic datasets here are generated in Julia; their geometry follows the playground’s examples rather than copying its samples.
"""

# ╔═╡ Cell order:
# ╠═e2d17b38-37fb-4d9a-a7b0-64c5b38bf67c
# ╟─1b465b7a-95ea-43d6-88af-571d8af7e1de
# ╟─7a3c4cb1-0d1f-4491-a394-cc30a7781b24
# ╠═3ab32098-21bb-4368-9074-c1ba11cdc3b4
# ╟─20115761-400b-417d-b330-92515427c0aa
# ╠═f3b0da4e-a24e-448e-91db-0899f2649f45
# ╟─ce7661db-4448-4890-9128-4893ab0b2c43
# ╠═4a0525ad-6c21-462d-b14f-5696e5cb4c24
# ╟─7cd9405a-88bf-433b-9e95-a2ede1790ed9
# ╠═2b555b86-9cbc-4bbc-8b11-3163a7b232fe
# ╟─2d6bf854-df60-47c4-bec2-01f77a7d0226
# ╠═c27b0963-d0c5-4c6f-a94e-aceb00e2d999
# ╟─b96e2990-5724-4568-ab2b-402a9ac878d9
# ╠═0ca78c2d-0751-4d52-9a3b-2e8711b98148
# ╠═18791bb0-9b07-4d9a-be58-acd6ef74c843
# ╟─a204cb42-afc7-4c83-8127-24bd189dfef0
# ╠═33eca583-2a71-4cc5-bcbd-b03cb72d727a
# ╟─a5304571-ba8b-424f-85ed-341c9e7f1381
# ╠═9eb5cc67-6505-4fca-88ac-896f0f1e73d6
# ╟─38d7930b-cc4e-42c1-b11a-89093d229584
