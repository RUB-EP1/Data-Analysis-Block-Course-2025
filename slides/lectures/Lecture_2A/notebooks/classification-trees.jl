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

# ╔═╡ 280dc575-0fd4-4188-a076-19bdd2e78281
begin
    import Pkg
    Pkg.activate(@__DIR__)
    using PlutoUI, Plots, Random, Statistics, Distributions
end

# ╔═╡ f2845bac-a600-4a28-b132-d14903c854a1
md"""# Decision trees, bagging, and AdaBoost
The training and validation samples use separate fixed seeds. Tune on validation; reserve a third sample for final assessment. Bagging here considers both features at every split."""

# ╔═╡ 6d987e7b-8391-4fc3-9bc6-2a0c9c3271a3
md"""## Algorithms in plain Julia
The functions below generate the data, try candidate tree splits, and combine predictions. Each split is a midpoint between observed feature values. Bagging trains on bootstrap samples and uses all features; AdaBoost reweights events after each stump. We use the convention Gini = p(1-p)."""

# ╔═╡ f1d6593f-7aef-4b55-9a26-6864d2d13576
function sample_data(n, seed)
    rng = MersenneTwister(seed)
    X = rand(rng, n, 2)
    y = [((X[i,1]-0.5)^2/0.075 + (X[i,2]-0.55)^2/0.13 < 1) ? 1 : -1 for i in 1:n]
    for i in 1:n
        rand(rng) < 0.08 && (y[i] *= -1)
    end
    (; X, y)
end

# ╔═╡ 152108ff-f368-4e1a-a966-e9f21c07a3a7
function impurity(p, criterion=:gini)
    if criterion == :gini
        return p * (1 - p)
    elseif p == 0 || p == 1
        return 0.0
    else
        return -p * log(p) - (1 - p) * log1p(-p)
    end
end

# ╔═╡ d986e9f3-af1d-4984-8735-09552c8511ec
function split_gain(y, left; weights=ones(length(y)), criterion=:gini)
    total = sum(weights)
    total > 0 || return 0.0
    wl, wr = sum(weights[left]), sum(weights[.!left])
    (wl == 0 || wr == 0) && return 0.0
    p = sum(weights[y .== 1])/total
    pl = sum(weights[left .& (y .== 1)])/wl
    pr = sum(weights[(.!left) .& (y .== 1)])/wr
    impurity(p,criterion) - (wl*impurity(pl,criterion)+wr*impurity(pr,criterion))/total
end

# ╔═╡ 621da496-2959-4430-aa0f-eaf9c0eb0d5f
function fit_tree(X, y; depth=3, minleaf=3, weights=ones(length(y)), criterion=:gini)
    @assert size(X,1) == length(y) == length(weights) > 0
    p = sum(weights[y .== 1])/sum(weights)
    node = Dict{String,Any}("p"=>p, "n"=>length(y), "value"=>(p >= 0.5 ? 1 : -1))
    (depth == 0 || length(y) < 2minleaf || p == 0 || p == 1) && return node
    gain, feature, cut = -Inf, 0, 0.0
    for j in axes(X,2)
        vals=sort(unique(X[:,j]))
        for k in 1:length(vals)-1
            t=(vals[k]+vals[k+1])/2
            left=X[:,j] .< t
            min(count(left),count(.!left)) < minleaf && continue
            g=split_gain(y,left;weights,criterion)
            if g > gain + 1e-14
                gain,feature,cut=g,j,t
            end
        end
    end
    (feature == 0 || gain <= 1e-14) && return node
    left=X[:,feature] .< cut
    node["feature"]=feature;node["cut"]=cut;node["gain"]=gain
    node["left"]=fit_tree(X[left,:],y[left];depth=depth-1,minleaf,weights=weights[left],criterion)
    node["right"]=fit_tree(X[.!left,:],y[.!left];depth=depth-1,minleaf,weights=weights[.!left],criterion)
    node
end

# ╔═╡ 777547e0-588c-41bb-88d5-357cbb4869d5
function predict_tree(tree, x)
    if !haskey(tree, "feature")
        return tree["value"]
    end
    if x[tree["feature"]] < tree["cut"]
        return predict_tree(tree["left"], x)
    else
        return predict_tree(tree["right"], x)
    end
end

# ╔═╡ 7ff5b0c4-596c-4088-8854-d7f2f8793444
accuracy(pred,y) = mean(pred .== y)

# ╔═╡ e0b5936c-274d-406f-8c77-6a72a0c96062
function forest(X, y; ntrees=30, depth=6, seed=404)
    rng = MersenneTwister(seed)
    trees = Dict{String,Any}[]
    for _ in 1:ntrees
        # Draw a bootstrap sample, with replacement, of the original size.
        ids = rand(rng, 1:length(y), length(y))
        tree = fit_tree(X[ids, :], y[ids]; depth, minleaf=2)
        push!(trees, tree)
    end
    return trees
end

# ╔═╡ ddd21973-5da4-4fa9-aa2e-077a8babb720
function fit_boost(X,y; rounds=20)
    w=fill(1/length(y),length(y)); result=Dict{String,Any}[]
    for m in 1:rounds
        besterr=Inf; best=nothing
        for j in axes(X,2)
            v=sort(unique(X[:,j])); cuts=vcat(v[1]-eps(),(v[1:end-1].+v[2:end])./2,v[end]+eps())
            for t in cuts, polarity in (-1,1)
                pred=ifelse.(X[:,j] .< t,polarity,-polarity)
                err=sum(w[pred .!= y])
                if err < besterr
                    besterr=err; best=(j,t,polarity,pred)
                end
            end
        end
        besterr >= 0.5-1e-12 && break
        j,t,polarity,pred=best
        α=0.5*log((1-clamp(besterr,1e-12,1-1e-12))/clamp(besterr,1e-12,1-1e-12))
        before=copy(w)
        w .*= exp.(-α .* y .* pred); w ./= sum(w)
        push!(result,Dict("feature"=>j,"cut"=>t,"polarity"=>polarity,"alpha"=>α,"error"=>besterr,"before"=>before,"after"=>copy(w)))
        besterr <= 1e-12 && break
    end
    result
end

# ╔═╡ 220868db-61b4-4e0f-987e-23bb33807617
function boost_predict(rounds, x)
    score = 0.0
    for stump in rounds
        prediction = if x[stump["feature"]] < stump["cut"]
            stump["polarity"]
        else
            -stump["polarity"]
        end
        score += stump["alpha"] * prediction
    end
    return score >= 0 ? 1 : -1
end

# ╔═╡ 4dd8f266-1cef-40da-8b04-25738a66b348
@bind depth Slider(1:10; default=3, show_value=true)

# ╔═╡ 9021ae28-da62-4591-8364-408e6dd8ee04
@bind ntrees Slider([1,5,10,20,30]; default=30, show_value=true)

# ╔═╡ 9f29e083-57b1-45a6-b062-d2b76a5d36e1
@bind rounds Slider(1:20; default=5, show_value=true)

# ╔═╡ 4d98d0af-fcc4-4b1d-b76b-ec20c17a76bd
begin
    train = sample_data(180,21)
    validation = sample_data(360,22)
end

# ╔═╡ 51663bd4-f27e-47f4-9ad8-61de5415f73c
model = fit_tree(train.X,train.y;depth,minleaf=2)

# ╔═╡ 4fadb5d5-17eb-423b-a2e7-02db5d2d44fe
bag = forest(train.X,train.y;depth,ntrees)

# ╔═╡ 941ed353-2b5b-4254-8063-8e759817338b
boost = fit_boost(train.X,train.y;rounds)

# ╔═╡ 8f5efd19-d8eb-4221-99fb-aee7b6eb758a
scores = (
    tree_train=accuracy([predict_tree(model,x) for x in eachrow(train.X)],train.y),
    tree_validation=accuracy([predict_tree(model,x) for x in eachrow(validation.X)],validation.y),
    bag_validation=accuracy([sum(predict_tree(t,x) for t in bag)>=0 ? 1 : -1 for x in eachrow(validation.X)],validation.y),
    boost_validation=accuracy([boost_predict(boost,x) for x in eachrow(validation.X)],validation.y))

# ╔═╡ 08f15a13-4f3c-4f2b-a15e-643894f8bd34
let
    g=range(0,1;length=81)
    p = heatmap(g,g, [predict_tree(model,[x,y]) for y in g,x in g];
        c=cgrad(["#d9e5ef","#ffe4a3"]),
        clims=(-1,1),
        colorbar=false,
        xlabel="feature 1",
        ylabel="feature 2",
        title="Tree depth = $depth")
    scatter!(p,train.X[:,1],train.X[:,2];
             marker_z=train.y,c=cgrad(["#253b53","#d99700"]),
             clims=(-1,1),label="training data",markersize=4,
                    markerstrokewidth=0)
end

# ╔═╡ 9d66270a-4726-4c4f-ab09-015316a8280d
md"""## Interpretation
Compare training and validation accuracy as depth changes. More trees reduce sampling variability but do not guarantee better validation accuracy. In AdaBoost, inspect `boost[end]["before"]` and `boost[end]["after"]`: misclassified events receive more weight."""

# ╔═╡ Cell order:
# ╠═280dc575-0fd4-4188-a076-19bdd2e78281
# ╟─f2845bac-a600-4a28-b132-d14903c854a1
# ╟─6d987e7b-8391-4fc3-9bc6-2a0c9c3271a3
# ╠═f1d6593f-7aef-4b55-9a26-6864d2d13576
# ╠═152108ff-f368-4e1a-a966-e9f21c07a3a7
# ╠═d986e9f3-af1d-4984-8735-09552c8511ec
# ╠═621da496-2959-4430-aa0f-eaf9c0eb0d5f
# ╠═777547e0-588c-41bb-88d5-357cbb4869d5
# ╠═7ff5b0c4-596c-4088-8854-d7f2f8793444
# ╠═e0b5936c-274d-406f-8c77-6a72a0c96062
# ╠═ddd21973-5da4-4fa9-aa2e-077a8babb720
# ╠═220868db-61b4-4e0f-987e-23bb33807617
# ╠═4d98d0af-fcc4-4b1d-b76b-ec20c17a76bd
# ╠═51663bd4-f27e-47f4-9ad8-61de5415f73c
# ╠═4fadb5d5-17eb-423b-a2e7-02db5d2d44fe
# ╠═941ed353-2b5b-4254-8063-8e759817338b
# ╠═8f5efd19-d8eb-4221-99fb-aee7b6eb758a
# ╠═9f29e083-57b1-45a6-b062-d2b76a5d36e1
# ╠═4dd8f266-1cef-40da-8b04-25738a66b348
# ╠═9021ae28-da62-4591-8364-408e6dd8ee04
# ╠═08f15a13-4f3c-4f2b-a15e-643894f8bd34
# ╟─9d66270a-4726-4c4f-ab09-015316a8280d
