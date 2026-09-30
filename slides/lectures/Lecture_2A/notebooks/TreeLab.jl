module TreeLab
using Random, Statistics
export sample_data, fit_tree, predict_tree, forest, fit_boost, boost_predict, accuracy, impurity, split_gain, cut_metrics

"""Seeded two-feature classification data; the 8% label noise is intentional."""
function sample_data(n, seed)
    rng = MersenneTwister(seed)
    X = rand(rng, n, 2)
    y = [((X[i,1]-0.5)^2/0.075 + (X[i,2]-0.55)^2/0.13 < 1) ? 1 : -1 for i in 1:n]
    for i in 1:n
        rand(rng) < 0.08 && (y[i] *= -1)
    end
    (; X, y)
end
impurity(p, criterion=:gini) = criterion == :gini ? p*(1-p) : (p == 0 || p == 1 ? 0.0 : -p*log(p)-(1-p)*log1p(-p))
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
"""Small transparent CART implementation. Every candidate is a midpoint between data values."""
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
predict_tree(t,x) = haskey(t,"feature") ? predict_tree(x[t["feature"]] < t["cut"] ? t["left"] : t["right"],x) : t["value"]
accuracy(pred,y) = mean(pred .== y)
function forest(X,y; ntrees=30,depth=6,seed=404)
    rng=MersenneTwister(seed)
    [let ids=rand(rng,1:length(y),length(y)); fit_tree(X[ids,:],y[ids];depth,minleaf=2) end for _ in 1:ntrees]
end
# Bagging uses all features. This is deliberately not named a random forest.
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
boost_predict(rounds,x) = sum(r["alpha"]*(x[r["feature"]]<r["cut"] ? r["polarity"] : -r["polarity"]) for r in rounds; init=0.0) >= 0 ? 1 : -1
function cut_metrics(signal,background,t; fraction=0.5)
    es=mean(signal .< t); eb=mean(background .< t)
    selected=fraction*es+(1-fraction)*eb
    (; efficiency=es, rejection=1-eb, purity=selected==0 ? 0.0 : fraction*es/selected)
end
end
