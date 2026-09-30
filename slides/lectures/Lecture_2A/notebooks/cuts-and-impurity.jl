### A Pluto.jl notebook ###
# v0.20.28

using Markdown
using InteractiveUtils

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


# ╔═╡ 28ad5c56-12cc-44b0-a7fd-c132ab587ae4
begin
    import Pkg
    Pkg.activate(@__DIR__)
    using PlutoUI, Plots, Random, Statistics, Distributions
end

# ╔═╡ 28383e46-d232-4264-8e9d-e63c701f5ec3
md"""# Cuts and impurity
Signal follows Beta(2,5), background Beta(5,2). Accept events with x < threshold. Change the class mixture to see why purity depends on prevalence."""

# ╔═╡ 63acf31b-9d25-482b-af6b-b7ca45ba718d
md"""## How the cut is evaluated
Efficiency and rejection are sample fractions. Purity also depends on the signal fraction before the cut. Split gain subtracts the weighted child impurities from the parent impurity. We use the convention Gini = p(1-p)."""

# ╔═╡ c51902c6-48d0-4313-9f66-dbf274f542ad
function cut_metrics(signal, background, threshold; fraction=0.5)
    efficiency = mean(signal .< threshold)
    background_efficiency = mean(background .< threshold)
    selected = fraction * efficiency + (1 - fraction) * background_efficiency
    purity = selected == 0 ? 0.0 : fraction * efficiency / selected
    return (; efficiency, rejection=1 - background_efficiency, purity)
end

# ╔═╡ 9a646bfa-9067-4867-9677-d52ae506bff6
function impurity(p, criterion=:gini)
    if criterion == :gini
        return p * (1 - p)
    elseif p == 0 || p == 1
        return 0.0
    else
        return -p * log(p) - (1 - p) * log1p(-p)
    end
end

# ╔═╡ 330580e0-5990-4fd9-a4cc-6b9eabe5828b
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

# ╔═╡ c072dc85-63ee-466f-927c-b44f3950c9ad
@bind threshold Slider(0:0.01:1; default=0.5, show_value=true)

# ╔═╡ c49cd0f3-46cf-4b99-885d-b91f70c6f54a
@bind fraction Slider(0.1:0.1:0.9; default=0.5, show_value=true)

# ╔═╡ c5d35fce-0e06-445d-9966-f363e5e4e6b2
begin
    signal = rand(MersenneTwister(42), Beta(2,5), 1000)
    background = rand(MersenneTwister(43), Beta(5,2), 1000)
end

# ╔═╡ 900e2409-9625-4bef-81c4-6581dbaf13bb
metrics = cut_metrics(signal, background, threshold; fraction)

# ╔═╡ 3b5ab787-4115-4865-8252-6425546b10cb
let
    p=plot(x->pdf(Beta(2,5),x),0,1;label="signal",xlabel="x",ylabel="density",lw=3)
    plot!(p,x->pdf(Beta(5,2),x),0,1;label="background",lw=3)
    vline!(p,[threshold];label="accept x < cut",lw=2)
end

# ╔═╡ 25f20e00-3f22-4466-b6f7-20d7f268f238
let
    cuts=range(0,1;length=101)
    m=[cut_metrics(signal,background,t;fraction) for t in cuts]
    plot([a.efficiency for a in m],[a.rejection for a in m];xlabel="signal efficiency",ylabel="background rejection",label="ROC",lw=3)
    scatter!([metrics.efficiency],[metrics.rejection];label="selected cut")
end

# ╔═╡ 389d3f5f-cbba-4d3e-8126-61de782fb8e4
let
    y=vcat(fill(1,length(signal)),fill(-1,length(background)))
    x=vcat(signal,background)
    w=vcat(fill(fraction/length(signal),length(signal)),fill((1-fraction)/length(background),length(background)))
    cuts=range(0,1;length=101)
    plot(cuts,[split_gain(y,x .< t;weights=w) for t in cuts];label="Gini gain",xlabel="cut",ylabel="weighted impurity decrease",lw=3)
    plot!(cuts,[split_gain(y,x .< t;weights=w,criterion=:entropy) for t in cuts];label="entropy gain",lw=3)
end

# ╔═╡ Cell order:
# ╠═28ad5c56-12cc-44b0-a7fd-c132ab587ae4
# ╠═28383e46-d232-4264-8e9d-e63c701f5ec3
# ╟─63acf31b-9d25-482b-af6b-b7ca45ba718d
# ╠═c51902c6-48d0-4313-9f66-dbf274f542ad
# ╠═9a646bfa-9067-4867-9677-d52ae506bff6
# ╠═330580e0-5990-4fd9-a4cc-6b9eabe5828b
# ╠═c072dc85-63ee-466f-927c-b44f3950c9ad
# ╠═c49cd0f3-46cf-4b99-885d-b91f70c6f54a
# ╠═c5d35fce-0e06-445d-9966-f363e5e4e6b2
# ╠═900e2409-9625-4bef-81c4-6581dbaf13bb
# ╠═3b5ab787-4115-4865-8252-6425546b10cb
# ╠═25f20e00-3f22-4466-b6f7-20d7f268f238
# ╠═389d3f5f-cbba-4d3e-8126-61de782fb8e4
