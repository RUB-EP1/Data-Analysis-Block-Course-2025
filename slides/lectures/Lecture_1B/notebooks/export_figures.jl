# Run the Lecture 1B notebook calculations and write the slide figures.
# Defaults match the Pluto notebooks: a = 1.1, μ = 1, σ = 0.3.
# The entropy plot also draws the reference density (the notebook checkbox).

using Plots
using Distributions
using QuadGK
using Statistics
using Random

const FIGDIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGDIR)

gr()
theme(:boxed;
    titlefontsize = 20,
    guidefontsize = 16,
    tickfontsize = 14,
    legendfontsize = 14,
    dpi = 140,
)

function save_slide(plt, name)
    path = joinpath(FIGDIR, name)
    savefig(plt, path)
    println("wrote ", path)
    return path
end

# --- lecture-1B-mse.jl ---

const p_gen = (; a = 1.0)
f(x; p) = 3x^2 - p.a

Random.seed!(1)
data = let
    xv = range(-2, 2, 30)
    yv = f.(xv; p = p_gen) .+ randn(length(xv))
    (; xv, yv)
end

mse(pars, xv, yv) = mean(abs2, yv .- f.(xv; p = pars))

function mse_figure(; a = 1.1)
    xv, yv = data.xv, data.yv
    loss = mse((; a), xv, yv)
    plt = plot(;
        ylim = (-6, 10),
        xlab = "log(flat area)",
        ylab = "price",
        title = "Hypothetical flat-price problem",
        size = (980, 640),
        legend = :top,
    )
    scatter!(plt, xv, yv; label = "data")
    plot!(plt, x -> f(x; p = (; a)), xv; lw = 3, label = "fit, MSE = $(round(loss; digits = 3))")
    return plt
end

save_slide(mse_figure(), "mse-fit.png")

# --- lecture-1B-entropy.jl ---

cross_entropy_numerical(d, d_ref, a = d_ref.μ - 5d_ref.σ, b = d_ref.μ + 5d_ref.σ) =
    -quadgk(a, b) do x
        q = max(pdf(d_ref, x), nextfloat(0.0))
        p = max(pdf(d, x), nextfloat(0.0))
        q * log(p)
    end[1]

dKL(d, d_ref; nSample = 100) = let
    sample_p = rand(d_ref, nSample)
    -mean(x -> logpdf(d, x) - logpdf(d_ref, x), sample_p)
end

dKL_numerical(d, d_ref, a = d_ref.μ - 5d_ref.σ, b = d_ref.μ + 5d_ref.σ) =
    cross_entropy_numerical(d, d_ref, a, b) -
    cross_entropy_numerical(d_ref, d_ref, a, b)

function entropy_figure(; μ = 1.0, σ = 0.3)
    p = Normal(2.0, 0.5)
    q = Normal(μ, σ)
    cS = cross_entropy_numerical(q, p)
    plt = plot(;
        ylim = (0, 4),
        xlim = (0, 5),
        xlab = "x",
        title = "PDF",
        size = (980, 640),
        legend = :topright,
    )
    plot!(plt, x -> pdf(q, x), 0, 5; fill = 0, fillalpha = 0.35, lw = 2,
        label = "entropy \$S_q\$ = $(round(entropy(q); digits = 2))")
    plot!(plt, x -> pdf(p, x), 0, 5; fill = 0, fillalpha = 0.35, lw = 2,
        label = "cross entropy \$H(p,q)\$ = $(round(cS; digits = 2))")
    return plt
end

function likelihood_figure(; μ = 1.0, σ = 0.3)
    p = Normal(2.0, 0.5)
    q = Normal(μ, σ)
    xlim = (-3, 13)
    bins = range(xlim..., 101)
    Random.seed!(1)
    plt = plot(;
        title = "Test statistic under each hypothesis",
        xlab = "Δ NLL",
        xlim,
        size = (980, 640),
        legend = :topright,
    )
    stephist!(plt, map(_ -> dKL(q, p), 1:1_000); bins, fill = 0, label = "\$t\\,|\\,p\$")
    stephist!(plt, map(_ -> dKL(p, q), 1:1_000); bins, fill = 0, label = "\$t\\,|\\,q\$")
    vline!(plt, [dKL_numerical(q, p)]; lw = 3, label = "\$D_{\\mathrm{KL}}(p\\,||\\,q)\$")
    vline!(plt, [dKL_numerical(p, q)]; lw = 3, label = "\$D_{\\mathrm{KL}}(q\\,||\\,p)\$")
    return plt
end

save_slide(entropy_figure(), "entropy-densities.png")
save_slide(likelihood_figure(), "likelihood-ratio.png")

println("mse = ", round(mse((; a = 1.1), data.xv, data.yv); digits = 4))
q = Normal(1.0, 0.3)
p = Normal(2.0, 0.5)
println("S_q = ", round(entropy(q); digits = 4))
println("H(p,q) = ", round(cross_entropy_numerical(q, p); digits = 4))
println("D_KL(p||q) = ", round(dKL_numerical(q, p); digits = 4))
println("D_KL(q||p) = ", round(dKL_numerical(p, q); digits = 4))

function gini_impurity_figure()
    p = range(0, 1, length = 201)
    plt = plot(;
        xlab = "signal fraction p",
        ylab = "Gini impurity",
        title = "G(p) = p(1 − p)",
        xlim = (0, 1),
        ylim = (0, 0.30),
        size = (1100, 720),
        legend = false,
        left_margin = 8Plots.mm,
        bottom_margin = 6Plots.mm,
    )
    plot!(plt, p, p .* (1 .- p); lw = 3)
    scatter!(plt, [0, 0.5, 1], [0, 0.25, 0]; ms = 7)
    return plt
end

save_slide(gini_impurity_figure(), "gini-impurity.png")
