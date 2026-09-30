import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Pluto
Pluto.run(notebook=joinpath(@__DIR__, "gradient-boosting-playground.jl"))
