ENV["GKSwstype"] = "100"
using Test, JLBoost
include("gradient-boosting-playground.jl")
train = dataset("XOR",400,42;noise=0)
val = dataset("XOR",500,1042;noise=0)
a = fit_path(train,val,val.X;rounds=20,depth=2)
@test all(isfinite,a.train_loss)
@test a.train_loss[end] < a.train_loss[1]
@test a.validation_loss[end] < a.validation_loss[1]
@test a.train[:,end] ≈ predict(vcat(a.models...), coordinates(train.X,0))
for newton in (false,true), rotate in (false,true)
    b = fit_path(train,val,val.X;rounds=3,newton,rotate,subsample=0.5)
    c = fit_path(train,val,val.X;rounds=3,newton,rotate,subsample=0.5)
    @test b.train == c.train
    @test all(isfinite,b.grid)
    @test b.train[:,end] ≈ sum(predict(m,coordinates(train.X,t)) for (m,t) in zip(b.models,b.angles))
end
using Pluto
session = Pluto.ServerSession()
session.options.server.disable_writing_notebook_files = true
nb = Pluto.SessionActions.open(session,joinpath(@__DIR__,"gradient-boosting-playground.jl");run_async=false)
failed = filter(c->c.errored,nb.cells)
for c in failed
    println(c.code,"\n",c.output.body)
end
@test isempty(failed)
if isempty(failed)
    mkpath(joinpath(@__DIR__,"exports"))
    write(joinpath(@__DIR__,"exports","gradient-boosting-playground.html"),Pluto.generate_html(nb))
end
Pluto.SessionActions.shutdown(session,nb)
