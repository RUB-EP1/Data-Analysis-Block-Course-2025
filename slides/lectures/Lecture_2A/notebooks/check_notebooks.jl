ENV["GKSwstype"]="100"
using Test
include(joinpath(@__DIR__,"TreeLab.jl"))
using .TreeLab
@testset "Tree and boosting mathematics" begin
    @test impurity(0)==impurity(1)==0
    @test impurity(0.5)==0.25
    @test impurity(0.5,:entropy) ≈ log(2)
    @test split_gain([-1,-1,1,1],[true,true,false,false]) ≈ 0.25
    @test split_gain([-1,1],falses(2)) == 0
    X=reshape([0.,0.2,0.8,1.],:,1);y=[-1,-1,1,1]
    t=fit_tree(X,y;depth=1,minleaf=1)
    @test [predict_tree(t,x) for x in eachrow(X)]==y
    b=fit_boost(X,y)
    @test length(b)==1
    @test [boost_predict(b,x) for x in eachrow(X)]==y
    d=sample_data(100,21); bs=fit_boost(d.X,d.y;rounds=10)
    @test all(r->sum(r["after"])≈1,bs)
    @test all(r->0<=r["error"]<0.5,bs)
    for r in bs
        pred=ifelse.(d.X[:,r["feature"]].<r["cut"],r["polarity"],-r["polarity"])
        wrong=pred .!= d.y
        @test sum(r["after"][wrong]) ≈ 0.5 atol=1e-10
    end
end
for file in ["cuts-and-impurity.jl","classification-trees.jl","regression-trees.jl"]
    println("RUN ",file)
    m=Module(gensym(:Notebook))
    Core.eval(m, :(include(p) = Base.include($m,p)))
    Base.include(m,joinpath(@__DIR__,file))
    println("PASS ",file)
end
