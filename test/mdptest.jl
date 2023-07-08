using DecisionMakingEnvironments
import DecisionMakingEnvironments: create_simple_chain, create_simple_chain_finitetime, create_finitetime_cartpole
using Test
using Distributions


@testset "Cart Pole Interface Tests" begin
    tMax = 20.0
    dt = 0.02
    prob = create_finitetime_cartpole(randomize=false, tMax=tMax, dt=dt, Atype=:Discrete, droptime=true)
    s0,x0 = prob.d0()
    @test s0 isa Tuple{Float64,Array{Float64,1}}
    @test all(x0 .== zeros(4))
    s,x,r,γ = sample(prob, s0, 1)
    @test s isa Tuple{Float64,Array{Float64,1}}
    @test x isa Array{Float64,1}
    @test r == 1.0
    @test γ isa Float64
    @test γ == 1.0
    
    s,x,r,γ = sample(prob, (20.0, zeros(4)), 2)
    @test γ == 0.0
    meta = meta_information(prob)
    @test meta[:minreturn] isa Float64
    @test meta[:minreturn] == 9.0
    @test meta[:minreward] == 1.0
    @test meta[:maxreturn] == ceil(tMax / dt)
    @test meta[:minhorizon] == 9
    @test meta[:maxhorizon] == ceil(Int, tMax / dt)
    @test meta[:discounted] == false

    π = s -> Categorical([1.0,0.0])
    τ = Trajectory(prob)
    sample_trajectory!(τ, prob, π)
    @test sum(τ.rewards) == meta[:minreturn]
    @test length(τ.rewards) == 9
    @test length(τ.states) == 9
    @test eltype(τ.states) == Array{Float64,1}
    @test eltype(τ.actions) == Int
    @test length(τ.actions) == 9
    @test length(τ.blogps) == 9
    @test eltype(τ.blogps) == Float64

    J = sample_objective(prob, π)
    @test J == 9.0
    @test J isa Float64
    @test J ≥ meta[:minreturn]
    @test J ≤ meta[:maxreturn]

end