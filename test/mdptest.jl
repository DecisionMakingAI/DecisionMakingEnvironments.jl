using DecisionMakingEnvironments
using Test
using Distributions


# create_finitetime_cartpole
@testset "Simple Chain Tests" begin
    prob = create_simple_chain(3, stochastic=false)
    @test length(prob.A) == 2
    @test eltype(prob.A) <: Int
    @test length(prob.S) == 3
    @test eltype(prob.S) <: Int
    @test prob.d0() isa Int
    @test prob.d0() == 1
    @test prob.p(prob.d0(),2) isa Int
    @test prob.p(1,2) == 2
    @test prob.p(3,2) == 1
    @test prob.r(1,2,1) isa Float64
    @test prob.r(1,2,1) == -1.0
    @test prob.γ(prob.d0(),2,1) isa Float64
    @test prob.γ(3,1,2) == 1.0
    @test prob.γ(3,2,1) == 0.0
    meta = meta_information(prob)
    @test meta[:minreturn] isa Float64
    @test meta[:minreward] == -1
    @test meta[:maxreturn] == -3
    @test meta[:minhorizon] == 3
    @test meta[:maxhorizon] == Inf
    prob = create_simple_chain(3, stochastic=true)
    @test prob.p(1,2) isa Int

    prob = create_simple_chain_finitetime(3, stochastic=false, droptime=false)
    maxT = 3*20
    @test length(prob.A) == 2
    @test eltype(prob.A) <: Int
    @test length(prob.S) == 2
    @test length(prob.S[1]) == maxT
    @test length(prob.S[2]) == 3
    @test eltype(prob.S[1]) <: Int
    @test eltype(prob.S[2]) <: Int
    s0 = prob.d0()
    @test s0 isa Tuple{Int,Int}
    @test s0[1] isa Int
    @test s0[2] isa Int
    @test s0[1] == 1
    @test s0[2] == 1
    @test prob.p(prob.d0(),2) isa Tuple{Int,Int}
    @test prob.p(s0,2) == (2,2)
    @test prob.p((3,3),2) == (1,1)
    @test prob.p((maxT-1,1),2) == (maxT,2)
    @test prob.p((maxT,1),2) == (1,1)
    @test prob.r(s0,2,(2,2)) isa Float64
    @test prob.r(s0,2,(2,2)) == -1.0
    @test prob.γ((maxT-1,1),2,(maxT,1)) isa Float64
    @test prob.γ((maxT-1,1),2,(maxT,1)) == 1.0
    @test prob.γ((maxT,1),2,(1,1)) == 0.0
    @test prob.γ((1,3),2,(2,1)) == 0.0
    @test prob.γ((1,2),2,(2,3)) == 1.0
    meta = meta_information(prob)
    @test meta[:minreturn] isa Float64
    @test meta[:minreturn] == -maxT
    @test meta[:minreward] == -1
    @test meta[:maxreward] == -1
    @test meta[:maxreturn] == -3
    @test meta[:minhorizon] == 3
    @test meta[:maxhorizon] == maxT

    prob2 = create_simple_chain_finitetime(3, stochastic=false, droptime=true)
    @test prob2.S == prob.S
    @test prob2.A == prob.A
    @test prob2.X == prob2.S[2]
    x = prob2.obs(s0)
    @test x == 1
    @test x isa Int
end

@testset "Simple Chain Interface Tests" begin
    prob = create_simple_chain(3, stochastic=false)
    @test prob.d0() isa Int
    @test prob.d0() == 1
    @test prob.p(prob.d0(),2) isa Int
    @test prob.p(1,2) == 2
    @test prob.p(3,2) == 1
    @test prob.r(1,2,1) isa Float64
    @test prob.r(1,2,1) == -1.0
    @test prob.γ(prob.d0(),2,1) isa Float64
    @test prob.γ(3,1,2) == 1.0
    @test prob.γ(3,2,1) == 0.0
    meta = meta_information(prob)
    @test meta[:minreturn] isa Float64
    @test meta[:minreward] == -1
    @test meta[:maxreturn] == -3
    @test meta[:minhorizon] == 3
    @test meta[:maxhorizon] == Inf
    prob = create_simple_chain(3, stochastic=true)
    @test prob.p(1,2) isa Int

    prob = create_simple_chain_finitetime(3, stochastic=false, droptime=false)
    maxT = 3*20
    @test length(prob.A) == 2
    @test eltype(prob.A) <: Int
    @test length(prob.S) == 2
    @test length(prob.S[1]) == maxT
    @test length(prob.S[2]) == 3
    @test eltype(prob.S[1]) <: Int
    @test eltype(prob.S[2]) <: Int
    s0 = prob.d0()
    @test s0 isa Tuple{Int,Int}
    @test s0[1] isa Int
    @test s0[2] isa Int
    @test s0[1] == 1
    @test s0[2] == 1
    @test prob.p(prob.d0(),2) isa Tuple{Int,Int}
    @test prob.p(s0,2) == (2,2)
    @test prob.p((3,3),2) == (1,1)
    @test prob.p((maxT-1,1),2) == (maxT,2)
    @test prob.p((maxT,1),2) == (1,1)
    @test prob.r(s0,2,(2,2)) isa Float64
    @test prob.r(s0,2,(2,2)) == -1.0
    @test prob.γ((maxT-1,1),2,(maxT,1)) isa Float64
    @test prob.γ((maxT-1,1),2,(maxT,1)) == 1.0
    @test prob.γ((maxT,1),2,(1,1)) == 0.0
    @test prob.γ((1,3),2,(2,1)) == 0.0
    @test prob.γ((1,2),2,(2,3)) == 1.0
    meta = meta_information(prob)
    @test meta[:minreturn] isa Float64
    @test meta[:minreturn] == -maxT
    @test meta[:minreward] == -1
    @test meta[:maxreward] == -1
    @test meta[:maxreturn] == -3
    @test meta[:minhorizon] == 3
    @test meta[:maxhorizon] == maxT

    prob2 = create_simple_chain_finitetime(3, stochastic=false, droptime=true)
    @test prob2.S == prob.S
    @test prob2.A == prob.A
    @test prob2.X == prob2.S[2]
    x = prob2.obs(s0)
    @test x == 1
    @test x isa Int
end

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
    sample_trajectory!(prob, τ, π)
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