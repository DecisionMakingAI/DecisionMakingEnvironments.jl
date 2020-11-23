using DecisionMakingEnvironments
import DecisionMakingEnvironments: create_simple_discrete_bandit, create_simple_discrete_contextualbandit
using Test
using Distributions

@testset "Simple Bandit Tests" begin
    b = create_simple_discrete_bandit(3; noise=2.0)
    @test length(b.A) == 3
    @test eltype(b.A) <: Int
    @test b.r(1) isa Float64
    @test meta_information(b)[:minreward] isa Float64

    b = create_simple_discrete_contextualbandit(3; noise_x=1.0, noise_r=0.5)
    @test size(b.X) == (1,2)
    @test all(b.X[1, :] .== [-2.0, 2.0])
    @test length(b.A) == 3
    @test eltype(b.A) <: Int
    x = b.d()
    @test x isa Float64
    @test x ≥ b.X[1, 1]
    @test x ≤ b.X[1, 2]
    @test b.r(x, 1) isa Float64
    @test meta_information(b)[:maxreward] isa Float64
end

@testset "Bandit Interface Tests" begin
    prob = create_simple_discrete_bandit(3; noise=0.0)
    r = sample(prob, 2)
    @test r == 2
    
    π = (x=0)->Categorical(ones(length(prob.A)) ./ length(prob.A))
    J = sample_objective(prob, π)
    @test J isa Float64
    @test J ≥ meta_information(prob)[:minreward]
    @test J ≤ meta_information(prob)[:maxreward]
   
    prob = create_simple_discrete_contextualbandit(3; noise_x=1.0, noise_r=0.0)
    x = sample(prob)
    @test x isa Float64
    @test x ≥ prob.X[1, 1]
    @test x ≤ prob.X[1, 2]
    x, r = sample(prob, x, 1)
    @test x isa Float64
    @test x ≥ prob.X[1, 1]
    @test x ≤ prob.X[1, 2]
    @test r ≥ meta_information(prob)[:minreward]
    @test r ≤ meta_information(prob)[:maxreward]
    J = sample_objective(prob, π)
    @test J isa Float64
    @test J ≥ meta_information(prob)[:minreward]
    @test J ≤ meta_information(prob)[:maxreward]
end