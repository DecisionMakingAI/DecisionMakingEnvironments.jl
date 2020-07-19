
@testset "Simple Bandit Tests" begin
    b = create_simple_discrete_bandit(3; noise=2.0)
    @test length(b.A) == 3
    @test eltype(b.A) <: Int
    @test b.r(1) isa Float64

    b = create_simple_discrete_contextualbandit(3; noise_x=1.0, noise_r=0.5)
    @test size(b.X) == (1,2)
    @test all(b.X[1, :] .== [-2.0, 2.0])
    @test length(b.A) == 3
    @test eltype(b.A) <: Int
    x = b.d()
    @test x isa Float64
    @test b.r(x, 1) isa Float64
end
