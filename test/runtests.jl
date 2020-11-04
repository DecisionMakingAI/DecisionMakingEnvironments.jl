using SafeTestsets

@safetestset "Bandit Tests" begin include("bandittests.jl") end
@safetestset "MDP Tests" begin include("mdptest.jl") end
