module DecisionMakingEnvironments


abstract type AbstractEnvironment end
export AbstractEnvironment

include("bandits/bandit.jl")
include("mdps/mdp.jl")
end
