module DecisionMakingEnvironments

import Base:length, push!
import StatsBase: sample, sample!
using Distributions
using RecipesBase

abstract type AbstractDecisionMakingProblem end

"""
    sample

For a Bandit problem this samples the reward given an action and posibily context. 
For a squential decision making problem this samples the next state, observation, reward, and terminal check given an action and current state. 
"""
function sample end 
function sample! end

"""
    sample_trajectory! 
    
records a trajectory for a sequential decision making problem
"""
function sample_trajectory! end


""" 
    sample_objective    

    returns a sample of the objective function given a policy
"""
function sample_objective end

"""
    meta_information 

returns problem meta-information useful for initializing a solver
"""
function meta_information end

"""
    solve(prob::AbstractDecisionMakingProblem, alg)

solves the decision making problem using the supplied algorithm. Returns a policy and any other algorithm specific properties
"""
function solve end


export AbstractDecisionMakingProblem
export sample, sample!, sample_trajectory!, sample_objective, meta_information, solve


export AbstractBanditProblem, BanditProblem, ContextualBanditProblem
export AbstractMDP, SequentialProblem
export BanditExperience, ContextualBanditExperience, Trajectory
export length, push!, finish!, finished

include("bandits/bandit.jl")
include("mdps/mdp.jl")
include("history.jl")

end

