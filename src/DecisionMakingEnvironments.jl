module DecisionMakingEnvironments

import Base:length, push!
import StatsBase: sample, sample!
using Distributions

abstract type AbstractDecisionMakingProblem end
# # Examples
# ```julia-repl

# ```


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
export AbstractMDP, MDP, POMDP, SequentialProblem
export BanditExperience, ContextualBanditExperience, Trajectory
export length, push!

export create_simple_discrete_bandit, create_simple_discrete_contextualbandit
export create_simple_chain, create_simple_chain_finitetime
export create_finitetime_cartpole

include("bandits/bandit.jl")
include("mdps/mdp.jl")
include("history.jl")

end





# What is a decision making problem
# Optimization problem that looks to find a policy that maximizes return. 
# Solution space: policy that maps states (possibly a history) to a distribtuion over the action space. No restrictions here. 
# Objective function - expectation so must be able to sample. 
# need a way to sample next states
# need a method to sample rewards 
# need a way to identify if a squence has terminated and should be reset
# needs a method of getting the initial state or sampling one
# needs a discount function. Default to 1 for finite horizon. 
# state space
# action space

# meta information: 
# problem type: finite horizon, POMDP, infinite horizon, Bandit, Contextual Bandit
# 

# decision making problem: {State space, action space, transition, reward function, initial state distribution, discount function}

# sample(prob, inputs...) get next state reward, termination
# sample_trajectory(prob, pi) record sequence of state actions rewards, logprobs
# sample_objective
# meta_info
# solve(prob, alg) return policy and other algorithms outputs

# problems I want to solve:
# MDP discrete / continuous
# POMDP discrete / continuous
# 



# state space, action type, obs type
# meta-info 

# space description
# tuple each element contains upper and lower bounds