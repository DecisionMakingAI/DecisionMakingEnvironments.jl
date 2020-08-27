
abstract type AbstractMDP <: AbstractEnvironment end

abstract type AbstractRLTask end

struct MDP{TS,TA,TP,TD} <: AbstractMDP where {TS, TA, TP, TD}
    S::TS
    A::TA
    p::TP
    d0::TD
end

struct POMDP{TS,TA,TX,TP,TO,TD} <: AbstractMDP where {TS, TA, TX, TP, TO, TD}
    S::TS
    A::TA
    X::TX
    p::TP
    obs::TO
    d0::TD
end

struct RLTask{TM, TR, TY} <: AbstractRLTask where {TM, TR, TY}
    m::TM
    r::TR  # reward function : S x A x S -> R, reward at time t = r(S_t,A_t,S_{t+1})
    γ::TY  # discount function : S x A x S -> [0,1] discount factor at time t = γ(S_t,A_t,S_{t+1}) and when S_{t+1} is start state and S_t is terminal state disount is 0.0
end

export MDP, POMDP, RLTask
export create_simple_chain, create_minimum_time_chain_task
export create_finitetime_cartpole, create_cartpole_balancetask

include("chains.jl")
include("cartpole.jl")
