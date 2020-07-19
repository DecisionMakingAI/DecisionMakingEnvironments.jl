
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
    r::TR
    γ::TY
end

include("chains.jl")

export MDP
