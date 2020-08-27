
abstract type AbstractBandit <: AbstractEnvironment end

struct Bandit{TA,TR} <: AbstractBandit where {TA, TR}
    A::TA
    r::TR
end

struct ContextualBandit{TX,TA,TD,TR} <: AbstractBandit where {TX, TA, TD, TR}
    X::TX
    A::TA
    d::TD
    r::TR
end


export Bandit, ContextualBandit

include("simplebandits.jl")
