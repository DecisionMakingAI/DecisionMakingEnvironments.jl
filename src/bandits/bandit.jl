
abstract type AbstractBanditProblem <: AbstractDecisionMakingProblem end


struct BanditProblem{TA,TR,TM} <: AbstractBanditProblem where {TA, TR, TM}
    A::TA
    r::TR
    meta::TM
end

struct ContextualBanditProblem{TX,TA,TD,TR,TM} <: AbstractBanditProblem where {TX, TA, TD, TR, TM}
    X::TX
    A::TA
    d::TD
    r::TR
    meta::TM
end

function sample(prob::BanditProblem, a)
    return prob.r(a)
end

function sample_objective(prob::BanditProblem, π)
    d = π()
    a = rand(d)
    return sample(prob, a)
end

function sample(prob::ContextualBanditProblem, x, a)
    r = prob.r(x,a)
    x = prob.d()
    return x, r
end

function sample(prob::ContextualBanditProblem)
    x = prob.d()
    return x
end

function sample!(prob::ContextualBanditProblem, x, a)
    r = prob.r(x,a)
    x .= prob.d()
    return r
end

function sample_objective(prob::ContextualBanditProblem, π)
    x = prob.d()
    d = π(x)
    a = rand(d)
    return prob.r(x, a)
end

function meta_information(prob::AbstractBanditProblem)
    return prob.meta
end

include("simplebandits.jl")
