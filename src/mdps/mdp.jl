
abstract type AbstractMDP <: AbstractDecisionMakingProblem end


struct SequentialProblem{TS,TX,TA,TP,TD,TM,TF} <: AbstractMDP where {TS, TX, TA, TP, TR, TG, TD, TM, TF}
    S::TS
    X::TX
    A::TA
    p::TP
    d0::TD
    meta::TM
    render::TF
end

function sample(prob::SequentialProblem, s, a)
    ret = prob.p(s,a)
    return ret
end

function sample_trajectory!(τ, prob::SequentialProblem, π)
    s, x = prob.d0()
    done = false
    while !done
        d = π(x)
        a = rand(d)
        logp = logpdf(d,a)
        s′, x′, r, γ = sample(prob,s,a)
        push!(τ, x, a, logp, r)
        if γ == 0
            done = true
            finish!(τ)
        end
        s = s′
        x = x′
    end
end


function sample_objective(prob::SequentialProblem, π)
    s, x = prob.d0()
    G = 0.0
    γt = 1.0
    done = false
    while !done
        d = π(x)
        a = rand(d)
        logp = logpdf(d,a)
        s′, x′, r, γ = sample(prob,s,a)
        G += γt * r
        γt *= γ
        if γ == 0
            done = true
        end
        s = s′
        x = x′
    end
    return G
end

function meta_information(prob::AbstractMDP)
    return prob.meta
end

function render(prob::AbstractMDP, state, cleardata=false)
    return prob.render(state, cleardata)
end


include("chains.jl")
include("cartpole.jl")
