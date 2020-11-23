
abstract type AbstractMDP <: AbstractDecisionMakingProblem end


struct MDP{TS,TA,TP,TR,TG,TD,TM,TF} <: AbstractMDP where {TS, TA, TP, TR, TG, TD, TM, TF}
    S::TS
    A::TA
    p::TP
    r::TR
    γ::TG
    d0::TD
    meta::TM
    render::TF
end

struct POMDP{TS,TA,TX,TP,TO,TR,TG,TD,TM,TF} <: AbstractMDP where {TS, TA, TX, TP, TO, TR, TG, TD, TM, TF}
    S::TS
    A::TA
    X::TX
    p::TP
    obs::TO
    r::TR
    γ::TG
    d0::TD
    meta::TM
    render::TF
end

struct SequentialProblem{TS,TX,TA,TP,TD,TM,TF} <: AbstractMDP where {TS, TX, TA, TP, TR, TG, TD, TM, TF}
    S::TS
    X::TX
    A::TA
    p::TP
    d0::TD
    meta::TM
    render::TF
end


export AbstractMDP, MDP, POMDP
export create_simple_chain, create_minimum_time_chain_task
export create_finitetime_cartpole, create_cartpole_balancetask

function sample(prob::SequentialProblem, s, a)
    # s′, x, r, γ = prob.p(s,a)
    # return s′, x, r, γ
    ret = prob.p(s,a)
    return ret
end

function sample(prob::MDP, s, a)
    s′ = prob.p(s,a)
    r = prob.r(s,a,s′) 
    γ = prob.γ(s,a,s′)
    return s′, r, γ
end

function sample(prob::POMDP, s, a)
    s′ = prob.p(s,a)
    x = prob.obs(s′)
    r = prob.r(s,a,s′) 
    γ = prob.γ(s,a,s′)
    return s′, x, r, γ
end


function sample_trajectory!(prob::MDP, τ, π)
    s = prob.d0()
    done = false
    while !done
        d = π(s)
        a = rand(d)
        logp = logpdf(d,a)
        s′, r, γ = sample(prob,s,a)
        push!(τ, s, a, logp, r)
        if γ == 0
            done = true
        end
        s = s′
    end
end

function sample_trajectory!(prob::POMDP, τ, π)
    s = prob.d0()
    x = prob.obs(s)
    done = false
    while !done
        d = π(x)
        a = rand(d)
        logp = logpdf(d,a)
        s′, x′, r, γ = sample(prob,s,a)
        push!(τ, x, a, logp, r)
        if γ == 0
            done = true
        end
        s = s′
        x = x′
    end
end

function sample_trajectory!(prob::SequentialProblem, τ, π)
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
        end
        s = s′
        x = x′
    end
end

function sample_objective(prob::MDP, π)
    s = prob.d0()
    G = 0.0
    γt = 1.0
    done = false
    while !done
        d = π(s)
        a = rand(d)
        logp = logpdf(d,a)
        s′, r, γ = sample(prob,s,a)
        G += γt * r
        γt *= γ
        if γ == 0
            done = true
        end
        s = s′
    end
    return G
end

function sample_objective(prob::POMDP, π)
    s = prob.d0()
    x = prob.obs(s)
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




include("chains.jl")
include("cartpole.jl")
