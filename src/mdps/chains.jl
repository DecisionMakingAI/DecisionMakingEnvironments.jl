using StaticArrays
using Distributions
using SparseArrays

export create_simple_chain

function create_simple_chain(num_states::Int; stochastic=false, failchance=0.1)
    S = 1:num_states
    A = 1:2

    if !stochastic
        function chain_transition(s,a,S)
            if a == 1
                s2 = s-1
            elseif a == 2
                s2 = s+1
            else
                error("action must be {1,2}, but was : ", action)
            end
            if s2 > max(S)
                s2 = min(S)
            elseif s2 < min(S)
                s2 = min(S)
            end
            return s2
        end
        p = (s,a)->chain_transition(s,a,S)
    else
        nS, nA = length(S), length(nA)
        P = zeros((nS, nA, nS))
        for s in S
            for a in A
                if a == 1
                    s2 = min(s-1, min(S))
                    P[s2, a, s] = 1.0 - failchance
                    P[s,a,s] = failchance / 2.0
                    s3 = s < max(S) ? s+1 : min(S)
                    P[s3,a,s] = failchance / 2.0
                else
                    s2 = s < max(S) ? s+1 : min(S)
                    P[s2, a, s] = 1.0 - failchance
                    P[s,a,s] = failchance / 2.0
                    s3 = min(s-1, min(S))
                    P[s3,a,s] = failchance / 2.0
                end
            end
        end
        P = sparse(P)
        p = (s,a)->Categorical(P[:, a, s])
    end
    d0 = ()->min(S)
    m = MDP(S,A,p,d0)
    return m
end

function create_minimum_time_chain_task(m)
    r = (s,a,s′)-> -1.0
    γ = (s,a,s′)-> (s==max(m.S) && s′==min(m.S)) ? 0.0 : 1.0
    task = RLTask(m, r, γ)
    return task
end
