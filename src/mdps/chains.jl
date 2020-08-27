using StaticArrays
using Distributions
using SparseArrays



function create_simple_chain(num_states::Int; stochastic=false, failchance=0.1)
    S = 1:num_states
    A = 1:2

    if !stochastic
        # function : current state s, current action a, state space S maps to next state s2
        function chain_transition(s,a,S)
            if a == 1  # go left
                s2 = s-1 # decrease state value by 1
            elseif a == 2 # go right
                s2 = s+1 # increase state value by 1
            else
                error("action must be {1,2}, but was : ", action)
            end
            if s2 > maximum(S)  # if s2 is larger than the maximum state
                s2 = minimum(S) # set it to the start state (smallest state)
            elseif s2 < minimum(S) # if s2 is smaller than smallest state
                s2 = minimum(S) # make it the smallest state
            end
            return s2  # return next state
        end
        p = (s,a)->chain_transition(s,a,S)  # mask out S in above function
    else
        nS, nA = length(S), length(nA)
        P = zeros((nS, nA, nS))
        for s in S
            for a in A
                if a == 1
                    s2 = minimum(s-1, minimum(S))
                    P[s2, a, s] = 1.0 - failchance
                    P[s,a,s] += failchance / 2.0
                    s3 = s < maximum(S) ? s+1 : minimum(S)
                    P[s3,a,s] += failchance / 2.0
                else
                    s2 = s < maximum(S) ? s+1 : minimum(S)
                    P[s2, a, s] = 1.0 - failchance
                    P[s,a,s] += failchance / 2.0
                    s3 = minimum(s-1, minimum(S))
                    P[s3,a,s] += failchance / 2.0
                end
            end
        end
        P = sparse(P)
        p = (s,a)->Categorical(P[:, a, s])
    end
    d0 = ()->minimum(S)
    m = MDP(S,A,p,d0)
    return m
end

function create_minimum_time_chain_task(m)
    r = (s,a,s′)-> -1.0
    γ = (s,a,s′)-> (s==maximum(m.S) && s′==minimum(m.S)) ? 0.0 : 1.0
    task = RLTask(m, r, γ)
    return task
end
