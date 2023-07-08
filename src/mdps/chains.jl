
# function : current state s, current action a, state space S maps to next state s2
function simplechain_transition(s,a,S)
    if a == 1  # go left
        s2 = s-1 # decrease state value by 1
    elseif a == 2 # go right
        s2 = s+1 # increase state value by 1
    elseif a == -1
        s2 = s
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

function simplechain_perturb_action(a::Int, failchance)
    ϵ = rand()
    if ϵ ≤ 1.0 - failchance
        return a
    elseif ϵ ≤ 1.0 - failchance / 2
        return Int(-1)
    else
        return a==1 ? 2 : 1
    end
end

function simplechain_step(s, a, S, r, γ)
    s′ = simplechain_transition(s,a,S)
    reward = r(s,a,s′)
    γt = γ(s,a,s′)
    return s′, s′, reward, γt
end



function create_simple_chain(num_states::Int; stochastic=false, failchance=0.1)
    S = 1:num_states
    A = 1:2

    r = (s,a,s′)-> -1.0
    γ = (s,a,s′)-> (s==num_states && s′==1) ? 0.0 : 1.0
    if !stochastic
        p = (s,a)->simplechain_step(s, a, S, r, γ)  # mask out S in above function    
    else
        p = (s,a)->simplechain_step(s, simplechain_perturb_action(a,failchance), S, r, γ)
    end
    d0 = ()->(minimum(S), minimum(S))
    
    meta = Dict{Symbol,Any}()
    meta[:minreward] = -1.0
    meta[:maxreward] = -1.0
    meta[:minreturn] = -Inf
    meta[:maxreturn] = -Float64(num_states)
    meta[:stochastic] = stochastic
    meta[:minhorizon] = num_states
    meta[:maxhorizon] = Inf
    meta[:discounted] = false

    m = SequentialProblem(S,S,A,p,d0,meta, ()->nothing)
    return m
end

function finite_horizon_transition(m0, s, a, maxT)
    t = s[1]
    t += 1
    s′ = m0.p(s[2],a)
    if t > maxT || m0.γ(s[2], a, s′)==0.0
        t = 1
        s′ = m0.d0()
    end

    return (t, s′)
end

function simplechain_step_finite_horizon(s, a, S, r, γ)
    t,x = s
    x′ = simplechain_transition(x,a,S)
    t += 1
    reward = r(x,a,x′)
    γt = γ(x,a,x′)
    s′ = (t,x′)
    return s′, reward, γt
end

function create_simple_chain_finitetime(num_states::Int; stochastic=false, failchance=0.1, droptime=true)
    maxT = num_states * 20
    S = (1:maxT, 1:num_states)
    if droptime
        X = S[2]
    else
        X = S
    end
    A = 1:2
    r = (s,a,s′)-> -1.0
    γ = (s,a,s′)-> (s==num_states && s′==1) ? 0.0 : 1.0
    if !stochastic
        pfun = (s,a)->simplechain_step_finite_horizon(s, a, S, r, γ)  # mask out S in above function    
    else
        pfun = (s,a)->simplechain_step_finite_horizon(s, simplechain_perturb_action(a,failchance), S, r, γ)
    end
    pdrop(s,a) = begin
        s, r, γ = pfun(s,a)
        return s, s[2], r, γ        
    end
    pfull(s,a) = begin
        s, r, γ = pfun(s,a)
        return s, s, r, γ        
    end
    if droptime
        d0 = ()->((1,minimum(S[2])), minimum(S[2]))
        p = pdrop
    else
        d0 = ()->((1,minimum(S[2])), (1,minimum(S[2])))
        p = pfull
    end
    
    meta = Dict{Symbol,Any}()
    meta[:minreward] = -1.0
    meta[:maxreward] = -1.0
    meta[:minreturn] = -Float64(maxT)
    meta[:maxreturn] = -Float64(num_states)
    meta[:stochastic] = stochastic
    meta[:minhorizon] = num_states
    meta[:maxhorizon] = maxT
    meta[:discounted] = false
    
    m = SequentialProblem(S,X,A,p,d0,meta, ()->nothing)

    return m
end
