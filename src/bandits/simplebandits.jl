
function create_simple_discrete_bandit(num_actions::Int; noise=1.0)
    A = 1:num_actions
    function simple_reward(a, noise)
        return a + clamp(randn(), -3, 3) * noise
    end
    r = a -> simple_reward(a, noise)
    meta = Dict{Symbol, Any}()
    meta[:minreward] = -Inf
    meta[:maxreward] = Inf
    meta[:minobjective] = 1 - 3 * noise
    meta[:maxobjective] = num_actions + 3 * noise
    b = BanditProblem(A, r, meta)
    
    return b
end

function create_simple_discrete_contextualbandit(num_actions::Int; noise_x=1.0, noise_r=1.0)
    X = [-2noise_x 2noise_x]
    A = 1:num_actions

    function sample_context(noise)
        return clamp(randn() * noise, -2*noise, 2*noise)
    end

    function simple_reward(x, a, noise)
        return x*a + randn() * noise
    end

    d = ()->sample_context(noise_x)
    r = (x,a) -> simple_reward(x,a, noise_r)
    meta = Dict{Symbol, Any}()
    meta[:minreward] = -Inf
    meta[:maxreward] = Inf
    meta[:minobjective] = -(√2 / π)
    meta[:maxobjective] = num_actions * (√2 / π)
    b = ContextualBanditProblem(X,A,d,r,meta)
    return b
end
