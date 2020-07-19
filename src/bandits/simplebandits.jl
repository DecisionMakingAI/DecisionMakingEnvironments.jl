using StaticArrays

export create_simple_discrete_bandit, create_simple_discrete_contextualbandit

function create_simple_discrete_bandit(num_actions::Int; noise=1.0)
    A = 1:num_actions
    function simple_reward(a, noise)
        return a + randn() * noise
    end
    r = a -> simple_reward(a, noise)
    b = Bandit(A, r)
    return b
end

function create_simple_discrete_contextualbandit(num_actions::Int; noise_x=1.0, noise_r=1.0)
    X = SMatrix{1,2}([-2noise_x, 2noise_x])
    A = 1:num_actions

    function sample_context(noise)
        return clamp(randn() * noise, -2*noise, 2*noise)
    end

    function simple_reward(x, a, noise)
        return x*a + randn() * noise
    end

    d = ()->sample_context(noise_x)
    r = (x,a) -> simple_reward(x,a, noise_r)
    b = ContextualBandit(X,A,d,r)
    return b
end
