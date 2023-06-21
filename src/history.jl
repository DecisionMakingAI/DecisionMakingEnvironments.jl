
struct BanditExperience{T,TA} <: Any where {T,TA}
    action::TA
    blogp::T
    reward::T
end

struct ContextualBanditExperience{T,TX,TA} <: Any where {T,TX,TA}
    obs::TX
    action::TA
    blogp::T
    reward::T
end

mutable struct Trajectory{T,TS,TA} <: Any where {T,TS,TA}
    states::Array{TS,1}
    actions::Array{TA,1}
    blogps::Array{T,1}
    rewards::Array{T,1}
    done::Bool

    function Trajectory(::Type{T}, ::Type{TS}, ::Type{TA}) where {T,TS,TA}
        new{T,TS,TA}(Array{TS,1}(), Array{TA,1}(), Array{T,1}(), Array{T,1}(), false)
    end

    function Trajectory(prob::SequentialProblem)
        TS = get_type(prob.X)
        TA = get_type(prob.A)
        T = eltype(TS)
        T = T == Int ? Float64 : T
        Trajectory(T,TS,TA)
    end

    function Trajectory(prob::MDP)
        TS = get_type(prob.S)
        TA = get_type(prob.A)
        T = eltype(TS)
        T = T == Int ? Float64 : T
        Trajectory(T,TS,TA)
    end

    function Trajectory(prob::POMDP)
        TS = get_type(prob.X)
        TA = get_type(prob.A)
        T = eltype(TS)
        T = T == Int ? Float64 : T
        Trajectory(T,TS,TA)
    end

end

function get_type(t::Tuple)
    return get_type.(t)
end

function get_type(::UnitRange{T}) where T
    return T
end

function get_type(::Array{T,1}) where T
    return T
end

function get_type(::Array{T, 2}) where T
    return Array{T,1}
end

function length(τ::Trajectory)
    return length(τ.rewards)
end

function push!(τ::Trajectory{T,TS,TA}, state::TS, action::TA, blogp::T, reward::T) where {T,TS,TA}
    push!(τ.states, deepcopy(state))
    push!(τ.actions, deepcopy(action))
    push!(τ.blogps, blogp)
    push!(τ.rewards, reward)
end


function finish!(τ::Trajectory)
    τ.done = true
end

function finished(τ::Trajectory)
    return τ.done
end