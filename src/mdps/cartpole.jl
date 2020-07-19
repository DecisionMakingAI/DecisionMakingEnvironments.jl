using StaticArrays

struct CartPoleParams{T} <:Any where {T<:Real}
    m::T # mass of pole
    l::T # length of pole
    mc::T # mass of cart
    muc::T # some constant?
    mup::T # some constant?
    fmag::T # magnitude of force applied
	g::T  # gravity (not signed)
    CartPoleParams() = new{Float64}(0.1, 0.5, 1., 0.0005, 0.000002, 10., 9.8)
    CartPoleParams(T::Type) = new{T}(T(0.1), T(0.5), T(1.), T(0.0005), T(0.000002), T(10.), T(9.8))
	CartPoleParams(T::Type, m, l, mc, g) = new{T}(m, l, mc, T(0.0005), T(0.000002), T(10.), g)
end

function cartpole_sim(state, constants::CartPoleParams, u, dt)
	x, xDot, theta, thetaDot, t = view(state, 1:5)
	omegaDot = 0.
	vDot = 0.
	m = constants.m     # mass of pole
    l = constants.l     # length of pole
    mc = constants.mc   # mass of cart
    muc = constants.muc # some constant?
    mup = constants.mup # some constant?
	g = constants.g
	# for i in 1:steps
	omegaDot = (g * sin(theta) + cos(theta) * (muc * sign(xDot) - u - m * l * thetaDot^2 * sin(theta)) / (m + mc) - mup * thetaDot / (m * l)) / (l * (4.0 / 3.0 - m / (m + mc) * cos(theta)^2))
    vDot = (u + m * l * (thetaDot^2 * sin(theta) - omegaDot*cos(theta)) - muc*sign(xDot)) / (m + mc)
    theta += dt * thetaDot
    thetaDot += dt * omegaDot
    x += dt * xDot
    xDot += dt * vDot
	t += dt

    theta = mod(theta + π, 2 * π) - π

	return SVector{5}(x, xDot, theta, thetaDot, t)
end

function cartpole_compute_torque(p::CartPoleParams, action::Int)
	if action <= 0 || action > 2
		error("Action needs to be an integer in [1, 2]")
	end
	u = 0.
	if action == 1
		u = -p.fmag
	else
		u =  p.fmag
	end
	return u
end

function cartpole_compute_torque(p::CartPoleParams, action)
	return clamp(action, -1.0, 1.0) * p.fmag
end


function cartpole_step(state, action, params, dt, tMax)
	u = cartpole_compute_torque(params, action)
	next_state = cartpole_sim(state, params, u, dt)
	if cartpole_terminal(next_state)
		next_state = zeros(SVector{5})
	end
	return next_state
end

function create_finitetime_cartpole(;randomize=false, tMax=20.0, dt=0.02, Atype=:Discrete, droptime=true)
	if randomize
		params = random_cartpole_params()
	else
		params = CartPoleParams()
	end
	return create_finitetime_cartpole(params; tMax=tMax, dt=dt, Atype=Atype, droptime=droptime)
end

function create_finitetime_cartpole(params::CartPoleParams; tMax=20.0, dt=0.02, Atype=:Discrete, droptime=true)
	S = @SMatrix [	-2.4 2.4 ;       	# x range
					-10. 10. ;       	# xDot range
					-π/12.0 π/12.0 ; # theta range
					-π π ;           	# thetaDot range
					0. tMax]			# time range
	if Atype==:Discrete
		A = 1:2
	else
		A = SMatrix{1,2}([-1.0,1.0])
	end
	p = (s,a)->cartpole_step(s,a, params, dt, tMax)
	d0 = ()->zeros(SVector{5})
	if droptime
		X = S[1:4, :]
		obs(s) = s[1:4]
		m = POMDP(S,A,X,p,obs,d0)
	else
		m = MDP(S,A,p,d0)
	end
	return m
end

function random_cartpole_params()
	m = rand(rng, Uniform(0.025, .25))  # mass of pole
	l = rand(rng, Uniform(0.1, 1.0))    # length of pole
	mc = rand(rng, Uniform(0.1, 5.))    # mass of cart
	g = rand(rng, Uniform(8., 10.))     # gravity
	params = CartPoleParams(T, m, l, mc, g)
	return params
end

function cartpole_terminal(state)
	polecond = abs(state[3]) > (π / 15.0)
	cartcond = abs(state[1]) ≥ 2.4
	timecond = state[5] > (20. - 1e-8)
	done = polecond | cartcond | timecond
	return done
end

function create_cartpole_balancetask(m, tMax)
	r = (s,a,s′)-> 1.0
	γ = (s,a,s′)-> (s[5] > 0.0 && s′[5] == 0.0) ? 0.0 : 1.0
	task = RLTask(m, r, γ)
    return task
end
