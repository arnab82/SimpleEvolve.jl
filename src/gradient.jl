using SimpleEvolve

"""
gradientsignal_ODE(ψ0, T, signals, n_sites, drives, eigvalues, device_action_independent_t, cost_ham, n_signals, ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites); tol_ode=1e-8)
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal using ODE
    args:
        ψ0        : Initial state vector
        T         : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        device_action_independent_t: Device action independent time evolution
        cost_ham  : Cost Hamiltonian
        n_signals : Number of signals
        ∂Ω        : Gradient matrix (default is uninitialized)
        tol_ode   : Tolerance for ODE solver (default is 1e-8)
    returns:
        ∂Ω       : Gradient matrix

"""

function gradientsignal_ODE(ψ0::Vector{ComplexF64},
                            T::Float64,
                            signals,
                            n_sites::Int64,
                            drives::Vector{Matrix{Float64}},
                            eigvalues::Vector{Float64},
                            eigvectors::Matrix{ComplexF64},
                            cost_ham,
                            n_signals::Int64,
                            ∂Ω_real=Matrix{Float64}(undef,n_signals+1,n_sites),
                            ∂Ω_imag=Matrix{Float64}(undef,n_signals+1,n_sites);
                            basis = "eigenbasis",
                            tol_ode=1e-8,
                            τ = T/n_signals) 

    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    δt    = T/n_signals
    tmp_σ = zeros(ComplexF64, length(ψ0))
    tmp_ψ = zeros(ComplexF64, length(ψ0))
   
    # Evolve the sigma state with ODE in forward direction
    parameters = [signals, n_sites, drives,eigvalues,false]
    prob = ODEProblem(dψdt!, σ, (0.0,T), parameters)
    sol  = solve(prob,RK4(), abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
    σ .= sol.u[end]
    σ .= σ /norm(σ)

    # If cost_Hamiltonian is in eigenbasis comment out next line
    transform!(σ, eigvectors,tmp_σ)            # Transform to qubitbasis
    σ .= mul!(tmp_σ,cost_ham,σ)                # Calculate C|ψ⟩    
    
    # Reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)            # Transform back to eigenbasis
    parameters = [signals, n_sites, drives,eigvalues,false]
    prob_ = ODEProblem(dψdt!, σ, (T,0.0), parameters)
    sol_  = solve(prob_,RK4(), abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
    σ .= sol_.u[end]
    σ .= σ /norm(σ)

    # Calculating gradient by evolving both ψ and σ states
    for i ∈ 1:n_signals
        t_i = t_[i]
        t_f = t_[i]+δt
        
        gradient_eachtimestep!(∂Ω_real, ∂Ω_imag, ψ, σ, signals, n_sites, 
                              drives, eigvalues, t_i, i, τ)
        parameters = [signals, n_sites, drives, eigvalues,false]
        
        # Evolve ψ forward
        prob_ψ = ODEProblem(dψdt!, ψ, (t_i, t_f), parameters)
        sol_ψ = solve(prob_ψ,RK4(),abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
        ψ .= sol_ψ.u[end]

        # Evolve σ forward
        prob_σ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_σ  = solve(prob_σ,RK4(), abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
        σ .= sol_σ.u[end]
    end
    
    # Final gradient calculation
    gradient_eachtimestep!(∂Ω_real, ∂Ω_imag, ψ, σ, signals, n_sites,
                          drives, eigvalues, t_[end], n_signals+1, τ)
    
    # Normalize and transform back if needed
    σ .= σ /norm(σ)
    ψ .= ψ /norm(ψ)
    if basis != "eigenbasis"
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_ψ, eigvectors, σ)
    end
    
    return ∂Ω_real, ∂Ω_imag, ψ, σ  
end
"""
gradientsignal_ODE_real(ψ0, T, signals, n_sites, drives, eigvalues, device_action_independent_t, cost_ham, n_signals, ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites); tol_ode=1e-8)
    Function to compute the gradient of the energy 
    with respect to the real amplitude of the signal using ODE
    args:
        ψ0        : Initial state vector
        T         : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        device_action_independent_t: Device action independent time evolution
        cost_ham  : Cost Hamiltonian
        n_signals : Number of signals
        ∂Ω       : Gradient matrix (default is uninitialized)
        tol_ode   : Tolerance for ODE solver (default is 1e-8)
    returns:
        ∂Ω       : Gradient matrix
"""

function gradientsignal_ODE_real(ψ0::Vector{ComplexF64},
                            T::Float64,
                            signals,
                            n_sites::Int64,
                            drives::Vector{Matrix{Float64}},
                            eigvalues::Vector{Float64},
                            eigvectors::Matrix{ComplexF64},
                            cost_ham,
                            n_signals::Int64,
                            ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                            basis = "eigenbasis",
                            tol_ode=1e-8,
                            τ = T/n_signals) 

    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    δt    = T/n_signals
    tmp_σ = zeros(ComplexF64, length(ψ0))
    tmp_ψ = zeros(ComplexF64, length(ψ0))
   
    # Evolve the sigma state with ODE in forward direction
    parameters = [signals, n_sites, drives,eigvalues,false]
    prob = ODEProblem(dψdt!, σ, (0.0,T), parameters)
    sol  = solve(prob,RK4(), abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
    σ .= sol.u[end]
    σ .= σ /norm(σ)

    # If cost_Hamiltonian is in eigenbasis comment out next line
    transform!(σ, eigvectors,tmp_σ)            # Transform to qubitbasis
    σ .= mul!(tmp_σ,cost_ham,σ)                # Calculate C|ψ⟩    
    
    # Reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)            # Transform back to eigenbasis
    parameters = [signals, n_sites, drives,eigvalues,false]
    prob_ = ODEProblem(dψdt!, σ, (T,0.0), parameters)
    sol_  = solve(prob_,RK4(), abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
    σ .= sol_.u[end]
    σ .= σ /norm(σ)

    # Calculating gradient by evolving both ψ and σ states
    for i ∈ 1:n_signals
        t_i = t_[i]
        t_f = t_[i]+δt
        gradient_eachtimestep_real!(∂Ω, ψ, σ, signals, n_sites, 
                              drives, eigvalues, t_i, i, τ)
        parameters = [signals, n_sites, drives, eigvalues,false]
        
        # Evolve ψ forward
        prob_ψ = ODEProblem(dψdt!, ψ, (t_i, t_f), parameters)
        sol_ψ = solve(prob_ψ,RK4(),abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
        ψ .= sol_ψ.u[end]

        # Evolve σ forward
        prob_σ = ODEProblem(dψdt!, σ, (t_i, t_f), parameters)
        sol_σ  = solve(prob_σ,RK4(), abstol=tol_ode, reltol=tol_ode,save_everystep=false,maxiters=1e8)
        σ .= sol_σ.u[end]
    end
    
    # Final gradient calculation
    gradient_eachtimestep_real!(∂Ω, ψ, σ, signals, n_sites,
                          drives, eigvalues, t_[end], n_signals+1, τ)
    
    # Normalize and transform back if needed
    σ .= σ /norm(σ)
    ψ .= ψ /norm(ψ)
    if basis != "eigenbasis"
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_ψ, eigvectors, σ)
    end
    
    return ∂Ω, ψ, σ  
end



"""
gradient_eachtimestep(δΩ, ψ, σ, multi_signal, n_sites, drives, device_action_independent_t, t, time_index, time_factor)
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal at each time step
    args:
        δΩ                : Gradient matrix
        ψ                 : Evolved state vector
        σ                 : Evolved state vector
        multi_signal      : Multi-channel signal
        n_sites           : Number of sites in the system
        drives            : External drives applied to the system
        device_action_independent_t: Device action independent time evolution
        t                 : Current time
        time_index        : Index of the current time step
        time_factor       : Time factor for exponentiation on device_action_independent_t
         to get the interaction Hamiltonian at time t
    returns:
        δΩ               : Updated gradient matrix at the current time step

"""


function gradient_eachtimestep!(∂Ω_real::Matrix{Float64}, 
                               ∂Ω_imag::Matrix{Float64},  
                               ψ::Vector{ComplexF64},
                               σ::Vector{ComplexF64},
                               multi_signal,
                               n_sites::Int64,
                               drives::Vector{Matrix{Float64}},
                               eigvalues::Vector{Float64},
                               t::Float64,
                               time_index::Int64,
                               τ::Float64)
    
    dim = length(ψ)
    dH_dΩ_real = zeros(ComplexF64, dim, dim)
    dH_dΩ_imag = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    ψ0 = copy(ψ)
    σ0 = copy(σ)
    
    for k in 1:n_sites
        # Construct Hermitian derivative operator 
        # amp = amplitude(signals.channels[k], t)
        term = exp(im * frequency(multi_signal.channels[k], t) * t) .* drives[k]
        dH_dΩ_real .= term + term'
        dH_dΩ_imag .= im * term - im*term' #phase factor
        # Interaction picture transformation
        device_action .= exp.((im * t) .* eigvalues)
        expD = Diagonal(device_action)
        lmul!(expD, dH_dΩ_imag)
        rmul!(dH_dΩ_imag, expD')
        lmul!(expD, dH_dΩ_real)
        rmul!(dH_dΩ_real, expD')

        # Compute gradient components
        AΨ_real = dH_dΩ_real * ψ0
        σAψ_real = -im * (σ' * AΨ_real) * τ
        AΨ_imag = dH_dΩ_imag * ψ0
        σAψ_imag = -im * (σ' * AΨ_imag) * τ
        ∂Ω_real[time_index, k] = 2 * real(σAψ_real)
        ∂Ω_imag[time_index, k] = 2 * real(σAψ_imag)

        # ∂Ω_real[time_index, k] = σAψ_real + σAψ_real'
        # ∂Ω_imag[time_index, k] = σAψ_imag + σAψ_imag'

        dH_dΩ_imag .= zeros(ComplexF64, dim, dim)
        dH_dΩ_real .= zeros(ComplexF64, dim, dim)
    end
end



"""
gradientsignal_direct_exponentiation(ψ0, T, signals, n_sites, drives, eigvalues, device_action_independent_t, n_trotter_steps, cost_ham, n_signals, ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites))
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal using direct exponentiation
    args:
        ψ0       : Initial state vector
        T        : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        device_action_independent_t: Device action independent time evolution
        n_trotter_steps: Number of Trotter steps for exponentiation
        cost_ham  : Cost Hamiltonian
        n_signals : Number of signals
        ∂Ω       : Gradient matrix (default is uninitialized)
    returns:
        ∂Ω       : Gradient matrix


"""

function gradientsignal_direct_exponentiation(ψ0::Vector{ComplexF64},
                                        T::Float64,
                                        signals,
                                        n_sites::Int64,
                                        drives::Vector{Matrix{Float64}},
                                        eigvalues::Vector{Float64},
                                        eigvectors::Matrix{ComplexF64},
                                        cost_ham,
                                        n_signals::Int64,
                                        ∂Ω = Matrix{Float64}(undef, n_signals+1, n_sites);
                                        basis = "eigenbasis",
                                        n_trotter_steps=1000,
                                        τ = T/n_signals)
           
    # eigvalues, eigvecs = eigen(Hstatic)
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    ψ = copy(ψ0)
    σ = copy(ψ0)
    t_series=range(0,T,n_trotter_steps+1)
    t_= range(0,T,length=n_signals+1)
    dt= T/n_trotter_steps
    Δt =T/n_signals
    tmp_σ = zeros(ComplexF64, length(ψ0))
    tmp_ψ = zeros(ComplexF64, length(ψ0))


    # time evolution with direct exponentiation
    for i in 1:n_trotter_steps+1
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues,dt,t_series[i])
    end
    σ .= σ /norm(σ)
    # σ .*=exp.((im*T)*eigvalues)               # rotate phases for final exp(iHDT)

    # if cost_Hamiltonian is in eigenbasis comment out next line
    transform!(σ, eigvectors,tmp_σ)            # transform the state to the qubitbasis
    σ .= mul!(tmp_σ,cost_ham,σ)                # calculate C|ψ⟩    
    
    # # reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)            # transform the state to the eigenbasis
    # σ .*=exp.((-im*T)*eigvalues)             # rotate phases for final exp(iHDT)


    #time evolution backward for sigma state
    for i in reverse(1:n_trotter_steps+1)
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, dt,t_series[i],true)
    end
    σ .= σ /norm(σ)
    #calculating gradient by evolving both ψ and σ states
    for i in 1:n_signals+1
        gradient_eachtimestep_real!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,t_[i],i,τ)
        ψ .= single_trotter_exponentiation_step(ψ,signals, n_sites, drives, eigvalues, Δt,t_[i])
        σ .= single_trotter_exponentiation_step(σ,signals, n_sites, drives, eigvalues, Δt,t_[i])
    end
    gradient_eachtimestep_real!(∂Ω,ψ,σ,signals,n_sites,drives,eigvalues,t_series[end],n_signals+1,τ)
  
    ψ .= ψ/norm(ψ)
    σ .= σ/norm(σ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_ψ, eigvectors, σ)
    end
    return ∂Ω, ψ, σ

end
"""
# Function to compute the gradient of the energy with respect to the amplitude of the signal at each time step
the signal should have real amplitudes
"""

function gradient_eachtimestep_real!(∂Ω,
                                ψ::Vector{ComplexF64},
                                σ::Vector{ComplexF64},
                                multi_signal,
                                n_sites::Int64,
                                drives::Vector{Matrix{Float64}},
                                eigvalues::Vector{Float64},
                                t::Float64,
                                time_index::Int64,
                                τ::Float64)
    
    dim= length(ψ)
    dH_dΩ = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    ψ0=copy(ψ)
    σ0=copy(σ)
    for k in 1:n_sites
        dH_dΩ .+= exp(im*frequency(multi_signal.channels[k], t)*t).*drives[k]
        dH_dΩ .+= dH_dΩ'

        # derivative of interaction picture Hamiltonian
        ## dt=multi_signal.channels[k].δt  
        device_action .= exp.((im*t) .* eigvalues) 
        ## device_action .= exp.((-im*dt) .* eigvalues)                 
        expD =Diagonal(device_action)                   
        lmul!(expD, dH_dΩ); rmul!(dH_dΩ, expD')  
        ## dH_dΩ = eigvectors' * dH_dΩ * eigvectors
        
        AΨ = dH_dΩ * ψ0
        # calculate gradient ⟨σ|A|ψ⟩
        # σAψ = -im * (σ0' * AΨ)
        σAψ = -im * (σ' * AΨ)*τ # this tau is generally equal to signals dt , 
        # but for signal reconstruction we need to use τ=T/n_signals not τ=T/n_samples_grad

        # ⟨σ|A|ψ⟩ + ⟨ψ|A|σ⟩ 
        ∂Ω[time_index,k] = σAψ + σAψ'                      
        dH_dΩ .= zeros(dim,dim) 
    end 
    
end
"""
gradientsignal_rotate(ψ0, T, signals, n_sites, drives, eigvalues, device_action_independent_t, cost_ham, n_signals, ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites), n_trotter_steps=1000)
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal using trotter step
    args:
        ψ0       : Initial state vector
        T        : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        device_action_independent_t: Device action independent time evolution
        cost_ham  : Cost Hamiltonian
        n_signals : Number of signals
        ∂Ω       : Gradient matrix (default is uninitialized)
        n_trotter_steps: Number of Trotter steps for exponentiation (default is 1000)
    returns:
        ∂Ω       : Gradient matrix
"""
function gradientsignal_rotate(ψ0::Vector{ComplexF64},
                            T::Float64,
                            signals,
                            n_sites::Int64,
                            n_levels::Int64,
                            a_q::Matrix{Float64},
                            eigvalues::Vector{Float64},
                            eigvectors::Matrix{ComplexF64},
                            cost_ham,
                            n_signals::Int64,
                            ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                            n_trotter_steps=1000,
                            basis = "eigenbasis",
                            V_evolve = nothing,
                            V_gradient = nothing) 
    
    ## transform!(σ, V, tmpV)=> σ=mul!(tmpV, V, σ) 
    # eigvalues, eigvecs = eigen(Hstatic)
    tmp_σ = zeros(ComplexF64, length(ψ0))
    if basis == "eigenbasis" # rotating out of the eigenspace 
        mul!(tmp_σ, eigvectors, ψ0)
    end
    tmp_ψ = zeros(ComplexF64, length(ψ0))
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    t_series=range(0,T,length=n_trotter_steps+1)
    δt    = T/n_signals
    Δt    = T/n_trotter_steps

    if V_evolve == nothing
        V_evolve = eigvectors*Diagonal(exp.((-im*Δt ) * eigvalues)) *eigvectors'
    end
    if V_gradient == nothing
        V_gradient     = eigvectors*Diagonal(exp.((-im*δt ) * eigvalues)) *eigvectors'
    end
    
    tmpM_ = [Matrix{ComplexF64}(undef, n_levels,n_levels) for q ∈ 1:n_sites]  
    tmpK_ = [Matrix{ComplexF64}(undef,  n_levels^q,  n_levels^q) for q ∈ 1:n_sites]


    σ .= single_step(σ, t_series[1], Δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    transform!(σ, V_evolve, tmp_σ)
    for i ∈ (2:n_trotter_steps)
        t_i = t_series[i]
        σ .= single_step(σ, t_i, Δt, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
        transform!(σ, V_evolve, tmp_σ)                #transform!(σ, V, tmpV)=> σ=mul!(tmpV, V, σ) 
    end
    σ .= single_step(σ, t_series[end], Δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    σ .= σ /norm(σ)
    
    transform!(σ, eigvectors',tmp_σ)          # transform the state to the eigen space
    σ .*=exp.((im*T)*eigvalues)               # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmp_σ)          # transform the state to the qubitspace

    σ .= mul!(tmp_σ,cost_ham,σ)               # calculate C|ψ⟩    
    
    # # reverse time evolution of the sigma state
    transform!(σ,eigvectors',tmp_σ)           # transform the state to the device space
    σ .*=exp.((-im*T)*eigvalues)               # rotate phases for final exp(iHDT)
    transform!(σ, eigvectors, tmp_σ)          # transform the state to the qubitspace
    

    σ .= single_step(σ, t_series[end], Δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_,true)
    transform!(σ, V_evolve', tmp_σ)
    for i ∈ reverse(2:n_trotter_steps)
        t_i = t_series[i]
        σ .= single_step(σ, t_i, Δt, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_,true)
        transform!(σ, V_evolve', tmp_σ)
    end
    σ .= single_step(σ, t_series[1], Δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_,true)
    σ .= σ /norm(σ)

    #calculating gradient by evolving both ψ and σ states
    gradient_eachstep!(∂Ω, 1, σ, ψ, t_[1],δt/2 , signals,
                                n_sites, a_q,tmp_σ,tmpM_,tmpK_)


    σ .= single_step(σ, t_[1], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    ψ .= single_step(ψ, t_[1], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    transform!(σ, V_gradient, tmp_σ)
    transform!(ψ, V_gradient, tmp_σ)


    for i ∈ (2:n_signals)
        t_i = t_[i]
        gradient_eachstep!(∂Ω, i, σ, ψ, t_i,δt , signals,
                                n_sites, a_q,tmp_σ,tmpM_,tmpK_)
        σ .= single_step(σ, t_i, δt, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
        ψ .= single_step(ψ, t_i, δt, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
        transform!(σ, V_gradient, tmp_σ)
        transform!(ψ, V_gradient, tmp_σ)


    end
    gradient_eachstep!(∂Ω, n_signals+1, σ, ψ, t_[end] ,δt/2 , signals,
                                n_sites, a_q,tmp_σ,tmpM_,tmpK_)
    σ .= single_step(σ, t_[end], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    ψ .= single_step(ψ, t_[end], δt/2, signals, n_sites, a_q, tmp_σ, tmpM_, tmpK_)
    
    transform!(ψ, eigvectors',tmp_ψ)           # transform the state to the device space
    transform!(σ, eigvectors',tmp_ψ)
    ψ .*=exp.((im*T)*eigvalues)                 # rotate phases for final exp(iHDT)
    σ .*=exp.((im*T)*eigvalues)
    
    ψ ./= norm(ψ)
    σ ./= norm(σ)
    if basis != "eigenbasis" # rotating out of the eigenspace
        ψ .= mul!(tmp_ψ, eigvectors, ψ)
        σ .= mul!(tmp_ψ, eigvectors, σ)
    end
    return ∂Ω,ψ,σ
end
"""
gradient_eachstep(∂Ω, i, σ, ψ, t, τ, multi_signal, n_sites, drives, eigvalues)
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal at each time step using trotter step
    args:
        ∂Ω                : Gradient matrix
        i                 : Index of the current time step
        σ                 : Evolved state vector
        ψ                 : Evolved state vector
        t                 : Current time
        τ                 : Time factor for exponentiation on device_action_independent_t
         to get the interaction Hamiltonian at time t
        multi_signal      : Multi-channel signal
        n_sites           : Number of sites in the system
        drives            : External drives applied to the system
        eigvalues         : Eigenvalues of the Hamiltonian
    returns:
        ∂Ω               : Updated gradient matrix at the current time step

"""

function gradient_eachstep!(∂Ω,
                            i::Int64,
                            σ::Vector{ComplexF64},
                            ψ::Vector{ComplexF64},
                            t::Float64,
                            τ::Float64,
                            multi_signal,
                            n_qubits::Int64,
                            a::Matrix{Float64},
                            tmpV::Vector{ComplexF64},
                            tmpM_::Vector{Matrix{ComplexF64}},
                            tmpK_::Vector{Matrix{ComplexF64}})
    for q ∈ 1:n_qubits
        tmpM_[q] .= one(a)
    end
    ψ0=copy(ψ)
    σ0=copy(σ)
    for q ∈ 1:n_qubits
        ν = frequency(multi_signal.channels[q], t)
        z = exp(im*ν*t)
        tmpM_[q] .= z .* a                  # ADD za TERM
        tmpM_[q] .= Hermitian(tmpM_[q])     # Hermitian ADDS a'
        O = kron_concat(tmpM_, tmpK_)
        mul!(tmpV, O, ψ0)

        # CALCULATE GRADIENT
        # σAψ = -im*τ * (σ0' * tmpV)       # THE BRAKET ⟨σ|A|ψ⟩
        σAψ = -im * (σ0' * tmpV) 
        ∂Ω[i,q] = σAψ + σAψ'              # THE GRADIENT ⟨σ|A|ψ⟩ + ⟨ψ|A|σ⟩
        tmpM_[q] .= one(a)
    end
end




""" Auxiliary function to evolve a single step in time. """
function single_step(ψ::Vector{ComplexF64},
                    t::Float64,
                    τ::Float64,
                    signals,
                    n_qubits::Int64,
                    a::Matrix{Float64},
                    tmpV::Vector{ComplexF64},
                    tmpM_::Vector{Matrix{ComplexF64}},
                    tmpK_::Vector{Matrix{ComplexF64}}, 
                    adjoint=false)
   
    for q ∈ 1:n_qubits
        Ω = amplitude(signals.channels[q], t)
        ν = frequency(signals.channels[q], t)
        z = Ω * exp(im*ν*t)
        tmpM_[q] .= z .* a  

        # construct the time evolution operator
        tmpM_[q] .= exp(( ((-1)^adjoint)* -im*τ) .* Hermitian(tmpM_[q]))
    end
    O = kron_concat(tmpM_, tmpK_)
    return mul!(tmpV, O, ψ)
end

"""
gradientsignal_finite_difference(ψ0, T, signals, n_sites, drives, eigvalues, cost_ham, n_signals, ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites); basis = "eigenbasis", ϵ = 1e-6)
    Function to compute the gradient of the energy 
    with respect to the amplitude of the signal using finite difference
    args:
        ψ0       : Initial state vector
        T        : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        cost_ham  : Cost Hamiltonian
        n_signals : Number of signals
        ∂Ω       : Gradient matrix (default is uninitialized)
        basis     : Basis in which to compute the gradient (default is "eigenbasis")
        ϵ         : Perturbation size for finite difference (default is 1e-6)
    returns:
        ∂Ω       : Gradient matrix


"""
function gradientsignal_finite_difference(ψ0::Vector{ComplexF64},
                            T::Float64,
                            signals,
                            n_sites::Int64,
                            drives::Vector{Matrix{Float64}},
                            eigvalues::Vector{Float64},
                            eigvectors::Matrix{ComplexF64},
                            cost_ham,
                            n_signals::Int64,
                            ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                            basis = "eigenbasis",
                            ϵ = 1e-6) 

    # eigvalues, eigvecs = eigen(Hstatic)
    if basis != "eigenbasis"
        ψ0 = eigvectors' * ψ0
    end
    ψ     = copy(ψ0)
    σ     = copy(ψ0)
    t_    = range(0,T,length=n_signals+1)
    δt    = T/n_signals 
    # println("n_signals: ", n_signals)
    for i in 1:n_sites
        # Create two signal copies for ±ε perturbations
        signals_δt_plus = copy(signals)
        signals_δt_minus = copy(signals)
        
        old_sample = signals_δt_plus.channels[i].samples
        # Compute finite differences
        for j in 1:n_signals+1
            
            signals_δt_plus.channels[i].samples[j]  = old_sample[j] + ϵ
            signals_δt_minus.channels[i].samples[j] = old_sample[j] - ϵ
            t_f = t_[j] 
            # display(j)
            # Positive perturbation
            e_plus, _ = costfunction_ode(ψ, eigvalues, signals_δt_plus, n_sites, 
                                        drives, eigvectors, t_f, cost_ham)
            
            # Negative perturbation
            e_minus, _ = costfunction_ode(ψ, eigvalues, signals_δt_minus, n_sites,
                                        drives, eigvectors, t_f, cost_ham)
            
            # Central difference gradient
            ∂Ω[j, i] = (e_plus - e_minus) / (2ϵ)
            signals_δt_plus  = copy(signals)
            signals_δt_minus = copy(signals)
        end
    end

    return ∂Ω
end

"""
computes gradient signal using finite difference (forward difference formula)
"""
function gradientsignal_fd(ψ0::Vector{ComplexF64},
                        T::Float64,
                        signals,
                        n_sites::Int64,
                        drives::Vector{Matrix{Float64}},
                        eigvalues::Vector{Float64},
                        eigvectors::Matrix{ComplexF64},
                        cost_ham,
                        n_signals::Int64,
                        ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                        basis="eigenbasis", ϵ=1e-6)
    # Basis transformation
    basis != "eigenbasis" && (ψ0 = eigvectors' * ψ0)
    
    ψ = copy(ψ0)
    t_ = range(0, T, length=n_signals+1)
    δt = T/n_signals

    # Precompute original samples for each channel
    original_samples = [copy(sig.samples) for sig in signals.channels]

    for i in 1:n_sites
        # Create lightweight copies for perturbation
        channel_plus = copy(signals.channels[i])
        channel_minus = copy(signals.channels[i])
        
        for j in 1:n_signals+1
            t_f = t_[j]

            # Perturb only the j-th sample in i-th channel
            channel_plus.samples[j] = original_samples[i][j] + ϵ
            channel_minus.samples[j] = original_samples[i][j] - ϵ

            # Build temporary signals with modified channel
            signals_plus = MultiChannelSignal(
                [k == i ? channel_plus : signals.channels[k] for k in 1:n_sites]
            )
            signals_minus = MultiChannelSignal(
                [k == i ? channel_minus : signals.channels[k] for k in 1:n_sites]
            )

            # Compute energies
            e_plus, _ = costfunction_ode(ψ, eigvalues, signals_plus, n_sites, 
                                       drives, eigvectors, t_f, cost_ham)
            # e_minus, _ = costfunction_ode(ψ, eigvalues, signals_minus, n_sites,
                                        # drives, eigvectors, t_f, cost_ham)
            e0, _ = costfunction_ode(ψ, eigvalues, signals, n_sites,
                                        drives, eigvectors, t_f, cost_ham)
            # Central difference gradient
            # ∂Ω[j, i] = (e_plus - e_minus) / (2ϵ)
            ∂Ω[j, i] = (e_plus - e0) / (ϵ)
            # Reset to original for next iteration
            channel_plus.samples[j] = original_samples[i][j]
            channel_minus.samples[j] = original_samples[i][j]
        end
    end

    return ∂Ω
end
function Base.copy(ds::DigitizedSignal{T}) where T
    # Deep copy the samples vector; Float64 fields are immutable
    return DigitizedSignal{T}(copy(ds.samples), ds.δt, ds.carrier_freq)
end


"""
gradientsignal_ODE_real_multiple_states(Ψ0, T, signals, n_sites, drives, eigvalues, cost_ham, n_signals, ∂Ω=Array{Float64}(undef, n_signals+1, n_sites); basis = "eigenbasis", tol_ode=1e-8, τ = T/n_signals)
    Function to compute the gradient of the energy 
    with respect to the real amplitude of the signal at each time step using ODE evolution for multiple states
    args:
        Ψ0       : Initial state matrix (each column is a state vector)
        T        : Total time for evolution
        signals   : Signals to be evolved
        n_sites   : Number of sites in the system
        drives    : External drives applied to the system
        eigvalues : Eigenvalues of the Hamiltonian
        cost_ham  : Cost Hamiltonian
        n_signals : Number of signals
        ∂Ω       : Gradient tensor (default is uninitialized)
        basis     : Basis in which to compute the gradient (default is "eigenbasis")
        tol_ode   : Tolerance for ODE solver (default is 1e-8)
        τ         : Time factor for exponentiation on device_action_independent_t
         to get the interaction Hamiltonian at time t (default is T/n_signals)
    returns:
        ∂Ω       : Gradient tensor
        Ψ       : Final evolved state matrix
        Σ       : Final evolved state matrix after applying cost Hamiltonian and reverse evolution

"""


function gradientsignal_ODE_real_multiple_states(
                                            Ψ0::Matrix{ComplexF64},
                                            T::Float64,
                                            signals,
                                            n_sites::Int64,
                                            drives::Vector{Matrix{Float64}},
                                            eigvalues::Vector{Float64},
                                            eigvectors::Matrix{ComplexF64},
                                            cost_ham,
                                            n_signals::Int64,
                                            ∂Ω=Array{Float64}(undef, n_signals+1, n_sites, size(Ψ0,2));
                                            # ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                                            basis = "eigenbasis",
                                            tol_ode=1e-8,
                                            τ = T/n_signals)

    dim, n_states = size(Ψ0)
    tmp_σ = zeros(ComplexF64, dim, n_states)
    tmp_ψ = zeros(ComplexF64, dim, n_states)

    # Rotate to eigenbasis if needed
    if basis != "eigenbasis"
        Ψ0 = eigvectors' * Ψ0
    end
    fill!(∂Ω, 0.0) 
    # Initialize states
    Ψ = copy(Ψ0)
    Σ = copy(Ψ0)
    t_ = range(0,T,length=n_signals+1)
    δt = T/n_signals

    # Forward evolution of sigma states
    parameters = [signals, n_sites, drives, eigvalues, false]
    prob = ODEProblem(dψdt_multiple_states!, Σ, (0.0,T), parameters)
    sol = solve(prob, RK4(), abstol=tol_ode, reltol=tol_ode, save_everystep=false, maxiters=1e8)
    Σ .= sol.u[end]
    
    # Normalize each state
    for i in 1:n_states
        Σ[:,i] ./= norm(Σ[:,i])
    end

    # Apply cost Hamiltonian using transform! 
    for i in 1:n_states
        transform!(view(Σ,:,i), eigvectors, view(tmp_σ,:,i))
        Σ[:,i]= mul!(view(tmp_σ,:,i), cost_ham, view(Σ,:,i))
        transform!(view(Σ,:,i), eigvectors', view(tmp_σ,:,i))
    end

    # Reverse time evolution (must use matrix-capable version)
    prob_rev = ODEProblem(dψdt_multiple_states!, Σ, (T,0.0), parameters)
    sol_rev = solve(prob_rev, RK4(), abstol=tol_ode, reltol=tol_ode, save_everystep=false, maxiters=1e8)
    Σ .= sol_rev.u[end]
    
    # Normalize each state
    for i in 1:n_states
        Σ[:,i] ./= norm(Σ[:,i])
    end

    # Gradient calculation loop
    for i ∈ 1:n_signals
        t_i = t_[i]
        t_f = t_i + δt
        
        gradient_eachtimestep_real_multiple_states!(∂Ω, Ψ, Σ, signals, n_sites, 
                                       drives, eigvalues, t_i, i, τ)
        
        # Evolve Ψ forward (use matrix-capable version)
        prob_ψ = ODEProblem(dψdt_multiple_states!, Ψ, (t_i,t_f), parameters)
        Ψ .= solve(prob_ψ, RK4(), abstol=tol_ode, reltol=tol_ode, 
                 save_everystep=false, maxiters=1e8).u[end]

        # Evolve Σ backward (use matrix-capable version)
        prob_σ = ODEProblem(dψdt_multiple_states!, Σ, (t_i,t_f), parameters)
        Σ .= solve(prob_σ, RK4(), abstol=tol_ode, reltol=tol_ode,
                 save_everystep=false, maxiters=1e8).u[end]
    end
    
    # Final step gradient calculation
    gradient_eachtimestep_real_multiple_states!(∂Ω, Ψ, Σ, signals, n_sites,
                                   drives, eigvalues, t_[end], n_signals+1, τ)

    # Normalize and transform back if needed
    for i in 1:n_states
        Ψ[:,i] ./= norm(Ψ[:,i])
        Σ[:,i] ./= norm(Σ[:,i])
    end

    if basis != "eigenbasis"
        Ψ .= eigvectors * Ψ
        Σ .= eigvectors * Σ
    end
    
    return ∂Ω, Ψ, Σ  
end
"""
This function computes gradient at each time step for complex pulse in cartesian representation
"""
function gradientsignal_ODE_multiple_states(
                                            Ψ0::Matrix{ComplexF64},
                                            T::Float64,
                                            signals,
                                            n_sites::Int64,
                                            drives::Vector{Matrix{Float64}},
                                            eigvalues::Vector{Float64},
                                            eigvectors::Matrix{ComplexF64},
                                            cost_ham,
                                            n_signals::Int64,
                                            ∂Ω_real=Array{Float64}(undef, n_signals+1, n_sites, size(Ψ0,2)),
                                            ∂Ω_imag=Array{Float64}(undef, n_signals+1, n_sites, size(Ψ0,2));
                                            # ∂Ω=Matrix{Float64}(undef,n_signals+1,n_sites);
                                            basis = "eigenbasis",
                                            tol_ode=1e-8,
                                            τ = T/n_signals)

    dim, n_states = size(Ψ0)
    tmp_σ = zeros(ComplexF64, dim, n_states)
    tmp_ψ = zeros(ComplexF64, dim, n_states)

    # Rotate to eigenbasis if needed
    if basis != "eigenbasis"
        Ψ0 = eigvectors' * Ψ0
    end
    fill!(∂Ω_real, 0.0) 
    fill!(∂Ω_imag, 0.0)
    # Initialize states
    Ψ = copy(Ψ0)
    Σ = copy(Ψ0)
    t_ = range(0,T,length=n_signals+1)
    δt = T/n_signals

    # Forward evolution of sigma states
    parameters = [signals, n_sites, drives, eigvalues, false]
    prob = ODEProblem(dψdt_multiple_states!, Σ, (0.0,T), parameters)
    sol = solve(prob, RK4(), abstol=tol_ode, reltol=tol_ode, save_everystep=false, maxiters=1e8)
    Σ .= sol.u[end]
    
    # Normalize each state
    for i in 1:n_states
        Σ[:,i] ./= norm(Σ[:,i])
    end

    # Apply cost Hamiltonian using transform! 
    for i in 1:n_states
        transform!(view(Σ,:,i), eigvectors, view(tmp_σ,:,i))
        Σ[:,i]= mul!(view(tmp_σ,:,i), cost_ham, view(Σ,:,i))
        transform!(view(Σ,:,i), eigvectors', view(tmp_σ,:,i))
    end

    # Reverse time evolution (must use matrix-capable version)
    prob_rev = ODEProblem(dψdt_multiple_states!, Σ, (T,0.0), parameters)
    sol_rev = solve(prob_rev, RK4(), abstol=tol_ode, reltol=tol_ode, save_everystep=false, maxiters=1e8)
    Σ .= sol_rev.u[end]
    
    # Normalize each state
    for i in 1:n_states
        Σ[:,i] ./= norm(Σ[:,i])
    end

    # Gradient calculation loop
    for i ∈ 1:n_signals
        t_i = t_[i]
        t_f = t_i + δt
        
        gradient_eachtimestep_multiple_states!(∂Ω_real,∂Ω_imag, Ψ, Σ, signals, n_sites, 
                                       drives, eigvalues, t_i, i, τ)
        
        # Evolve Ψ forward (use matrix-capable version)
        prob_ψ = ODEProblem(dψdt_multiple_states!, Ψ, (t_i,t_f), parameters)
        Ψ .= solve(prob_ψ, RK4(), abstol=tol_ode, reltol=tol_ode, 
                 save_everystep=false, maxiters=1e8).u[end]

        # Evolve Σ backward (use matrix-capable version)
        prob_σ = ODEProblem(dψdt_multiple_states!, Σ, (t_i,t_f), parameters)
        Σ .= solve(prob_σ, RK4(), abstol=tol_ode, reltol=tol_ode,
                 save_everystep=false, maxiters=1e8).u[end]
    end
    
    # Final step gradient calculation
    gradient_eachtimestep_multiple_states!(∂Ω_real,∂Ω_imag, Ψ, Σ, signals, n_sites, 
                                       drives, eigvalues, t_[end], n_signals+1, τ)
        
    # Normalize and transform back if needed
    for i in 1:n_states
        Ψ[:,i] ./= norm(Ψ[:,i])
        Σ[:,i] ./= norm(Σ[:,i])
    end

    if basis != "eigenbasis"
        Ψ .= eigvectors * Ψ
        Σ .= eigvectors * Σ
    end
    
    return ∂Ω_real,∂Ω_imag, Ψ, Σ  
end
"""
This function computes gradient at each time step for real pulse only
"""

function gradient_eachtimestep_real_multiple_states!(∂Ω,
                                Ψ::Matrix{ComplexF64},
                                Σ::Matrix{ComplexF64},
                                multi_signal,
                                n_sites::Int64,
                                drives::Vector{Matrix{Float64}},
                                eigvalues::Vector{Float64},
                                t::Float64,
                                time_index::Int64,
                                τ::Float64)
    
    dim, n_states = size(Ψ)
    dH_dΩ = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)

    for k in 1:n_sites
        dH_dΩ .= 0.0
        # Build derivative Hamiltonian
        dH_dΩ .+= exp(im*frequency(multi_signal.channels[k], t)*t) .* drives[k]
        dH_dΩ .+= adjoint(dH_dΩ)  # Ensure Hermiticity

        # Interaction picture transformation
        device_action .= exp.((im*t) .* eigvalues)
        expD = Diagonal(device_action)
        lmul!(expD, dH_dΩ)
        rmul!(dH_dΩ, expD')

        # Process all states
        # for j in 1:n_states
        #     AΨ = dH_dΩ * Ψ[:,j]
        #     σAψ = -im * (Σ[:,j]' * AΨ) * τ
        #     ∂Ω[time_index,k] += σAψ + σAψ'
        # end
        for j in 1:n_states
            AΨ = dH_dΩ * Ψ[:,j]
            σAψ = -im * (Σ[:,j]' * AΨ) * τ
            ∂Ω[time_index,k,j] = σAψ + σAψ' # Store per-state gradient
        end
    end
end
"""
This function computes gradient for multiple states in eigenbasis for complex pulse at each time step for both real and imaginary parts of the pulse
"""


function gradient_eachtimestep_multiple_states!(∂Ω_real, 
                               ∂Ω_imag,  
                               ψ::Matrix{ComplexF64},
                               σ::Matrix{ComplexF64},
                               multi_signal,
                               n_sites::Int64,
                               drives::Vector{Matrix{Float64}},
                               eigvalues::Vector{Float64},
                               t::Float64,
                               time_index::Int64,
                               τ::Float64)
    
    dim, n_states = size(ψ)
    dH_dΩ_real = zeros(ComplexF64, dim, dim)
    dH_dΩ_imag = zeros(ComplexF64, dim, dim)
    device_action = Vector{ComplexF64}(undef, dim)
    ψ0 = copy(ψ)
    σ0 = copy(σ)
    
    for k in 1:n_sites
        # Construct Hermitian derivative operator 
        # amp = amplitude(signals.channels[k], t)
        term = exp(im * frequency(multi_signal.channels[k], t) * t) .* drives[k]
        dH_dΩ_real .= term + term'
        dH_dΩ_imag .= im * term - im*term' 
        # Interaction picture transformation
        device_action .= exp.((im * t) .* eigvalues)
        expD = Diagonal(device_action)
        lmul!(expD, dH_dΩ_imag)
        rmul!(dH_dΩ_imag, expD')
        lmul!(expD, dH_dΩ_real)
        rmul!(dH_dΩ_real, expD')

        # Compute gradient components
        for j in 1:n_states
            AΨ_real = dH_dΩ_real * ψ0[:,j]
            σAψ_real = -im * (σ[:,j]' * AΨ_real) * τ
            AΨ_imag = dH_dΩ_imag * ψ0[:,j]
            σAψ_imag = -im * (σ[:,j]' * AΨ_imag) * τ

            ∂Ω_real[time_index, k,j] = σAψ_real + σAψ_real'
            ∂Ω_imag[time_index, k,j] = σAψ_imag + σAψ_imag'
        end
        
        dH_dΩ_imag .= zeros(ComplexF64, dim, dim)
        dH_dΩ_real .= zeros(ComplexF64, dim, dim)
    end
end