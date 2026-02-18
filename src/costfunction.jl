"""
costfunction_ode(ψ0, eigvals, signal, n_sites, drives, T, Cost_ham; tol_ode=1e-8)
    cost_function for the system using ODE solver
    args:
        ψ0     : Initial state
        eigvals: Eigenvalues of the Hamiltonian
        signal : Signal to be evolved
        n_sites: Number of sites in the system
        drives : External drives applied to the system # annhilation operators in case of qubits
        T      : Total time for evolution
        Cost_ham: Hamiltonian used for cost function calculation
        tol_ode: Tolerance for ODE solver (default is 1e-8)
    returns:
        cost   : Cost function value
        ψ_ode  : Evolved state

"""

function costfunction_ode(ψ0::Vector{ComplexF64},
                         eigvals::Vector{Float64},
                         signal, 
                         n_sites::Int, 
                         drives,
                         eigvectors::Matrix{ComplexF64}, 
                         T::Float64, 
                         Cost_ham;
                         basis = "eigenbasis",
                         tol_ode=1e-8)

    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives,eigvals,eigvectors;basis=basis,tol_ode=tol_ode)
    return real(ψ_ode'*Cost_ham*ψ_ode),  ψ_ode
end
"""
costfunction_direct_exponentiation(ψ0, 
                                   eigvals,
                                   signal, 
                                   n_sites, 
                                   drives, 
                                   T,  
                                   n_trotter_steps, 
                                   Cost_ham)

    cost_function for the system using direct exponentiation
    args:
        ψ0     : Initial state
        eigvals: Eigenvalues of the Hamiltonian
        signal : Signal to be evolved
        n_sites: Number of sites in the system
        drives : External drives applied to the system # annhilation operators in case of qubits
        T      : Total time for evolution
        n_trotter_steps: Number of Trotter steps for exponentiation
        Cost_ham: Hamiltonian used for cost function calculation
    returns:
        cost   : Cost function value
        ψ_direct: Evolved state


"""
function costfunction_direct_exponentiation(ψ0::Vector{ComplexF64}, 
                            eigvals::Vector{Float64},
                            eigvectors::Matrix{ComplexF64},
                            signal, 
                            n_sites::Int, 
                            drives,
                            Cost_ham, 
                            T::Float64;
                            basis = "eigenbasis",
                            n_trotter_steps=1000)

    ψ_direct = evolve_direct_exponentiation(ψ0, T, signal, n_sites, drives, eigvals,eigvectors;basis=basis, n_trotter_steps=n_trotter_steps)
    return real(ψ_direct'*Cost_ham*ψ_direct), ψ_direct
end

#costfunction that uses trotter time evolution 
function costfunction_trotter(ψ0::Vector{ComplexF64}, 
                            eigvalues::Vector{Float64},
                            eigvectors::Matrix{ComplexF64},
                            signals, 
                            n_sites::Int,
                            n_levels::Int, 
                            a_q::Matrix{Float64},
                            Cost_ham,
                            T::Float64;
                            basis = "eigenbasis", 
                            n_trotter_steps=1000
                            )
    ψ_trotter = trotter_evolve(ψ0,T,signals,n_sites,n_levels,a_q,eigvalues,eigvectors;basis=basis,n_trotter_steps=n_trotter_steps)
    return real(ψ_trotter'*Cost_ham*ψ_trotter), ψ_trotter
end

"""
costfunction_ode_excited_states(ψ0, eigvals, signal, n_sites, drives, eigvectors, T, Cost_ham; basis="eigenbasis", tol_ode=1e-8)

    cost_function for the system using ODE solver for excited states
    args:
        ψ0     : Initial state
        eigvals: Eigenvalues of the Hamiltonian
        signal : Signal to be evolved
        n_sites: Number of sites in the system
        drives : External drives applied to the system # annhilation operators in case of qubits
        eigvectors: Eigenvectors of the Hamiltonian
        T      : Total time for evolution
        Cost_ham: Hamiltonian used for cost function calculation
        basis  : Basis in which the evolution is performed (default is "eigenbasis")
        tol_ode: Tolerance for ODE solver (default is 1e-8)
    returns:
        cost   : Cost function value
        ψ_ode  : Evolved state

cost_function=sum_i real(ψ_i' * Cost_ham * ψ_i)

"""
function costfunction_ode_ssvqe(
                            Ψ0::Matrix{ComplexF64},
                            eigvals::Vector{Float64},
                            signal,
                            n_sites::Int,
                            drives,
                            eigvectors::Matrix{ComplexF64},
                            T::Float64,
                            Cost_ham;
                            basis = "eigenbasis",
                            tol_ode = 1e-8,
                            weights = nothing,
                            weighted = false
                        )
    n_states = size(Ψ0, 2)
    if weights === nothing && weighted===true
        weights = reverse(collect(1.0:-0.1:1.0-0.1*(n_states-1)))
    end

    Ψ_ode = evolve_ODE_multiple_states(
        Ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
        basis=basis, tol_ode=tol_ode
    )

    energies = [real(ψ' * Cost_ham * ψ) for ψ in eachcol(Ψ_ode)]
    if weighted == true
        cost = sum(energies .* weights[1:n_states])
    else
        cost = sum(energies .* ones(length(energies)))
    end
    return cost, Ψ_ode, energies
end

# costfunction for excited states using VQD approach
function costfunction_ode_vqd(
                        ψ0::Vector{ComplexF64},
                        eigvals::Vector{Float64},
                        signal, 
                        n_sites::Int, 
                        drives,
                        eigvectors::Matrix{ComplexF64}, 
                        T::Float64, 
                        Cost_ham,
                        previous_states::Vector{Vector{ComplexF64}},  # List of lower states
                        βs::Vector{Float64};                          # Penalty coefficients
                        basis = "eigenbasis",
                        tol_ode=1e-8
)
    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives, eigvals, eigvectors; basis=basis, tol_ode=tol_ode)
    energy = real(ψ_ode' * Cost_ham * ψ_ode)
    penalty = 0.0
    for (j, ψ_prev) in enumerate(previous_states)
        penalty += βs[j] * abs2(ψ_prev' * ψ_ode)
    end
    return energy + penalty, ψ_ode
end
"""
costfunction_ode_with_penalty(ψ0::Vector{ComplexF64}, eigvals::Vector{Float64}, signal, n_sites::Int, drives, eigvectors::Matrix{ComplexF64}, T::Float64, Cost_ham; basis = "eigenbasis", tol_ode=1e-8, λ::Float64=0.1, Ω₀::Float64=1.0+2π+0.02)
    Computes the cost function with a penalty term for the control amplitudes.
    for ground state
    Args:
        ψ0         : Initial state vector
        eigvals    : Eigenvalues of the Hamiltonian
        signal     : Control signal
        n_sites    : Number of sites
        drives     : Drives applied to the system
        eigvectors : Eigenvectors of the Hamiltonian
        T          : Total time for evolution
        Cost_ham   : Hamiltonian for cost function
        basis      : Basis for the computation (default: "eigenbasis")
        tol_ode    : Tolerance for ODE solver (default: 1e-8)
        λ          : Penalty weight (default: 0.1)
        Ω₀         : Reference amplitude (default: 1.0 + 2π + 0.02)
        
    Returns:
        cost      : Computed cost function value with penalty
        ψ_ode      : Evolved state vector

"""


function costfunction_ode_with_penalty(ψ0::Vector{ComplexF64},
                         eigvals::Vector{Float64},
                         signal, 
                         n_sites::Int, 
                         drives,
                         eigvectors::Matrix{ComplexF64}, 
                         T::Float64, 
                         Cost_ham;
                         basis = "eigenbasis",
                         tol_ode=1e-8,
                         λ::Float64=1.0,
                         Ω₀::Float64=2π*0.02) 

    # Evolve the state using ODE
    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
                       basis=basis, tol_ode=tol_ode)
    fidelity_cost = real(ψ_ode' * Cost_ham * ψ_ode)

    # Extract control amplitudes Ω from signal
    n_timesteps = length(signal.channels[1].samples)
    Ω = zeros(ComplexF64, n_timesteps, n_sites)
    for i in 1:n_sites
        Ω[:, i] = signal.channels[i].samples
    end
    # Extract control amplitudes Ω from signal
    Ω_flat_real = reshape(real.(Ω), :)
    Ω_flat_imag = reshape(imag.(Ω), :)
    Ω_total=hcat(Ω_flat_real,Ω_flat_imag)
    Ω_flat=reshape(Ω_total,:)

    # Compute penalty
    penalty = SimpleEvolve.penalty_function(Ω_flat, Ω₀)

    return fidelity_cost + λ * penalty, ψ_ode
end
"""
costfunction_ode_with_penalty_real(ψ0::Vector{ComplexF64},
                         eigvals::Vector{Float64},
                         signal, 
                         n_sites::Int, 
                         drives,
                         eigvectors::Matrix{ComplexF64}, 
                         T::Float64, 
                         Cost_ham;
                         basis = "eigenbasis",
                         tol_ode=1e-8,
                         λ::Float64=1.0,
                         Ω₀::Float64=2π*0.02) 

    Computes the cost function with a penalty term for the real part of control amplitudes.
    for ground state
    Args:
        ψ0         : Initial state vector
        eigvals    : Eigenvalues of the Hamiltonian
        signal     : Control signal
        n_sites    : Number of sites
        drives     : Drives applied to the system
        eigvectors : Eigenvectors of the Hamiltonian
        T          : Total time for evolution
        Cost_ham   : Hamiltonian for cost function
        basis      : Basis for the computation (default: "eigenbasis")
        tol_ode    : Tolerance for ODE solver (default: 1e-8)
        λ          : Penalty weight (default: 0.1)
        Ω₀         : Reference amplitude (default: 1.0 + 2π + 0.02)
    Returns:
        cost      : Computed cost function value with penalty
        ψ_ode      : Evolved state vector   
"""

function costfunction_ode_with_penalty_real(ψ0::Vector{ComplexF64},
                         eigvals::Vector{Float64},
                         signal, 
                         n_sites::Int, 
                         drives,
                         eigvectors::Matrix{ComplexF64}, 
                         T::Float64, 
                         Cost_ham;
                         basis = "eigenbasis",
                         tol_ode=1e-8,
                         λ::Float64=1.0,
                         Ω₀::Float64=2π*0.02) 

    # Evolve the state using ODE
    ψ_ode = evolve_ODE(ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
                       basis=basis, tol_ode=tol_ode)
    fidelity_cost = real(ψ_ode' * Cost_ham * ψ_ode)

    # Extract control amplitudes Ω from signal
    n_timesteps = length(signal.channels[1].samples)
    Ω = zeros(ComplexF64, n_timesteps, n_sites)
    for i in 1:n_sites
        Ω[:, i] = signal.channels[i].samples
    end
    Ω_flat=reshape(Ω,:)

    # Compute penalty
    penalty = SimpleEvolve.penalty_function(Ω_flat, Ω₀)

    return fidelity_cost + λ * penalty, ψ_ode
end
"""
costfunction for multiple states with complex pulse.
    implements Subspace search algorithm for excited states
penalty terms are added in real and complex amplitudes both
"""


function costfunction_ode_ssvqe_with_penalty(
                                        Ψ0::Matrix{ComplexF64},
                                        eigvals::Vector{Float64},
                                        signal,
                                        n_sites::Int,
                                        drives,
                                        eigvectors::Matrix{ComplexF64},
                                        T::Float64,
                                        Cost_ham;
                                        basis = "eigenbasis",
                                        tol_ode = 1e-8,
                                        weights = nothing,
                                        weighted = false,
                                        λ::Float64 = 1.0,
                                        Ω₀::Float64 = 2π*0.02
)
    n_states = size(Ψ0, 2)
    if weights === nothing && weighted === true
        weights = reverse(collect(1.0:-0.1:1.0-0.1*(n_states-1)))
    end

    # Evolve all initial states
    Ψ_ode = evolve_ODE_multiple_states(
        Ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
        basis = basis, tol_ode = tol_ode
    )

    # Compute energies for each state
    energies = [real(ψ' * Cost_ham * ψ) for ψ in eachcol(Ψ_ode)]
    if weighted == true
        cost = sum(energies .* weights[1:n_states])
    else
        cost = sum(energies)
    end

    # Extract control amplitudes Ω from signal
    n_timesteps = length(signal.channels[1].samples)
    Ω_real = zeros(ComplexF64, n_timesteps, n_sites)
    Ω_imag = zeros(ComplexF64, n_timesteps, n_sites)
    for i in 1:n_sites
        Ω_real[:, i] = real(signal.channels[i].samples)
        Ω_imag[:, i] = imag(signal.channels[i].samples)
    end
    Ω_flat_real = reshape(Ω_real, :)
    Ω_flat_imag = reshape(Ω_imag, :)
    Ω_total=hcat(Ω_flat_real,Ω_flat_imag)
    Ω_flat=reshape(Ω_total,:)

    # Compute penalty
    penalty = SimpleEvolve.penalty_function(Ω_flat, Ω₀)

    return cost + λ * penalty, Ψ_ode, energies
end
"""
cost function that adds penalty terms for only real amplitudes
Subspace search algorithm for excited states
"""
function costfunction_ssvqe_with_penalty_real(
                                        Ψ0::Matrix{ComplexF64},
                                        eigvals::Vector{Float64},
                                        signal,
                                        n_sites::Int,
                                        drives,
                                        eigvectors::Matrix{ComplexF64},
                                        T::Float64,
                                        Cost_ham;
                                        basis = "eigenbasis",
                                        tol_ode = 1e-8,
                                        weights = nothing,
                                        weighted = false,
                                        λ::Float64 = 1.0,
                                        Ω₀::Float64 = 2π*0.02
)
    n_states = size(Ψ0, 2)
    if weights === nothing && weighted === true
        weights = reverse(collect(1.0:-0.1:1.0-0.1*(n_states-1)))
    end

    # Evolve all initial states
    Ψ_ode = evolve_ODE_multiple_states(
        Ψ0, T, signal, n_sites, drives, eigvals, eigvectors;
        basis = basis, tol_ode = tol_ode
    )

    # Compute energies for each state
    energies = [real(ψ' * Cost_ham * ψ) for ψ in eachcol(Ψ_ode)]
    if weighted == true
        cost = sum(energies .* weights[1:n_states])
    else
        cost = sum(energies)
    end

    # Extract control amplitudes Ω from signal
    n_timesteps = length(signal.channels[1].samples)
    Ω_real = zeros(ComplexF64, n_timesteps, n_sites)
    for i in 1:n_sites
        Ω_real[:, i] = real(signal.channels[i].samples)
    end
    Ω_flat_real = reshape(Ω_real, :)

    # Compute penalty
    penalty = SimpleEvolve.penalty_function(Ω_flat_real, Ω₀)

    return cost + λ * penalty, Ψ_ode, energies
end
"""
costfunction analysis for the optimized pulse, to check the frequency components before and after lowpass filtering
    

"""

function costfunction_analysis(samples::Vector{Float64})
    # Split real vector into real and imaginary components
    n = length(samples) ÷ 2
    samples_real = samples[1:n]
    samples_imag = samples[n+1:end]

    # Reshape into complex matrix (n_samples+1 × n_qubits)
    samples_complex = complex.(
        reshape(samples_real, (n_samples + 1, n_qubits)),
        reshape(samples_imag, (n_samples + 1, n_qubits))
    )

    # Pre-allocate arrays for FFT analysis
    fft_before = Matrix{ComplexF64}(undef, n_samples + 1, n_qubits)
    fft_after = similar(fft_before)
    max_freq_before = Vector{Float64}(undef, n_qubits)
    max_freq_after = similar(max_freq_before)

    # Apply lowpass filter to each qubit's pulse
    fs = 1 / (δt)      # Sampling frequency (GHz)
    cutoff = fs / 2.0001  # Cutoff frequency (GHz) 
    order = 4        # Filter order

    for i in 1:n_qubits
        pulse = samples_complex[:, i]

        # Compute FFT before filtering
        fft_before[:, i] = fft(pulse)

        # Apply lowpass filter
        filtered_pulse = SimpleEvolve.lowpass_filter(pulse, cutoff, fs; order=order)
        samples_complex[:, i] = filtered_pulse

        # Compute FFT after filtering
        fft_after[:, i] = fft(filtered_pulse)

        # Calculate maximum frequency magnitude
        max_freq_before[i] = maximum(abs.(fft_before[:, i]))
        max_freq_after[i] = maximum(abs.(fft_after[:, i]))
    end

    # Build signals from filtered complex samples
    signals_ = [DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)

    # Compute energy 
    if penalty == true
        energy, Ψ_ode, energies = SimpleEvolve.costfunction_ode_ssvqe_with_penalty(
            ψ_initial_, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
            basis="qubitbasis", tol_ode=tol_ode, weights=weights, weighted=weighted
        )
    else
        energy, Ψ_ode, energies = SimpleEvolve.costfunction_ode_ssvqe(
            ψ_initial_, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
            basis="qubitbasis", tol_ode=tol_ode, weights=weights, weighted=weighted
        )
    end

    N = n_samples + 1  # Number of time points (FFT length)
    freqs = FFTW.fftfreq(N, 1 / fs)  # 1D frequency array

    # Thresholds for significant frequencies
    threshold_before = 0.1 * maximum(abs.(fft_before))
    threshold_after = 0.1 * maximum(abs.(fft_after))

    # Find significant indices (CartesianIndex)
    inds_before = findall(x -> x > threshold_before, abs.(fft_before))
    inds_after = findall(x -> x > threshold_after, abs.(fft_after))

    # Extract row (frequency bin) and column (qubit/channel) indices
    rows_before = [Tuple(ci)[1] for ci in inds_before]
    rows_after = [Tuple(ci)[1] for ci in inds_after]
    freqs_before = freqs[rows_before]
    freqs_after = freqs[rows_after]

    # Magnitudes 
    mags_before = [abs(fft_before[Tuple(ci)...]) for ci in inds_before]
    mags_after = [abs(fft_after[Tuple(ci)...]) for ci in inds_after]


    println("Frequencies present before filtering (GHz): ", freqs_before)
    println("Magnitudes before filtering: ", mags_before)
    println("Frequencies present after filtering (GHz): ", freqs_after)
    println("Magnitudes after filtering: ", mags_after)

    # Return FFT data if requested
    if return_fft
        return energy, freqs_before, freqs_after, mags_before, mags_after
    else
        return energy
    end
end
