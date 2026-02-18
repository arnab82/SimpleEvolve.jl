using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using Random
using JLD2
using DSP   # for dpss / DPSS (Slepian) sequences
using Arpack
using FFTW
using Interpolations
# -------------------------
# User / device parameters
# -------------------------
@testset "Slepian Optimization Pipeline" begin
    T = 15.0
    println("T=", T)
    Cost_ham = npzread("h215.npy") 
    n_qubits = round(Int, log2(size(Cost_ham, 1)))
    println("n_qubits=", n_qubits)
    n_levels = 2
    SYSTEM = "H215"
    freqs = 2π * collect(4.8 .+ (0.02 * (1:n_qubits)))
    anharmonicities = 2π * 0.3 * ones(n_qubits)
    coupling_map = Dict{QubitCoupling,Float64}()

    # linear mapping
    for p in 1:(n_qubits-1)
        q = p + 1
        coupling_map[QubitCoupling(p, q)] = 2π * 0.02
    end

    device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)

    # Ground state (Hartree-Fock) initial state
    initial_state_ground = "0011"
    ψ_initial_g = zeros(ComplexF64, n_levels^n_qubits)
    ψ_initial_g[1+parse(Int, initial_state_ground, base=n_levels)] = 1.0 + 0im
    ψ_initial = copy(ψ_initial_g)
    H_static = static_hamiltonian(device, n_levels)
    drives = a_fullspace(n_qubits, n_levels)
    eigvalues, eigvectors = eigen(Hermitian(H_static))
    println("Eigenvalues of our static Hamiltonian")
    display(eigvalues)
    tol_ode = 1e-10
    Λ, U = eigs(Cost_ham)
    E_actual = minimum(real.(Λ))
    println("Actual energy: $E_actual")
    for i in 1:n_qubits
        drives[i] = eigvectors' * drives[i] * eigvectors
    end

    seed_box_energies  = Dict{Tuple{Float64,Float64,Int64}, Vector{Float64}}()
    seed_box_errors    = Dict{Tuple{Float64,Float64,Int64}, Vector{Float64}}()
    seed_box_gradients = Dict{Tuple{Float64,Float64,Int64}, Vector{Float64}}()
    penalty = true
    fs=1.0
    B = 0.4
    seed= 12
    Random.seed!(seed)
    N= Int(floor(T * fs))
    δt = T / N
    W = B / fs
    K = Int(round((2 * N * W) - 3))
    println("Using K = $K Slepian basis functions for N = $N, W = $W")
    n_samples = N
    nw = N * W
    # if nw >= (N + 1) / 2
    #     println("SKIPPING: nw = $nw must be less than (N+1)/2 = $((N+1)/2)")
    #     continue
    # end
    # if K<1
    #     println("there is available slepian function that has eigenvalue > 0.99")
    #     continue
    # end

    dpss_basis = dpss(N + 1, N * W, K)
    eigenvalues = dpsseig(dpss_basis, N * W)
    Φ = zeros(N + 1, K)
    for k in 1:K
        Φ[:, k] = dpss_basis[:, k] / norm(dpss_basis[:, k])
    end
    println("DPSS eigenvalues: ", eigenvalues)
    @assert all(eigenvalues .> 0.98) "Not all selected DPSS eigenvalues > 0.98"
    Grad = zeros(Float64, 2 * K * n_qubits)  # Gradient vector size
    # Initial pulse random guess setup
    carrier_freqs = freqs
    samples_matrix = Matrix{Float64}(undef, N + 1, n_qubits)
    samples_matrix[1:N+1, :] .= 2π * 0.0000002 .* (2 .* rand(N + 1, n_qubits) .- 1)
    samples_matrix = samples_matrix .+ im .* samples_matrix

    # Project initial guess to Slepian basis
    coeffs_initial_real = zeros(K, n_qubits)
    coeffs_initial_imag = zeros(K, n_qubits)
    for q in 1:n_qubits
        coeffs_initial_real[:, q] = Φ \ real(samples_matrix[:, q])
        coeffs_initial_imag[:, q] = Φ \ imag(samples_matrix[:, q])
    end
    coeffs_initial = vcat(vec(coeffs_initial_real), vec(coeffs_initial_imag))  # Flat initial vector

    function coeffs_to_samples_matrix(coeffs::Vector{Float64})
        nK = K * n_qubits
        real_flat = coeffs[1:nK]
        imag_flat = coeffs[nK+1:end]
        coeffs_real = reshape(real_flat, K, n_qubits)
        coeffs_imag = reshape(imag_flat, K, n_qubits)
        samples_real = Φ * coeffs_real
        samples_imag = Φ * coeffs_imag
        return samples_real .+ im .* samples_imag
    end

    function costfunction_coeffs(coeffs::Vector{Float64})
        samples_complex = coeffs_to_samples_matrix(coeffs)
        signals = MultiChannelSignal([DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits])
        if penalty
            energy, Ψ_ode = SimpleEvolve.costfunction_ode_with_penalty(
                ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
                basis="qubitbasis", tol_ode=tol_ode)
        else
            energy, Ψ_ode = SimpleEvolve.costfunction_ode(
                ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
                basis="qubitbasis", tol_ode=tol_ode
            )
        end
        return energy
    end

    function gradient_coeffs!(Grad::Vector{Float64}, coeffs::Vector{Float64}; Ω₀=2π * 0.02, λ=1.0)
        samples_complex = coeffs_to_samples_matrix(coeffs)
        signals = MultiChannelSignal([
            DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i])
            for i in 1:n_qubits
        ])
        ∂Ω_real, ∂Ω_imag, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE(
            ψ_initial, T, signals, n_qubits, drives, eigvalues, eigvectors,
            Cost_ham, n_samples;
            basis="qubitbasis", tol_ode=tol_ode
        )
        grad_real = Φ' * ∂Ω_real
        grad_imag = Φ' * ∂Ω_imag
        grad_vec = vcat(vec(grad_real), vec(grad_imag))
        if penalty
            samples_flat = vec(samples_complex)
            sample_penalty_grad = zeros(ComplexF64, size(samples_complex))
            for q in 1:n_qubits
                for ti in 1:N+1
                    x = samples_complex[ti, q] / Ω₀
                    y = abs(x) - 1
                    g = 0im
                    if y > 0
                        h = exp(y - 1 / y)
                        dh_dmag = h * (1 + 1 / y^2) / Ω₀
                        g = dh_dmag * (x / max(abs(x), eps()))
                    end
                    sample_penalty_grad[ti, q] = g
                end
            end
            sample_pen_real = real(sample_penalty_grad)
            sample_pen_imag = imag(sample_penalty_grad)
            pen_coeffs_real = Φ' * sample_pen_real
            pen_coeffs_imag = Φ' * sample_pen_imag
            grad_vec .+= λ * vcat(vec(pen_coeffs_real), vec(pen_coeffs_imag))
        end
        Grad .= grad_vec
        return Grad
    end
    tol_ode = 1e-6
    @time energy_hf = costfunction_coeffs(zeros(length(coeffs_initial)))
    println("Hartree Fock energy ", energy_hf)
    @time energy1 = costfunction_coeffs(coeffs_initial)
    println("initial energy ", energy1)

    COST_THRESHOLD = 1e4
    function safe_cost(x)
        val = costfunction_coeffs(x)
        # if !isfinite(val) || val > COST_THRESHOLD
        #     println("Exiting: cost too large ($val)")
        #     return Inf
        # end
        return val
    end
    function safe_gradient!(g, x)
        val = costfunction_coeffs(x)
        # if !isfinite(val) || val > COST_THRESHOLD
        #     println("Exiting in gradient: cost too large ($val)")
        #     fill!(g, Inf)
        #     return Inf
        # end
        gradient_coeffs!(g, x)
    end

    linesearch = LineSearches.MoreThuente()
    optimizer = Optim.LBFGS(linesearch=linesearch)
    options = Optim.Options(
        show_trace = true,
        show_every = 1,
        f_reltol   = 1e-15,
        g_tol      = 1e-6,
        iterations = 1000,
    )

    tol_ode = 1e-6
    optimization = Optim.optimize(safe_cost, safe_gradient!, coeffs_initial, optimizer, options)
    coeffs_final0 = Optim.minimizer(optimization)

    # val0 = costfunction_coeffs(coeffs_final0)
    # if !isfinite(val0) || val0 > COST_THRESHOLD
    #     println("Initial cost too large ($val0). Skipping...")
    #     continue
    # else
        tol_ode = 1e-8
        optimization = Optim.optimize(safe_cost, safe_gradient!, coeffs_final0, optimizer, options)
        coeffs_final1 = Optim.minimizer(optimization)
    # end

    # val1 = costfunction_coeffs(coeffs_final1)
    # if !isfinite(val1) || val1 > COST_THRESHOLD
    #     println("Initial cost too large ($val1). Skipping...")
    #     continue
    # else
        tol_ode = 1e-10
        optimization = Optim.optimize(safe_cost, safe_gradient!, coeffs_final1, optimizer, options)
    # end
    coeffs_final2 = Optim.minimizer(optimization)

    optimization = Optim.optimize(safe_cost, safe_gradient!, coeffs_final2, optimizer, options)
    coeffs_final = Optim.minimizer(optimization)

    samples_final = coeffs_to_samples_matrix(coeffs_final)
    final_energy = costfunction_coeffs(coeffs_final)
    final_gradient = gradient_coeffs!(Grad, coeffs_final)
    println("Optimization complete for K=$K, N=$N, W=$W, T=$T, seed=$seed, fs=$fs")
    println("Final energy: ", final_energy)
    println("Final gradient norm: ", norm(final_gradient))
    println("Final gap to actual: ", final_energy - real(E_actual))
    @info "Final energy ($final_energy) should be less than Hartree–Fock energy ($energy_hf)"
    @test final_energy < energy_hf
    
    @info "Final energy should be close to actual ground state energy (gap = $(final_energy - real(E_actual)))"
    @test abs(final_energy - real(E_actual)) < 1e-5
    
end