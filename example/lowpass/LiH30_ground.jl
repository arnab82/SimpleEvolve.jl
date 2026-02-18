using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using Random
using JLD2
using FFTW
using Random
T = 16.0
println("T=", T)
Cost_ham = npzread("./../lih30/lih30.npy")
# display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham, 1)))
n_levels = 2
SYSTEM = "lih30"
freqs = 2π * collect(4.8 .+ (0.02 * (1:n_qubits)))
anharmonicities = 2π * 0.3 * ones(n_qubits)
coupling_map = Dict{QubitCoupling,Float64}()
for p in 1:n_qubits
    q = (p == n_qubits) ? 1 : p + 1
    coupling_map[QubitCoupling(p, q)] = 2π * 0.02
end
device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)

fs=1.0 
n_samples = Int(floor(T * fs))
carrier_freqs = freqs 
δt = T / n_samples


# INITIAL PARAMETERS
samples_matrix = [2π*0.0000002* sin(2π * t / N) for t in 1:N+1, q in 1:n_qubits]  # (N, n_qubits)
samples_matrix = samples_matrix .+ im .* samples_matrix           # Complex pulse at N points
# display(samples_matrix)
samples_matrix = [samples_matrix[:, i] .+ im * samples_matrix[:, i] for i in 1:n_qubits]
samples_matrix = hcat(samples_matrix...)
pulse_windows = range(0, T, length=n_samples + 1)
samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
signals_ = [DigitizedSignal([samples_matrix[:, i]], δt, carrier_freqs[i]) for i in 1:n_qubits]
signals = MultiChannelSignal(signals_)

# Ground state (Hartree-Fock)
initial_state_ground ="0011"# "1100"
ψ_initial_g = zeros(ComplexF64, n_levels^n_qubits)
ψ_initial_g[1+parse(Int, initial_state_ground, base=n_levels)] = 1.0 + 0im
ψ_initial = copy(ψ_initial_g)
# display(ψ_initial)
H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives = a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
println("Eignvalues of our static Hamiltonian")
display(eigvalues)
tol_ode = 1e-6
Λ, U = eigen(Cost_ham)
E_actual = Λ[1]
println("Actual energy: $E_actual")
for i in 1:n_qubits
    drives[i] = eigvectors' * drives[i] * eigvectors
end



# we have to optimize the samples in the signal
n_samples_grad = n_samples
δΩ_ = Matrix{Float64}(undef, 2 * (n_samples + 1), n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad + 1, n_qubits)
τ = T / n_samples_grad
function apply_lowpass!(samples_complex, W, filter_order, n_qubits;fs=1.0)
    cutoff = W/fs
    fft_before = Matrix{ComplexF64}(undef, size(samples_complex, 1), n_qubits)

    for i in 1:n_qubits
        pulse = samples_complex[:, i]
        fft_before[:, i] = fft(pulse)
        filtered_pulse = SimpleEvolve.lowpass_filter(pulse, cutoff, fs; order=filter_order)
        samples_complex[:, i] = filtered_pulse
    end

    return samples_complex
end
W=0.4
function gradient_ode!(Grad::Vector{Float64}, samples::Vector{Float64};
    λ=1.0, Ω₀=2π * 0.02)

    Grad .= 0.0
    n = length(samples) ÷ 2

    # Split into real/imag parts
    samples_real = samples[1:n]
    samples_imag = samples[n+1:end]

    # Reshape into complex signal matrix
    samples_complex = complex.(
        reshape(samples_real, (n_samples + 1, n_qubits)),
        reshape(samples_imag, (n_samples + 1, n_qubits))
    )
    samples_complex = apply_lowpass!(samples_complex, W, 4, n_qubits)  # order=8 example

    # Construct signal object
    signals = MultiChannelSignal([
        DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i])
        for i in 1:n_qubits
    ])

    # Call gradient computation
    ∂Ω_real, ∂Ω_imag, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE(
        ψ_initial, T, signals, n_qubits, drives, eigvalues, eigvectors,
        Cost_ham, n_samples_grad;
        basis="qubitbasis", tol_ode=tol_ode
    )


    # === Apply Penalty if Required ===
    grad_final = hcat(vec(∂Ω_real), vec(∂Ω_imag))
    grad_final = reshape(grad_final, :)

    if penalty
        for i in 1:2*(n_samples+1)
            x = samples[i] / Ω₀
            y = abs(x) - 1
            grad_penalty = 0.0
            if y > 0
                h = exp(y - 1 / y)
                dh_dx = h * (1 + 1 / y^2) / Ω₀
                grad_penalty = sign(x) * dh_dx
            end
            grad_final[i] += λ * grad_penalty
        end
    end
    Grad .= grad_final

    return Grad
end



function costfunction_o(samples::Vector{Float64})
    # Split real vector into real and imaginary components
    n = length(samples) ÷ 2
    samples_real = samples[1:n]
    samples_imag = samples[n+1:end]
    # Reshape into complex matrix (n_samples+1 × n_qubits)
    samples_complex = complex.(
        reshape(samples_real, (n_samples + 1, n_qubits)),
        reshape(samples_imag, (n_samples + 1, n_qubits))
    )
    samples_complex = apply_lowpass!(samples_complex, W, 4, n_qubits)  # order=8 example

    # display(size(samples_complex))
    # Build signals from complex samples
    signals_ = [DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)

    # Compute energy 
    if penalty == true
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

#initialization 
samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
Grad = zeros(Float64, 2 * (n_samples + 1), n_qubits)
Grad_final = zeros(Float64, 2 * (n_samples + 1), n_qubits)
samples_0 = zeros(length(samples_initial))
penalty = true

@time energy_hf = costfunction_o(samples_0)
println("Hartree Fock energy ", energy_hf)
@time energy1 = costfunction_o(samples_initial)
println("initial energy ", energy1)


# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
# optimizer = Optim.BFGS(linesearch=linesearch)
optimizer = Optim.LBFGS(linesearch=linesearch)
# OPTIMIZATION OPTIONS
options = Optim.Options(
    show_trace=true,
    show_every=1,
    f_reltol=1e-12,
    g_tol=1e-8,
    iterations=1000,
)

W=0.4

tol_ode = 1e-6
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
tol_ode = 1e-8
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
tol_ode = 1e-10
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
tol_ode = 1e-10
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS

# calculate final energy and gap
final_energy = costfunction_o(samples_final)
println("Final energy: ", final_energy)
println("Final gap to actual: ", final_energy - real(E_actual))