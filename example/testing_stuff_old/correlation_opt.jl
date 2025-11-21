using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using DSP
using Interpolations

# -------------------------------
# Device and system configuration
# -------------------------------
T = 1000.0
n_levels = 2
SYSTEM = "H4"
Cost_ham = npzread("H4_sto-3g_singlet_1.5_P-m.npy") 
n_qubits = round(Int, log2(size(Cost_ham, 1)))
freqs = 2π * collect(4.8 .+ (0.02 * (1:n_qubits)))
anharmonicities = 2π * 0.3 * ones(n_qubits)
coupling_map = Dict{QubitCoupling,Float64}()
for p in 1:n_qubits
    q = (p == n_qubits) ? 1 : p + 1
    coupling_map[QubitCoupling(p, q)] = 2π * 0.02
end
device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)
carrier_freqs = freqs
penalty = true
tol_ode = 1e-9
# -----------------------
# Pulse parameterization
# -----------------------
N = 20                    # Number of pulse sample points
δt = T / N
ratio = 10
N_long = N * ratio
W = 4 / N
K = Int(2 * N_long * W) - 1

# Slepian (DPSS) basis construction
long_dpss = dpss(N_long + 1, N_long * W, K)
Φ = zeros(N + 1, K)
x_long = range(0, stop=1, length=N_long + 1)
x_signal = range(0, stop=1, length=N + 1)
for k in 1:K
    itp = LinearInterpolation(x_long, long_dpss[:, k])
    Φ[:, k] = itp.(x_signal)
    Φ[:, k] /= norm(Φ[:, k])
end

# Initial pulse guess (real + imag, very small amplitude)
samples_matrix = [2π * 0.00002 * sin(2π * t / N) for t in 1:N+1, q in 1:n_qubits]
samples_matrix = samples_matrix .+ im .* samples_matrix

coeffs_initial_real = zeros(K, n_qubits)
coeffs_initial_imag = zeros(K, n_qubits)
for q in 1:n_qubits
    coeffs_initial_real[:, q] = Φ \ real(samples_matrix[:, q])
    coeffs_initial_imag[:, q] = Φ \ imag(samples_matrix[:, q])
end
coeffs_initial = vcat(vec(coeffs_initial_real), vec(coeffs_initial_imag))

ψ_initial = npzread("REF_8_4_2_P-m.npy")
drives = a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(static_hamiltonian(device, n_levels)))
for i in 1:n_qubits
    drives[i] = eigvectors' * drives[i] * eigvectors
end

# ---------------------------------------------------------
# Pulse reconstruction: coefficients vector → pulse samples
# ---------------------------------------------------------
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

# -----------------------------------
# Smoothed Exponential Penalty 
# -----------------------------------
const OMEGA_0 = 2π * 0.03   # Example device constraint (30 MHz)
function penalty_g(x)
    if x < -1
        y = -x - 1
        return exp(y - 1 / y)
    elseif x > 1
        y = x - 1
        return exp(y - 1 / y)
    else
        return 0.0
    end
end

function penalty_grad_g(x)
    if abs(x) > 1
        y = abs(x) - 1
        val = exp(y - 1 / y) * (1 + 1 / y^2)
        return sign(x) * val
    else
        return 0.0
    end
end

function penalty_and_grad(samples_matrix, λ_penalty=1.0)
    penalty = 0.0
    grad_penalty_real = zeros(size(samples_matrix))
    grad_penalty_imag = zeros(size(samples_matrix))
    for t in 1:size(samples_matrix, 1)
        for q in 1:size(samples_matrix, 2)
            x_re = real(samples_matrix[t, q]) / OMEGA_0
            x_im = imag(samples_matrix[t, q]) / OMEGA_0
            penalty += penalty_g(x_re) + penalty_g(x_im)
            grad_penalty_real[t, q] = penalty_grad_g(x_re) / OMEGA_0
            grad_penalty_imag[t, q] = penalty_grad_g(x_im) / OMEGA_0
        end
    end
    penalty *= λ_penalty
    grad_penalty_real .*= λ_penalty
    grad_penalty_imag .*= λ_penalty
    return penalty, grad_penalty_real, grad_penalty_imag
end

# ---------------------------------------------------------
# Overlap Cost function with Penalty
# ---------------------------------------------------------
function costfunction_overlap(coeffs::Vector{Float64}; λ_penalty=1.0)
    samples_complex = coeffs_to_samples_matrix(coeffs)
    signals = MultiChannelSignal([DigitizedSignal(samples_complex[:, i], δt, freqs[i]) for i in 1:n_qubits])
    Ψ_final = SimpleEvolve.evolve_ODE(ψ_initial,T, signals, n_qubits, drives, eigvalues, eigvectors; basis="qubitbasis")
    overlap = abs(dot(ψ_initial, Ψ_final))           # C(T) = |<ψ_initial | Ψ_final>|
    println("Overlap: $overlap")
    penalty, _, _ = penalty_and_grad(samples_complex, λ_penalty)
    return -(overlap - penalty)                      # NEGATE for maximization via minimization!
end

# ---------------------------------------------------------
# Gradient function with Penalty
# ---------------------------------------------------------
function gradient_overlap!(Grad::Vector{Float64}, coeffs::Vector{Float64}; λ_penalty=1.0)
    samples_complex = coeffs_to_samples_matrix(coeffs)
    signals = MultiChannelSignal([DigitizedSignal(samples_complex[:, i], δt, freqs[i]) for i in 1:n_qubits])
    grad_real, grad_imag, ψ_final = SimpleEvolve.grad_overlap_ODE(
        ψ_initial, T, signals, n_qubits, drives, eigvalues, eigvectors, N; basis="qubitbasis"
    )
    _, grad_penalty_real, grad_penalty_imag = penalty_and_grad(samples_complex, λ_penalty)
    grad_vec = vcat(vec(Φ' * (grad_real .- grad_penalty_real)), vec(Φ' * (grad_imag .- grad_penalty_imag)))
    Grad .= -grad_vec                              # NEGATE for maximization
    return Grad
end
function costfunction_coeffs(coeffs::Vector{Float64})
    samples_complex = coeffs_to_samples_matrix(coeffs)
    # Apply lowpass filter before sending to device for frequency bandwidth constraint
    # samples_complex = apply_lowpass!(samples_complex, sample_rate, 4, n_qubits)  # order=8 example

    # Build signals
    signals = MultiChannelSignal([DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits])

    # Compute energy (call SimpleEvolve)
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
# -------------------------------
# Optimization setup & execution
# -------------------------------
λ_penalty = 1.0 
optimizer = Optim.LBFGS()
options = Optim.Options(show_trace=true, iterations=500, g_tol=1e-45, f_tol=1e-99)
optimization = Optim.optimize(coeffs -> costfunction_overlap(coeffs; λ_penalty=λ_penalty),
                             (Grad, coeffs) -> gradient_overlap!(Grad, coeffs; λ_penalty=λ_penalty),
                             coeffs_initial, optimizer, options)
coeffs_final = Optim.minimizer(optimization)
final_energy = costfunction_coeffs(coeffs_final)
println("final energy: $final_energy")
# -------------------------------
# Reconstruct and plot final pulse
# -------------------------------
samples_final = coeffs_to_samples_matrix(coeffs_final)
# plot(samples_final, title="Optimized Pulse Samples (Complex)", xlabel="Time Index", ylabel="Pulse Amplitude")
