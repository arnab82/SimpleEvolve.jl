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
T = 22.0
println("T=", T)
Cost_ham = npzread("lih30.npy") 
n_qubits = round(Int, log2(size(Cost_ham, 1)))
println("n_qubits=", n_qubits)
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


# ----- PARAMETERS -----
N = 10                # Number of pulse sample points (control grid)
δt= T / N             # Time step
ratio = 10          # Oversampling factor for DPSS generation
N_long = N * ratio    # Dense grid for smoothness
W = 2 / N             # Time-bandwidth product
K = Int(2* N_long * W) - 1  # Number of basis functions to keep (bandwidth)

# ----- BUILD SLEPIAN (DPSS) BASIS -----
long_dpss = dpss(N_long+1, N_long * W, K)  # (N_long, K)
Φ = zeros(N+1, K)
x_long = range(0, stop=1, length=N_long+1)
x_signal = range(0, stop=1, length=N+1)

for k in 1:K
    itp = LinearInterpolation(x_long, long_dpss[:,k])
    Φ[:, k] = itp.(x_signal)
    Φ[:, k] /= norm(Φ[:, k])         # Normalize columns
end

# ----- INITIAL PULSE -----
carrier_freqs = freqs
samples_matrix = [2π*0.0000002* sin(2π * t / N) for t in 1:N+1, q in 1:n_qubits]  # (N, n_qubits)
samples_matrix = samples_matrix .+ im .* samples_matrix           # Complex pulse at N points

# ----- PROJECT INITIAL GUESS TO SLEPIAN SPACE -----
coeffs_initial_real = zeros(K, n_qubits)
coeffs_initial_imag = zeros(K, n_qubits)
for q in 1:n_qubits
    coeffs_initial_real[:, q] = Φ \ real(samples_matrix[:, q])
    coeffs_initial_imag[:, q] = Φ \ imag(samples_matrix[:, q])
end

coeffs_initial = vcat(vec(coeffs_initial_real), vec(coeffs_initial_imag))  # Flat initial vector

# -------------------------
# Ground state (Hartree-Fock)
# initial_state_ground = "1100"
initial_state_ground = "0011"
ψ_initial_g = zeros(ComplexF64, n_levels^n_qubits)
ψ_initial_g[1+parse(Int, initial_state_ground, base=n_levels)] = 1.0 + 0im
ψ_initial = copy(ψ_initial_g)

H_static = static_hamiltonian(device, n_levels)
drives = a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))
println("Eignvalues of our static Hamiltonian")
display(eigvalues)
tol_ode = 1e-6
Λ, U = eigs(Cost_ham)
E_actual = Λ[1]
println("Actual energy: $E_actual")

for i in 1:n_qubits
    drives[i] = eigvectors' * drives[i] * eigvectors
end

# ----- HELPER: RECONSTRUCT PULSE FROM COEFFS -----
function coeffs_to_samples_matrix(coeffs::Vector{Float64})
    nK = K * n_qubits
    real_flat = coeffs[1:nK]
    imag_flat = coeffs[nK+1:end]
    coeffs_real = reshape(real_flat, K, n_qubits)
    coeffs_imag = reshape(imag_flat, K, n_qubits)
    samples_real = Φ * coeffs_real
    samples_imag = Φ * coeffs_imag
    return samples_real .+ im .* samples_imag  # (N, n_qubits)
end
function apply_lowpass!(samples_complex, sample_rate, filter_order, n_qubits)
    fs = 1 / sample_rate
    cutoff = fs / 2.0001

    fft_before = Matrix{ComplexF64}(undef, size(samples_complex, 1), n_qubits)

    for i in 1:n_qubits
        pulse = samples_complex[:, i]
        fft_before[:, i] = fft(pulse)
        filtered_pulse = SimpleEvolve.lowpass_filter(pulse, cutoff, fs; order=filter_order)
        samples_complex[:, i] = filtered_pulse
    end

    return samples_complex
end
penalty = true
sample_rate=0.2
n_samples = N 
# ----- COST FUNCTION -----
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
∂Ω_real= zeros(Float64, N+1, n_qubits)
∂Ω_imag= zeros(Float64, N+1, n_qubits)
# ----- GRADIENT FUNCTION -----
function gradient_coeffs!(Grad::Vector{Float64}, coeffs::Vector{Float64};Ω₀=2π*0.02,λ=1.0)
    samples_complex = coeffs_to_samples_matrix(coeffs)
    # Apply lowpass filter before sending to device
    # samples_complex = apply_lowpass!(samples_complex, sample_rate, 4, n_qubits)  # order=8 example

    # build signal objects (DigitizedSignal expects vector of complex samples)
    signals = MultiChannelSignal([
        DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i])
        for i in 1:n_qubits
    ])

    # Call gradient routine from SimpleEvolve (returns time-domain gradients)
    # We expect ∂Ω_real, ∂Ω_imag each of size (n_time, n_qubits)
    ∂Ω_real, ∂Ω_imag, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE(
        ψ_initial, T, signals, n_qubits, drives, eigvalues, eigvectors,
        Cost_ham, n_samples;
        basis="qubitbasis", tol_ode=tol_ode
    )

    # Project time-domain gradients onto Slepian basis:
    # grad_coeffs_real(:, q) = Φ' * ∂Ω_real[:, q] (size K)
    grad_real = Φ' * ∂Ω_real    # K × n_qubits
    grad_imag = Φ' * ∂Ω_imag    # K × n_qubits

    # Grad is a flat vector of size 2 * K * n_qubits
    grad_vec = vcat(vec(grad_real), vec(grad_imag))
    # Apply amplitude penalty (mapped to coefficient space)
    if penalty
        # We apply a simple amplitude penalty by mapping time-domain amplitude to coefficients gradient.
        # A direct mapping of the sample-space penalty used originally is more involved (nonlinear), but
        # here we project the sample-space penalty gradient into coefficient space as well.
        # Compute sample amplitudes and penalty gradient in sample space, then project:
        samples_flat = vec(samples_complex)                 # (n_time*n_qubits)
        # build sample penalty gradient (same shape)
        sample_penalty_grad = zeros(ComplexF64, size(samples_complex))
        for q in 1:n_qubits
            for ti in 1:N+1
                x = samples_complex[ti, q] / Ω₀
                y = abs(x) - 1
                g = 0.0 + 0.0im
                if y > 0
                    h = exp(y - 1 / y)
                    # derivative of h wrt x magnitude: dh_d|x| = h * (1 + 1/y^2) * (1/Ω₀)
                    dh_dmag = h * (1 + 1 / y^2) / Ω₀
                    # derivative wrt complex x: dh/dx = dh_dmag * x/|x|
                    g = dh_dmag * (x / max(abs(x), eps()))
                end
                sample_penalty_grad[ti, q] = g
            end
        end
        # project sample penalty grad to coefficient space
        # separate into real/imag parts
        sample_pen_real = real(sample_penalty_grad)
        sample_pen_imag = imag(sample_penalty_grad)
        pen_coeffs_real = Φ' * sample_pen_real
        pen_coeffs_imag = Φ' * sample_pen_imag
        grad_vec .+= λ * vcat(vec(pen_coeffs_real), vec(pen_coeffs_imag))
    end

    Grad .= grad_vec
    return Grad
end
@time energy_hf = begin
    # zero coefficients for HF baseline
    zeros_coeffs = zeros(length(coeffs_initial))
    costfunction_coeffs(zeros_coeffs)
end
println("Hartree Fock energy ", energy_hf)
@time energy1 = costfunction_coeffs(coeffs_initial)
println("initial energy ", energy1)
# -------------------------
# optimization setup (optimize coefficients)
# -------------------------
linesearch = LineSearches.MoreThuente()
optimizer = Optim.LBFGS(linesearch=linesearch)
options = Optim.Options(
    show_trace=true,
    show_every=1,
    f_reltol=1e-12,
    g_tol   =1e-8,
    iterations=1000,
)
# multi-stage tightening of ODE tolerance, as in your original script
tol_ode = 1e-4
optimization = Optim.optimize(costfunction_coeffs, gradient_coeffs!, coeffs_initial, optimizer, options)
coeffs_final = Optim.minimizer(optimization)
tol_ode = 1e-6
optimization = Optim.optimize(costfunction_coeffs, gradient_coeffs!, coeffs_final, optimizer, options)
coeffs_final = Optim.minimizer(optimization)
tol_ode = 1e-8
optimization = Optim.optimize(costfunction_coeffs, gradient_coeffs!, coeffs_final, optimizer, options)
coeffs_final = Optim.minimizer(optimization)
tol_ode = 1e-10
optimization = Optim.optimize(costfunction_coeffs, gradient_coeffs!, coeffs_final, optimizer, options)
coeffs_final = Optim.minimizer(optimization)

# ----- RECONSTRUCT FINAL PULSE --------
samples_final = coeffs_to_samples_matrix(coeffs_final)  # (N, n_qubits)
@save "results_slepian_LiH_K_$(K).jld2" samples_final coeffs_final 
pulse_windows = range(0, 1, length=N)
Ω = samples_final
pulse_windows = range(0, T, length=N+1)  # Time windows for plotting

Ω_plots = plot(
    [plot(pulse_windows, real.(Ω[:, q])) for q in 1:n_qubits]...,
    title="Final Signals (real)",
    legend=false,
    layout=(n_qubits, 1),
)
Ω_plots_final = plot(
    [plot(pulse_windows, imag.(Ω[:, q])) for q in 1:n_qubits]...,
    title="Final Signals (imag)",
    legend=false,
    layout=(n_qubits, 1),
)
plot(Ω_plots, Ω_plots_final, layout=(1, 2))
savefig("final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T)_slepian_K_$(K).pdf")
