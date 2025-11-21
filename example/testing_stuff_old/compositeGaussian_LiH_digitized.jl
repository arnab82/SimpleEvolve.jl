using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using ForwardDiff: GradientConfig, Chunk
using Random
Cost_ham = npzread("lih30.npy")
# display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 2
SYSTEM="lih30"
device = choose_qubits(1:n_qubits, Transmon(
    # 2π*[4.8, 4.84, 4.86, 4.88],                    # QUBIT RESONANCE FREQUENCIES
    2π*[3.7, 4.2, 3.5, 4.0],                    # QUBIT RESONANCE FREQUENCIES
    2π*[0.3, 0.3, 0.3, 0.3],                    # QUBIT ANHARMONICITIES
    Dict{QubitCoupling,Float64}(                # QUBIT COUPLING CONSTANTS
        QubitCoupling(1,2) => 2π*.018,
        QubitCoupling(2,3) => 2π*.021,
        QubitCoupling(3,4) => 2π*.020,
        QubitCoupling(1,3) => 2π*.021,
        QubitCoupling(2,4) => 2π*.020,
        QubitCoupling(1,4) => 2π*.021,
    )
))

# freqs = 2π*[4.8, 4.84, 4.86, 4.88]
freqs = 2π*[3.7, 4.2, 3.5, 4.0]
carrier_freqs = freqs.-2π*0.1  
T=30.0
n_samples = 200
δt = T/n_samples

function costfunction_o(samples::Vector{Float64})
    # Split real vector into real and imaginary components
    n = length(samples) ÷ 2
    samples_real = samples[1:n]
    samples_imag = samples[n+1:end]
    
    # Reshape into complex matrix (n_samples+1 × n_qubits)
    samples_complex = complex.(
        reshape(samples_real, (n_samples+1, n_qubits)),
        reshape(samples_imag, (n_samples+1, n_qubits))
    )
    
    # Build signals from complex samples
    signals_ = [DigitizedSignal(samples_complex[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)
    
    # Compute energy (existing logic)
    energy, _ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
                               basis="qubitbasis", tol_ode=tol_ode)
    return energy
end

n_samples_grad = n_samples
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
τ = T/n_samples_grad
a=a_q(n_levels)
tol_ode=1e-6
drives =a_fullspace(n_qubits, n_levels)
eigvectors = eigvecs(Cost_ham)
for i in 1:n_qubits
    drives[i] = eigvectors' * drives[i] * eigvectors
end

function gradient_ode!(Grad::Vector{Float64}, samples::Vector{Float64})
    # Split real vector into real and imaginary components
    n = length(samples) ÷ 2
    samples_real = samples[1:n]
    samples_imag = samples[n+1:end]
    
    # Reshape into complex matrix
    samples_complex = complex.(
        reshape(samples_real, (n_samples+1, n_qubits)),
        reshape(samples_imag, (n_samples+1, n_qubits))
    )
    
    # Build signals from complex samples
    signals_ = [DigitizedSignal(samples_complex[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)
    
    # Compute real and imaginary gradients
    ∂Ω_real, ∂Ω_imag, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE(
                                                ψ_initial,
                                                T,
                                                signals,
                                                n_qubits,
                                                drives,
                                                eigvalues,
                                                eigvectors,
                                                Cost_ham,
                                                n_samples_grad; 
                                                basis="qubitbasis", 
                                                tol_ode=tol_ode
    )
    
    # Flatten and concatenate gradients for BFGS
    Grad[1:n] = vec(∂Ω_real)
    Grad[n+1:end] = vec(∂Ω_imag)
    return Grad
end




# initial state
initial_state = "1"^(n_qubits÷2) * "0"^(n_qubits÷2)
ψ_initial = zeros(ComplexF64, n_levels^n_qubits)  
ψ_initial[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 
H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
println("Eignvalues of our static Hamiltonian")
display(eigvalues)
# fci energy
E,ϕ=eigen(Cost_ham )
println("FCI energy: ", E[1])


Random.seed!(4)  
n_windows = 5
phases_1 = 0.0 * rand(n_windows)
phases_2 = π * rand(n_windows)
phases_3 = π/2 * rand(n_windows)
phases_4 = π/4 * rand(n_windows)
centers = LinRange(T/(n_windows+1), T-T/(n_windows+1), n_windows)
widths = fill(T/(6n_windows), n_windows)
# amplitudes = complex.(2 .* rand(n_windows) .- 1, 2 .* rand(n_windows) .- 1)  
amplitudes = complex.(2 .* rand(n_windows) , 2 .* rand(n_windows) )  
frequencies_1 = carrier_freqs[1] * ones(n_windows)  # Constant frequency for all windows
frequencies_2 = carrier_freqs[2] * ones(n_windows)  
frequencies_3 = carrier_freqs[3] * ones(n_windows)  
frequencies_4 = carrier_freqs[4] * ones(n_windows)
pulse_1 = SimpleEvolve.WindowedGaussianPulse(amplitudes, centers, widths,phases_1, frequencies_1)
pulse_2 = SimpleEvolve.WindowedGaussianPulse(amplitudes, centers, widths,phases_2, frequencies_2)
pulse_3 = SimpleEvolve.WindowedGaussianPulse(amplitudes, centers, widths,phases_3, frequencies_3)
pulse_4 = SimpleEvolve.WindowedGaussianPulse(amplitudes, centers, widths,phases_4, frequencies_4)

τ = δt
t = 0:τ:T
t_=0:δt:T
samples_1 = [SimpleEvolve.amplitude(pulse_1, τ) for τ in t]
samples_2 = [SimpleEvolve.amplitude(pulse_2, τ) for τ in t]
samples_3 = [SimpleEvolve.amplitude(pulse_3, τ) for τ in t]
samples_4 = [SimpleEvolve.amplitude(pulse_4, τ) for τ in t]
samples_matrix = hcat(samples_1, samples_2, samples_3, samples_4)
plot(t, real.(samples_1), label="Real pulse 1",color=:blue)
plot!(t, imag.(samples_1), label="Imag pulse 1", color=:orange)
plot!(t, real.(samples_2), label="Real Pulse 2", color=:red, linewidth=1.5)
plot!(t, imag.(samples_2), label="Imag Pulse 2", color=:green)
plot!(t, real.(samples_3), label="Real Pulse 3", color=:purple, linewidth=1.5)
plot!(t, imag.(samples_3), label="Imag Pulse 3", color=:brown)
plot!(t, real.(samples_4), label="Real Pulse 4", color=:cyan, linewidth=1.5)
plot!(t, imag.(samples_4), label="Imag Pulse 4", color=:magenta)
xlabel!("Time")
ylabel!("Amplitude")
title!("Windowed Gaussian Pulse with Frequency")
savefig("windowed_gaussian_pulses_$(SYSTEM).pdf")

# Create DigitizedSignal objects

signals_ =[
    pulse_1, pulse_2, pulse_3, pulse_4
]
signals= MultiChannelSignal(signals_)

wGP_samples = [
    samples_matrix[:, q] for q in 1:n_qubits
]
samples_matrix = hcat(wGP_samples...)

# Plot real and imaginary parts
pulse_windows = range(0, T, length=n_samples+1)
Ω_plots = plot(
    [plot(pulse_windows, real.(samples_matrix[:, q]), title="Qubit $q (Real)") for q in 1:n_qubits]...,
    layout=(n_qubits, 1), legend=false
)

Ω_plots_imag = plot(
    [plot(pulse_windows, imag.(samples_matrix[:, q]), title="Qubit $q (Imag)") for q in 1:n_qubits]...,
    layout=(n_qubits, 1), legend=false
)

plot(Ω_plots, Ω_plots_imag, layout=(1, 2))
savefig("complex_windowed_pulses_$(SYSTEM).pdf")

# Use your existing optimization setup
samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
optimizer = Optim.BFGS(linesearch=LineSearches.MoreThuente())
options = Optim.Options(show_trace=true, iterations=100,f_reltol=1e-11, g_tol=1e-9)

# Optimize using ODE gradients (already complex-compatible)
result = Optim.optimize(
    costfunction_o,
    gradient_ode!,
    samples_initial,
    optimizer,
    options
)
samples_final = result.minimizer
result = Optim.optimize(
    costfunction_o,
    gradient_ode!,
    samples_final,
    optimizer,
    options
)
Ω_opt = reshape(result.minimizer,2*( n_samples+1), n_qubits)
n = (n_samples + 1) * n_qubits
samples_real = reshape(result.minimizer[1:n], (n_samples+1, n_qubits))
samples_imag = reshape(result.minimizer[n+1:end], (n_samples+1, n_qubits))
Ω_opt = complex.(samples_real, samples_imag)
# Display optimization results
println("Optimization result:")
println("Minimum cost: ", result.minimum)
# println("Optimized samples:")
# println(Ω_opt)
#plot final signals


# Plot optimized signals
final_samples = [Ω_opt[:, q] for q in 1:n_qubits]
final_samples_matrix = hcat(final_samples...)
final_Ω_plots = plot(
    [plot(pulse_windows, real.(final_samples_matrix[:, q]), title="Qubit $q (Real)") for q in 1:n_qubits]...,
    layout=(n_qubits, 1), legend=false
)
final_Ω_plots_imag = plot(
    [plot(pulse_windows, imag.(final_samples_matrix[:, q]), title="Qubit $q (Imag)") for q in 1:n_qubits]...,
    layout=(n_qubits, 1), legend=false
)
plot(final_Ω_plots, final_Ω_plots_imag, layout=(1, 2))
savefig("optimized_complex_windowed_gaussian_pulses_$(SYSTEM)_$(n_samples)_$(T).pdf")