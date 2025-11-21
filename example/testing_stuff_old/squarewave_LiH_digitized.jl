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

# Complex-amplitude windowed square wave parameters
n_windows = 20
Random.seed!(4)
window_amplitudes = [
    complex.(2 .* rand(n_windows) .- 1, 2 .* rand(n_windows) .- 1) 
    for _ in 1:n_qubits  
]
# window_amplitudes = [
#     [0.1 * exp(-(t/T)^2) * (1 + 0.5im * sin(2π*t/T)) for t in LinRange(0, T, n_windows)]
#     for _ in 1:n_qubits
# ]
window_durations = [fill(T/n_windows, n_windows) for _ in 1:n_qubits]
frequencies = carrier_freqs
duty_cycles = Vector{Float64}(undef, n_qubits)
for i in 1:n_qubits
    duty_cycles[i] = 1.0  # Set duty cycle to 1.0 for all qubits
end

# Create WindowedSquareWave objects
sw_complex = [
    WindowedSquareWave(
        frequencies[q],
        duty_cycles[q],
        window_amplitudes[q],  # Now a vector
        window_durations[q]
    ) for q in 1:n_qubits
]


# Generate complex samples for each qubit
t= collect(0:δt:T)
samples_complex = [
    [SimpleEvolve.amplitude(sw_complex[q], δt) for δt in t] for q in 1:n_qubits
]

samples_matrix = hcat(samples_complex...) 

# Build MultiChannelSignal with complex windowed pulses
channels = [
    DigitizedSignal(
        samples_matrix[:, q],  # Complex samples for channel q
        δt,
        carrier_freqs[q]
    ) for q in 1:n_qubits
]
signals = MultiChannelSignal(channels)

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
savefig("complex_windowed_square_pulses_$(SYSTEM).pdf")

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
final_signals = [
    WindowedSquareWave(
        carrier_freqs[q],
        1.0,  # Duty cycle
        Ω_opt[:, q], 
        fill(T/n_samples, size(Ω_opt, 1)) 
    ) for q in 1:n_qubits
]

# Plot optimized signals
final_samples = [
    [SimpleEvolve.amplitude(final_signals[q], δt) for δt in t] for q in 1:n_qubits
]
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
savefig("optimized_complex_windowed_square_pulses_$(SYSTEM)_$(T)_$(n_samples).pdf")