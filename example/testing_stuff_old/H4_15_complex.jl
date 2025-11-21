using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using ForwardDiff: GradientConfig, Chunk
using Random

Cost_ham = npzread("H4_sto-3g_singlet_1.5_P-m.npy") 
# display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 2
SYSTEM="h415"
freqs = 2π*collect(4.8 .+ (0.02 * (1:n_qubits)))
anharmonicities = 2π*0.3 * ones(n_qubits)
coupling_map = Dict{QubitCoupling,Float64}()
for p in 1:n_qubits
    q = (p == n_qubits) ? 1 : p + 1
    coupling_map[QubitCoupling(p,q)] = 2π*0.02
end
device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)

# with detuning 
T=30.0
n_samples = 200
carrier_freqs = freqs.-2π*0.1
#without detuning
# T=40.0
# n_samples = 200
# carrier_freqs = freqs
# δt = T/n_samples
# t_=collect(0:δt:T)


# INITIAL PARAMETERS
samples_matrix = [2π*0.00002* sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
# display(samples_matrix)
samples_matrix = [samples_matrix[:,i] .+ im*samples_matrix[:,i] for i in 1:n_qubits] 
samples_matrix = hcat(samples_matrix...)
pulse_windows=range(0, T, length=n_samples+1)
samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
# carrier_freqs=freqs # on resonance running as complex pulse can optimize without detuning
signals_ = [DigitizedSignal([samples_matrix[:,i]], δt, carrier_freqs[i]) for i in 1:n_qubits]
signals = MultiChannelSignal(signals_)


# # initial state
# initial_state = "1"^(n_qubits÷2) * "0"^(n_qubits÷2)
# ψ_initial_ = zeros(ComplexF64, n_levels^n_qubits)  
# ψ_initial_[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 

ψ_initial = npzread("REF_8_4_2_P-m.npy")
# display(ψ_initial)
H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives =a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
println("Eignvalues of our static Hamiltonian")
display(eigvalues)
tol_ode=1e-6
Λ, U = eigen(Cost_ham)
E_actual = Λ[1]
println("Actual energy: $E_actual") 
# display(drives[1])
# display(eigvectors)
for i in 1:n_qubits
    drives[i] = eigvectors' * drives[i] * eigvectors
end



# we have to optimize the samples in the signal
n_samples_grad = n_samples
δΩ_ = Matrix{Float64}(undef, 2*(n_samples+1), n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
τ = T/n_samples_grad
a=a_q(n_levels)


function gradient_ode!(Grad::Vector{Float64}, samples::Vector{Float64}; λ=1.0, Ω₀=1.0+2π+0.02)
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
    
    

    if n_samples==n_samples_grad
        # Flatten and concatenate gradients for BFGS
        Grad[1:n] = vec(∂Ω_real)
        Grad[n+1:end] = vec(∂Ω_imag)
    else
        Grad = validate_and_expand(δΩ_, grad_ode,
                                                n_samples_grad,
                                                n_samples,
                                                n_qubits, 
                                                T, 
                                                carrier_freqs,
                                                :whittaker_shannon)
    end
    # Step 3: Compute penalty gradient and add to total gradient
    Grad = reshape(Grad, 2*(n_samples+1), n_qubits)

    for k in 1:n_qubits
        for i in 1:n_samples+1
            grad_fidelity = Grad[i, k]
            x = samples_complex[i, k] / Ω₀
            y = abs(x) - 1

            grad_penalty = 0.0
            if y > 0
                h = exp(y - 1 / y)
                dh_dx = h * (1 + 1 / y^2) / Ω₀
                grad_penalty = sign(x) * dh_dx
            end

            Grad_final[i, k] = grad_fidelity + λ * grad_penalty
        end
    end
    return Grad_final
end


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
    # display(size(samples_complex))
    # Build signals from complex samples
    signals_ = [DigitizedSignal(samples_complex[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)
    
    # Compute energy (existing logic)
    energy,ϕ = costfunction_ode_with_penalty(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors,  T,Cost_ham,basis="qubitbasis",tol_ode=tol_ode, λ=1.0)   
    return energy
end

samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
Grad = zeros(Float64, 2*(n_samples+1), n_qubits)
Grad_final = zeros(Float64, 2*(n_samples+1), n_qubits)
samples_0=zeros(length(samples_initial))
@time energy_hf = costfunction_o(samples_0)
println("Hartree Fock energy ",energy_hf)
@time energy1= costfunction_o(samples_initial)   
println("initial energy ",energy1)


# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
optimizer = Optim.BFGS(linesearch=linesearch)
# optimizer = Optim.LBFGS(linesearch=linesearch)
# OPTIMIZATION OPTIONS
options = Optim.Options(
        show_trace = true,
        show_every = 1,
        f_reltol   = 1e-9,
        g_tol      = 1e-9,
        iterations = 1000,
)




optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS

#post processing
n=Int(length(samples_final)/2)
samples_real = reshape(samples_final[1:n], (n_samples+1, n_qubits))
samples_imag = reshape(samples_final[n+1:end], (n_samples+1, n_qubits))
Ω = complex.(samples_real, samples_imag)
Ω= reshape(Ω, n_samples+1, n_qubits)
Ω0=copy(samples_matrix)
pulse_windows=range(0, T, length=n_samples+1)
Ω_plots = plot(                       
    [plot(
            pulse_windows, real.(Ω[:,q])
    ) for q in 1:3]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
Ω_plots_final = plot(                       
    [plot(
            pulse_windows, real.(Ω[:,q])
    ) for q in 4:6]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
plot(Ω_plots, Ω_plots_final, layout=(1,2))

savefig("real_part_final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T).pdf")
Ω_plots = plot(                       
    [plot(
            pulse_windows, imag.(Ω[:,q])
    ) for q in 1:3]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
Ω_plots_final = plot(                       
    [plot(
            pulse_windows, imag.(Ω[:,q])
    ) for q in 4:6]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
plot(Ω_plots, Ω_plots_final, layout=(1,2))

savefig("imag_part_final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T)_on_resonance.pdf")