using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using Random

# Cost_ham = npzread("h207.npy")
Cost_ham = npzread("./qubit_op_H2_sto3g_4qubit.npy")
# Cost_ham = npzread("./h215.npy")
# display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 2
SYSTEM="h207"
freqs = 2π*collect(4.8 .+ (0.02 * (1:n_qubits)))
anharmonicities = 2π*0.3 * ones(n_qubits)
coupling_map = Dict{QubitCoupling,Float64}()
for p in 1:n_qubits
    q = (p == n_qubits) ? 1 : p + 1
    coupling_map[QubitCoupling(p,q)] = 2π*0.02
end
device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)


T=40.0
n_samples = 80
δt = T/n_samples
t_=collect(0:δt:T)
# INITIAL PARAMETERS
# samples_matrix=[2π*0.02* sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
samples_matrix=[ sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 

pulse_windows=range(0, T, length=n_samples+1)
samples_initial=reshape(samples_matrix, :)
carrier_freqs = freqs.-2π*0.1
# signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
signals_ = [DigitizedSignal(samples_matrix[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
signals = MultiChannelSignal(signals_)


# initial state
# initial_state = "1"^(n_qubits÷2) * "0"^(n_qubits÷2)
initial_state = "0011"
ψ_initial_ = zeros(ComplexF64, n_levels^n_qubits)  
ψ_initial_[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 
ψ_initial=copy(ψ_initial_)

H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives =a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
println("Eignvalues of our static Hamiltonian")
display(eigvalues)

tol_ode=1e-6
Λ, U = eigen(Cost_ham)
E_actual = Λ[1]+Λ[2]
println("Actual energy: $E_actual") 
# display(drives[1])
display(eigvectors)
for i in 1:n_qubits
    drives[i] = eigvectors' * drives[i] * eigvectors
end

function costfunction_o(samples)
    # considering the frequency remain as constants
    # samples is a vector of amplitudes as a function of time
    # vector is considered to optimize using BFGS
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    energy,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors,  T,Cost_ham,basis="qubitbasis",tol_ode=tol_ode)   
    return energy
end


# we have to optimize the samples in the signal
n_samples_grad = n_samples
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
τ = T/n_samples_grad


function gradient_ode!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    grad_ode,ψ_ode, σ_ode =SimpleEvolve.gradientsignal_ODE(ψ_initial,
                            T,
                            signals,
                            n_qubits,
                            drives,
                            eigvalues,
                            eigvectors,
                            Cost_ham,
                            n_samples_grad,
                            ∂Ω0;
                            basis="qubitbasis",
                            tol_ode=tol_ode)
    for k in 1:n_qubits
        for i in 1:n_samples+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_ode[i,k] 
        end
    end
    return Grad
end

# OPTIMIZATION ALGORITHM
linesearch = LineSearches.MoreThuente()
# optimizer = Optim.BFGS(linesearch=linesearch)
optimizer = Optim.LBFGS(linesearch=linesearch)
# OPTIMIZATION OPTIONS
options = Optim.Options(
        show_trace = true,
        show_every = 1,
        f_reltol   = 1e-15,
        g_tol      = 1e-12,
        iterations = 200,
)


samples_initial=reshape(samples_matrix, :)
Grad = zeros(Float64, n_samples+1, n_qubits)
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization) 
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)


samples_final_reshaped = reshape(samples_final, n_samples+1, n_qubits)
Ω0=copy(samples_matrix)
Ω = copy(samples_final_reshaped)
pulse_windows=range(0, T, length=n_samples+1)
Ω_plots = plot(                       
    [plot(
            pulse_windows, Ω0[:,q]
    ) for q in 1:n_qubits]...,
    title = "Initial Signals",
    legend = false,
    layout = (n_qubits,1),
)
Ω_plots_final = plot(                       
    [plot(
            pulse_windows, Ω[:,q]
    ) for q in 1:n_qubits]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
plot(Ω_plots, Ω_plots_final, layout=(1,2))

savefig("final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T).pdf")

# getting the ground state
samples = reshape(samples_final, n_samples+1, n_qubits)
signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
signals= MultiChannelSignal(signals_)
energy,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors,  T,Cost_ham,basis="qubitbasis",tol_ode=tol_ode)

# initial_state_excited = "0"^(n_qubits÷2) * "1"^(n_qubits÷2)
# initial_state_excited = "1"^(n_qubits÷2) * "1"^(n_qubits÷2)
initial_state_excited = "0101"
# initial_state_excited = "0"^(n_qubits÷2) * "0"^(n_qubits÷2)
ψ_initial_exc = zeros(ComplexF64, n_levels^n_qubits)
ψ_initial_exc[1 + parse(Int, initial_state_excited, base=n_levels)] = one(ComplexF64)
# ψ_initial_exc = copy(ψ_initial_exc) +ψ_initial
βs = [10.0] 
previous_states = [ϕ]

function costfunction_o_vqd(samples)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_ = [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals = MultiChannelSignal(signals_)
    energy, _ = SimpleEvolve.costfunction_ode_vqd(
        ψ_initial_exc, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham,
        previous_states, βs; basis="qubitbasis", tol_ode=tol_ode
    )
    return energy
end

optimization = Optim.optimize(costfunction_o_vqd, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)
optimization = Optim.optimize(costfunction_o_vqd, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)
optimization = Optim.optimize(costfunction_o_vqd, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)

samples = reshape(samples_final, n_samples+1, n_qubits)
signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
signals= MultiChannelSignal(signals_)
energy_excited, psi_excited = SimpleEvolve.costfunction_ode_vqd(
    ψ_initial_exc, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham,
    previous_states, βs; basis="qubitbasis", tol_ode=tol_ode
)
println("First excited state energy: ", energy_excited)

initial_state_excited = "1001"
ψ_initial_exc = zeros(ComplexF64, n_levels^n_qubits)
ψ_initial_exc[1 + parse(Int, initial_state_excited, base=n_levels)] = one(ComplexF64)

βs = [10.0, 10.0] 
previous_states = [ϕ,psi_excited]
optimization = Optim.optimize(costfunction_o_vqd, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)
optimization = Optim.optimize(costfunction_o_vqd, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)
optimization = Optim.optimize(costfunction_o_vqd, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)
samples = reshape(samples_final, n_samples+1, n_qubits)
signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
signals= MultiChannelSignal(signals_)
energy_excited, psi_excited2 = SimpleEvolve.costfunction_ode_vqd(
    ψ_initial_exc, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham,
    previous_states, βs; basis="qubitbasis", tol_ode=tol_ode
)
println("Second excited state energy: ", energy_excited)