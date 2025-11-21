using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using Random

Cost_ham = npzread("lih30.npy")

# display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 2
# Π = projector(n_qubits, 2, n_levels)    
# Cost_ham = Hermitian(Π'*Cost_ham*Π)
SYSTEM="lih30"
device = choose_qubits(1:n_qubits, Transmon(
    # 2π*[3.7, 4.2, 3.5, 4.0],
    2π*[4.8, 4.82, 4.84, 4.86],                 # QUBIT RESONANCE FREQUENCIES
    2π*[0.3, 0.3, 0.3, 0.3],                    # QUBIT ANHARMONICITIES
    Dict{QubitCoupling,Float64}(                # QUBIT COUPLING CONSTANTS
        QubitCoupling(1,2) => 2π*.02,
        QubitCoupling(2,3) => 2π*.02,
        QubitCoupling(3,4) => 2π*.02,
        QubitCoupling(1,3) => 2π*.02,
        QubitCoupling(2,4) => 2π*.02,
        QubitCoupling(1,4) => 2π*.02,
    )
))
# freqs= 2π*[3.7, 4.2, 3.5, 4.0]
freqs = 2π*[4.8, 4.82, 4.84, 4.86]
T=40.0
n_samples=200
# n_samples=250
# n_samples=320
n_samples=800
δt = T/n_samples
t_=collect(0:δt:T)

# INITIAL PARAMETERS
# samples_matrix=[2π*0.02*sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
samples_matrix=[2π*0.000002*sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
pulse_windows=range(0, T, length=n_samples+1)

samples_initial=reshape(samples_matrix, :)
carrier_freqs = freqs.-2π*0.2
# carrier_freqs = freqs
signals_ = [DigitizedSignal(samples_matrix[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
signals = MultiChannelSignal(signals_)


# Ground state (Hartree-Fock)
initial_state_ground = "1100"
ψ_initial_g = zeros(ComplexF64, n_levels^n_qubits)
ψ_initial_g[1 + parse(Int, initial_state_ground, base=n_levels)] = 1.0 + 0im
# Excited states (single/double excitations)
excited_configs = ["1010", "1001", "0110", "0101", "0011", "0000"]
ψ_initial_excited = [
    zeros(ComplexF64, n_levels^n_qubits)
    for _ in 1:length(excited_configs)
]
for (i, config) in enumerate(excited_configs)
    ψ_initial_excited[i][1 + parse(Int, config, base=n_levels)] = 1.0 + 0im
end
# Combine into matrix for SSVQE
Ψ0 = hcat(ψ_initial_g, ψ_initial_excited...)
ψ_initial_ =ψ_initial= copy(Ψ0)

H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives =a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
println("Eignvalues of our static Hamiltonian")
display(eigvalues)
tol_ode=1e-8
Λ, U = eigen(Cost_ham)
E_actual = Λ[1]+Λ[2]+Λ[3]+Λ[4]+Λ[5]+Λ[6]#+Λ[7]+Λ[8]+Λ[9]+Λ[10]+Λ[11]+Λ[12]+Λ[13]+Λ[14]+Λ[15]+ Λ[16]
println("Actual energy: $E_actual") 
# display(drives[1])
# display(eigvectors)
for i in 1:n_qubits
    drives[i] = eigvectors' * drives[i] * eigvectors
end

@time energy1,ϕ,energies = SimpleEvolve.costfunction_ode_ssvqe(ψ_initial_, eigvalues, signals, n_qubits, drives,eigvectors, T,Cost_ham;basis="qubitbasis",tol_ode=1e-10)   
println("ode evolved energy is ",energy1)
n_states= size(ψ_initial_, 2)
"""weights = reverse(collect(1.0:-1.0/(n_states-1):0.0))
 1.6640986881810704e-6
 -4.111047537946888e-7
 1.291442337958415e-6
-2.0273108551904784e-6
 3.911929381317236e-6
 0.22988115753196592
 but if I include an extra state
    then it resolves and the error shows up in the extra state"""
weights = reverse(collect(1.0:-0.2/(n_states-1):0.0))
println("Weights: ", weights)
weighted=true
function costfunction_o(samples)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_ = [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)
    energy, Ψ_ode, energies = SimpleEvolve.costfunction_ode_ssvqe(
        ψ_initial_, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
        basis="qubitbasis", tol_ode=tol_ode,weights=weights, weighted=weighted
    )
    return energy
end


# Initial Hf LEVEL energy
samples= zeros(Float64, n_samples+1, n_qubits)
samples=reshape(samples, :)
energy_initial = costfunction_o(samples)
println("Initial hf LEVEL energy: $energy_initial")


# we have to optimize the samples in the signal
n_samples_grad = n_samples
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
τ = T/n_samples_grad
a=a_q(n_levels)

function gradient_ode!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    Grad .= 0.0
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    grad_ode,ψ_ode, σ_ode =SimpleEvolve.gradientsignal_ODE_real_multiple_states(ψ_initial,
                            T,
                            signals,
                            n_qubits,
                            drives,
                            eigvalues,
                            eigvectors,
                            Cost_ham,
                            n_samples_grad;
                            basis="qubitbasis",
                            tol_ode=tol_ode)
    dim, n_states = size(ψ_ode)
    weights_states = weights

    for k in 1:n_qubits
        for i in 1:n_samples+1
            if weighted==true
                Grad[i, k] = sum(weights_states[j] * grad_ode[i, k, j] for j in 1:n_states)
            else
                Grad[i, k] = sum(grad_ode[i, k, j] for j in 1:n_states)
            end
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
        f_reltol   = 1e-11,
        g_tol      = 1e-9,
        iterations = 1000,
)


tol_ode=1e-4
samples_initial=reshape(samples_matrix, :)
Grad = zeros(Float64, n_samples+1, n_qubits)
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
tol_ode=1e-6
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
tol_ode=1e-8
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization) 
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)
tol_ode=1e-10
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


samples=samples_final
samples = reshape(samples, n_samples+1, n_qubits)
signals_ = [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:n_qubits]
signals = MultiChannelSignal(signals_)
energy, Ψ_ode, energies = SimpleEvolve.costfunction_ode_ssvqe(
        ψ_initial_, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
        basis="qubitbasis", tol_ode=tol_ode,weights=weights, weighted=weighted)
a = zeros(Float64, n_states, n_states)
for i in 1:n_states
    for j in i:n_states
        a[i, j] = real(Ψ_ode[:,i]' * Cost_ham * Ψ_ode[:,j])
        a[j, i] = a[i, j]  # Ensure symmetry
    end
end
A=eigen(a)
B=[Λ[1]
    Λ[2]
    Λ[3]
    Λ[4]
    Λ[5]
    Λ[6]]
display(A.values[1:6].-B)
Λ[1],Λ[2],Λ[3],Λ[4],Λ[5],Λ[6],Λ[7],Λ[8],Λ[9],Λ[10],Λ[11],Λ[12],Λ[13],Λ[14],Λ[15], Λ[16]
