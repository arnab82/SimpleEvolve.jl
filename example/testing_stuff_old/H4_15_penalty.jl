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

#with detuning
T=40.0
n_samples=400
carrier_freqs = freqs.-2π*0.1 
# without detuning
# T=50.0
# n_samples=400
# carrier_freqs = freqs

δt = T/n_samples
t_=collect(0:δt:T)


# INITIAL PARAMETERS
samples_matrix=[2π*0.00002* sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
pulse_windows=range(0, T, length=n_samples+1)
samples_initial=reshape(samples_matrix, :)

carrier_freqs = freqs
# signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
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
# display(eigvalues)

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
n_samples_grad = Int(n_samples/2)
δΩ_ = zeros(Float64, n_samples+1, n_qubits)
∂Ω0 = zeros(Float64, n_samples_grad+1, n_qubits)
τ = T/n_samples_grad
a=a_q(n_levels)



function gradient_ode_opt_penalty!(Grad, samples; λ=1.0, Ω₀=2π+0.02)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)

    # Step 1: Reconstruct MultiChannelSignal
    signals_ = [DigitizedSignal(samples[:, i], δt, carrier_freqs[i]) for i in 1:n_qubits]
    signals = MultiChannelSignal(signals_)

    # Step 2: Compute fidelity gradient
    grad_ode, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE_real(ψ_initial,
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
    grad_ode_expanded = Matrix{Float64}(undef, n_samples+1, n_qubits)
    if n_samples==n_samples_grad
        for k in 1:n_qubits
            grad_ode_expanded[:,k] = grad_ode[:, k]
        end
    else
        grad_ode_expanded = validate_and_expand(δΩ_, grad_ode,
                                                n_samples_grad,
                                                n_samples,
                                                n_qubits, 
                                                T, 
                                                carrier_freqs,
                                                :whittaker_shannon)
    end
    # Step 3: Compute penalty gradient and add to total gradient
    for k in 1:n_qubits
        for i in 1:n_samples+1
            grad_fidelity = grad_ode_expanded[i, k]
            x = samples[i, k] / Ω₀
            y = abs(x) - 1

            grad_penalty = 0.0
            if y > 0
                h = exp(y - 1 / y)
                dh_dx = h * (1 + 1 / y^2) / Ω₀
                grad_penalty = sign(x) * dh_dx
            end

            Grad[i, k] = grad_fidelity + λ * grad_penalty
        end
    end
    Grad=reshape(Grad, (n_samples+1)*n_qubits)
    return Grad

end


function costfunction_ode_opt(samples)
    # considering the frequency remain as constants
    # samples is a vector of amplitudes as a function of time
    # vector is considered to optimize using BFGS
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    energy,ϕ = costfunction_ode_with_penalty(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors,  T,Cost_ham,basis="qubitbasis",tol_ode=tol_ode, λ=1.0)   
    return energy
end

samples_initial=reshape(samples_matrix, :)
Grad = zeros(Float64, n_samples+1, n_qubits)
samples_0=zeros(length(samples_initial))
# display(samples_0)
@time energy_hf = costfunction_ode_opt(samples_0)
println("Hartree Fock energy ",energy_hf)
@time energy1= costfunction_ode_opt(samples_initial)   
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




optimization = Optim.optimize(costfunction_ode_opt, gradient_ode_opt_penalty!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_ode_opt, gradient_ode_opt_penalty!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_ode_opt, gradient_ode_opt_penalty!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
tol_ode=1e-8
optimization = Optim.optimize(costfunction_ode_opt, gradient_ode_opt_penalty!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
tol_ode=1e-10
optimization = Optim.optimize(costfunction_ode_opt, gradient_ode_opt_penalty!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS

samples_final_reshaped = reshape(samples_final, n_samples+1, n_qubits)
Ω0=copy(samples_matrix)
Ω = copy(samples_final_reshaped)
pulse_windows=range(0, T, length=n_samples+1)
Ω_plots = plot(                       
    [plot(
            pulse_windows, Ω[:,q]
    ) for q in 1:3]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
Ω_plots_final = plot(                       
    [plot(
            pulse_windows, Ω[:,q]
    ) for q in 4:6]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
plot(Ω_plots, Ω_plots_final, layout=(1,2))

savefig("final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T).pdf")