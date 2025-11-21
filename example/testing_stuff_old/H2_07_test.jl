using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using ForwardDiff: GradientConfig, Chunk
using Random

Cost_ham = npzread("h207.npy") 
display(Cost_ham)
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


T=10.0
n_samples = 20
δt = T/n_samples
t_=collect(0:δt:T)
# for i in 1:n_samples+1
#     display(t_[i]) 
# end

# INITIAL PARAMETERS
# samples_matrix=[2π*sin(4π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
samples_matrix=[sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
pulse_windows=range(0, T, length=n_samples+1)

samples_initial=reshape(samples_matrix, :)
# carrier_freqs = freqs
# carrier_freqs = [30.207288739056587,30.48828132829821]#this does not work

carrier_freqs = [22.728727738461984,26.20275686353819]
# signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
signals = MultiChannelSignal(signals_)


# using NPZ
# amp0=npzread("./pulses0_amp.npy")
# sample_matrix=amp0
# amp_vec= reshape(samples_matrix, :)
# pulse_windows=range(0, T, length=n_samples+1)
# samples_initial=amp_vec
# # Build MultiChannelSignal
# channels = [
#     DigitizedSignal(
#         sample_matrix[1:n_samples+1, i],  # Amplitude samples for channel i
#         δt,
#         carrier_freqs[i],   # Carrier frequency for channel i
#     ) for i in 1:n_qubits
# ]
# multi_signal = MultiChannelSignal(channels)


# initial state
initial_state = "1"^(n_qubits÷2) * "0"^(n_qubits÷2)
ψ_initial_ = zeros(ComplexF64, n_levels^n_qubits)  
ψ_initial_[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 

# @time energy1,ϕ = costfunction_ode(ψ_initial_, eigvalues, signals, n_qubits, drives,eigvectors, T,Cost_ham;basis="qubitbasis",tol_ode=1e-10)   
# ψ_initial=copy(ϕ)
ψ_initial=copy(ψ_initial_)

H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives =a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
println("Eignvalues of our static Hamiltonian")
display(eigvalues)

tol_ode=1e-10
Λ, U = eigen(Cost_ham)
E_actual = Λ[1]
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

function costfunction_t(samples)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    energy,ϕ =costfunction_trotter(ψ_initial, eigvalues,eigvectors,signals, n_qubits,n_levels, a,Cost_ham,T;basis="qubitbasis",  n_trotter_steps=n_trotter_steps )  
    return energy
end


# we have to optimize the samples in the signal
n_samples_grad = n_samples
δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
τ = T/n_samples_grad
a=a_q(n_levels)
tol_ode=1e-10

function gradient_rotate!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    grad_ode,ψ_rotate, σ_rotate =SimpleEvolve.gradientsignal_rotate(ψ_initial,
                            T,
                            signals,
                            n_qubits,
                            n_levels,
                            a,
                            eigvalues,
                            eigvectors,
                            Cost_ham,
                            n_samples_grad,
                            ∂Ω0;
                            basis="qubitbasis",
                            n_trotter_steps=n_trotter_steps)

    for k in 1:n_qubits
        for i in 1:n_samples+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_ode[i,k] 
        end
    end
    return Grad
end

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

function gradient_direct_exp!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    grad_direct, ψ_direct, σ_direct =gradientsignal_direct_exponentiation(ψ_initial,
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
                            n_trotter_steps=n_trotter_steps)
    for k in 1:n_qubits
        for i in 1:n_samples+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_direct[i,k] 
        end
    end
    return Grad
end
function gradient_fd!(Grad, samples)
    Grad = reshape(Grad, :, n_qubits)
    samples = reshape(samples, n_samples+1, n_qubits)
    signals_= [DigitizedSignal(samples[:,i], δt, carrier_freqs[i]) for i in 1:length(carrier_freqs)]
    signals= MultiChannelSignal(signals_)
    grad_fd= SimpleEvolve.gradientsignal_fd(ψ_initial,
                            T,
                            signals,
                            n_qubits,
                            drives,
                            eigvalues,
                            eigvectors,
                            Cost_ham,
                            n_samples_grad,
                            ∂Ω0;
                            basis = "qubitbasis",
                            ϵ = 5e-6)
    # display(grad_fd)
    for k in 1:n_qubits
        for i in 1:n_samples+1
            # considering the frequency remain as constants
            Grad[i,k] = grad_fd[i,k] 
        end
    end
    return Grad
end


@time energy1,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,eigvectors, T,Cost_ham;basis="qubitbasis",tol_ode=1e-10)   
println("ode evolved energy is ",energy1)

# trotter direct exponentiation evolution
n_trotter_steps = n_samples
@time energy2,ψ_d = costfunction_direct_exponentiation(ψ_initial, eigvalues,eigvectors, signals, n_qubits, drives,Cost_ham, T;basis="qubitbasis", n_trotter_steps=n_trotter_steps)
println("direct evolved energy is ",energy2)


@time energy3,ψ_t = costfunction_trotter(ψ_initial, eigvalues,eigvectors,signals, n_qubits,n_levels, a,Cost_ham,T;basis="qubitbasis",  n_trotter_steps=n_trotter_steps) 
println("trotter evolved energy is ",energy3)


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
        iterations = 200,
)


Grad = zeros(Float64, n_samples+1, n_qubits)
grad_initial=gradient_ode!(Grad, samples_initial)
Grad = zeros(Float64, n_samples+1, n_qubits)
grad_initial__=gradient_rotate!(Grad, samples_initial)
Grad = zeros(Float64, n_samples+1, n_qubits)
grad_initial_direct=gradient_direct_exp!(Grad, samples_initial)
grad_initial_fd=gradient_fd!(Grad, samples_initial)
display(grad_initial_fd)
method = "trotter"
# method = "ode"
# method = "direct"
# method = "rotate_ode"
Ω0=copy(samples_matrix)

pulse_windows=range(0, T, length=n_samples+1)
Ω_plots = plot(                       
    [plot(pulse_windows, Ω0[:,q]) for q in 1:n_qubits]...,
    title = "Initial Signals",
    legend = false,
    layout = (n_qubits,1),
)
grad_plots=plot(                       
    [plot(
            pulse_windows, grad_initial__[:,q]
    ) for q in 1:n_qubits]...,
    title = "Rotate Trotter Gradients",
    legend = false,
    layout = (n_qubits,1),
)
grad_plots2=plot(                       
    [plot(
            pulse_windows, grad_initial[:,q]
    ) for q in 1:n_qubits]...,
    title = " ODE Gradients",
    legend = false,
    layout = (n_qubits,1),
)
grad_plots3=plot(                       
    [plot(
            pulse_windows, grad_initial_fd[:,q]
    ) for q in 1:n_qubits]...,
    title = "fd Gradients",
    legend = false,
    layout = (n_qubits,1),
)

plot(Ω_plots, grad_plots2,grad_plots,grad_plots3, layout=(2,2))
savefig("initial_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T)_$(method).pdf")



samples_initial=reshape(samples_matrix, :)
Grad = zeros(Float64, n_samples+1, n_qubits)
# optimization = Optim.optimize(costfunction_o, gradient_fd!, samples_initial, optimizer, options)
# samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
# optimization = Optim.optimize(costfunction_o, gradient_fd!, samples_final, optimizer, options)
# samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
# optimization = Optim.optimize(costfunction_o, gradient_fd!, samples_final, optimizer, options)
# samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS

optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization) 
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)


# optimization = Optim.optimize(costfunction_t, gradient_rotate!, samples_initial, optimizer, options)
# samples_final = Optim.minimizer(optimization) 
# optimization = Optim.optimize(costfunction_t, gradient_rotate!, samples_final, optimizer, options)
# samples_final = Optim.minimizer(optimization) 
# optimization = Optim.optimize(costfunction_t, gradient_rotate!, samples_final, optimizer, options)
# samples_final = Optim.minimizer(optimization) 

# optimization = Optim.optimize(costfunction_o, gradient_direct_exp!, samples_initial, optimizer, options)
# samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
# optimization = Optim.optimize(costfunction_o, gradient_direct_exp!, samples_final, optimizer, options)
# samples_final = Optim.minimizer(optimization)     # FINAL PARAMETERS
# optimization = Optim.optimize(costfunction_o, gradient_direct_exp!, samples_final, optimizer, options)
# samples_final = Optim.minimizer(optimization)     # FINAL PARAMETERS




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