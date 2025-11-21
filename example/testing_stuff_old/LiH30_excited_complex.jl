using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using Random
using JLD2

println("T=",T)
Cost_ham = npzread("lih30.npy")
# display(Cost_ham)
n_qubits = round(Int, log2(size(Cost_ham,1)))
n_levels = 2
SYSTEM="lih30"
freqs = 2π*collect(4.8 .+ (0.02 * (1:n_qubits)))
anharmonicities = 2π*0.3 * ones(n_qubits)
coupling_map = Dict{QubitCoupling,Float64}()
for p in 1:n_qubits
    q = (p == n_qubits) ? 1 : p + 1
    coupling_map[QubitCoupling(p,q)] = 2π*0.02
end
device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)

# with detuning 
# T=200.0
# n_samples = 1000
# carrier_freqs = freqs.-2π*0.1
#without detuning
# T = 150.0
n_samples = 1000
carrier_freqs = freqs
δt = T/n_samples
t_=collect(0:δt:T)


# INITIAL PARAMETERS
samples_matrix = [2π*0.0000002* sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
# display(samples_matrix)
samples_matrix = [samples_matrix[:,i] .+ im*samples_matrix[:,i] for i in 1:n_qubits] 
samples_matrix = hcat(samples_matrix...)
pulse_windows=range(0, T, length=n_samples+1)
samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
signals_ = [DigitizedSignal([samples_matrix[:,i]], δt, carrier_freqs[i]) for i in 1:n_qubits]
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

n_states=size(ψ_initial,2)
# display(ψ_initial)
H_static = static_hamiltonian(device, n_levels)
#eigenvalues and eigenvectors of the static Hamiltonian
drives =a_fullspace(n_qubits, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))  # Ensures real eigenvalues
println("Eignvalues of our static Hamiltonian")
display(eigvalues)
tol_ode=1e-6
Λ, U = eigen(Cost_ham)
E_actual = Λ[1]+Λ[2]+Λ[3]+Λ[4]+Λ[5]+Λ[6]+Λ[7]
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
# weights=[0.0,0.02,0.04,0.06,0.08,0.1,1.0]
weights=[0.0,0.01,0.02,0.03,0.04,0.05,1.0]
# weights = reverse(collect(1.0:-1.0/(n_states-1):0.0))
println("Weights: ", weights)
weighted=true

function gradient_ode!(Grad::Vector{Float64}, samples::Vector{Float64};
                       λ=1.0, Ω₀=2π+0.02)

    Grad .= 0.0  
    n = length(samples) ÷ 2

    # Split into real/imag parts
    samples_real = samples[1:n]
    samples_imag = samples[n+1:end]

    # Reshape into complex signal matrix
    samples_complex = complex.(
        reshape(samples_real, (n_samples+1, n_qubits)),
        reshape(samples_imag, (n_samples+1, n_qubits))
    )

    # Construct signal object
    signals = MultiChannelSignal([
        DigitizedSignal(samples_complex[:, i], δt, carrier_freqs[i])
        for i in 1:n_qubits
    ])

    # Call gradient computation
    ∂Ω_real, ∂Ω_imag, ψ_ode, σ_ode = SimpleEvolve.gradientsignal_ODE_multiple_states(
        ψ_initial, T, signals, n_qubits, drives, eigvalues, eigvectors,
        Cost_ham, n_samples_grad;
        basis="qubitbasis", tol_ode=tol_ode
    )

    # === Aggregate gradients across states ===
    grad_real = zeros(n_samples+1, n_qubits)
    grad_imag = zeros(n_samples+1, n_qubits)

    for j in 1:n_states
        if weighted
            grad_real .+= weights[j] .* ∂Ω_real[:, :, j]
            grad_imag .+= weights[j] .* ∂Ω_imag[:, :, j]
        else
            grad_real .+= ∂Ω_real[:, :, j]
            grad_imag .+= ∂Ω_imag[:, :, j]
        end
    end

    # === Apply Penalty if Required ===
    grad_final=hcat(vec(grad_real),vec(grad_imag))
    grad_final=reshape(grad_final,:)
    
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
    Grad .=grad_final

    return Grad
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
    
    # Compute energy 
    if penalty==true
        energy, Ψ_ode, energies = SimpleEvolve.costfunction_ode_ssvqe_with_penalty(
        ψ_initial_, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
        basis="qubitbasis", tol_ode=tol_ode,weights=weights, weighted=weighted)
    else
        energy, Ψ_ode, energies = SimpleEvolve.costfunction_ode_ssvqe(
            ψ_initial_, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
            basis="qubitbasis", tol_ode=tol_ode,weights=weights, weighted=weighted
        )
    end
    return energy
end

#initialization 
samples_initial = [real(samples_matrix[:]); imag(samples_matrix[:])]
Grad = zeros(Float64, 2*(n_samples+1), n_qubits)
Grad_final = zeros(Float64, 2*(n_samples+1), n_qubits)
samples_0=zeros(length(samples_initial))
penalty=true


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
        f_reltol   = 1e-12,
        g_tol      = 1e-9,
        iterations = 1000,
)



tol_ode=1e-6
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_initial, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)      # FINAL PARAMETERS
tol_ode=1e-8
optimization = Optim.optimize(costfunction_o, gradient_ode!, samples_final, optimizer, options)
samples_final = Optim.minimizer(optimization)       # FINAL PARAMETERS
tol_ode=1e-10
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
    ) for q in 1:n_qubits]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
Ω_plots_final = plot(                       
    [plot(
            pulse_windows, imag.(Ω[:,q])
    ) for q in 1:n_qubits]...,
    title = "Final Signals",
    legend = false,
    layout = (n_qubits,1),
)
plot(Ω_plots, Ω_plots_final, layout=(1,2))
savefig("final_signals_$(n_qubits)_$(n_levels)_$(SYSTEM)_$(n_samples)_$(T).pdf")


#post-processing
samples=samples_final
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
    
# Compute energy 
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
energies=A.values
@save "samples_final_$(T)_$(SYSTEM)_penalty_on_amps.jld2" samples_final energies

