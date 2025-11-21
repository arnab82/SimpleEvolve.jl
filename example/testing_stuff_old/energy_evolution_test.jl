using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
using Optim
using LineSearches
using ForwardDiff
using ForwardDiff: GradientConfig, Chunk
function energy()
    Cost_ham = npzread("h215.npy") 
    display(Cost_ham)
    n_qubits = round(Int, log2(size(Cost_ham,1)))
    n_levels = 2
    SYSTEM="h215"
    freqs = 2π*collect(4.8 .+ (0.02 * (1:n_qubits)))
    anharmonicities = 2π*0.3 * ones(n_qubits)
    coupling_map = Dict{QubitCoupling,Float64}()
    for p in 1:n_qubits
        q = (p == n_qubits) ? 1 : p + 1
        coupling_map[QubitCoupling(p,q)] = 2π*0.02
    end
    device = Transmon(freqs, anharmonicities, coupling_map, n_qubits)


    T=10.0
    n_samples = 1000
    δt = T/n_samples


    # INITIAL PARAMETERS
    # samples_matrix=[2π*0.02*sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
    samples_matrix=[sin(2π*(t/n_samples)) for t in 0:n_samples,i in 1:n_qubits] 
    pulse_windows=range(0, T, length=n_samples+1)

    samples_initial=reshape(samples_matrix, :)
    # carrier_freqs = freqs
    carrier_freqs = [30.207288739056587,30.48828132829821]

    # signals_ = [DigitizedSignal([2π*0.02* sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
    signals_ = [DigitizedSignal([sin(2π*(t/n_samples)) for t in 0:n_samples], δt, f) for f in carrier_freqs]
    signals = MultiChannelSignal(signals_)


    # initial state
    initial_state = "1"^(n_qubits÷2) * "0"^(n_qubits÷2)
    ψ_initial = zeros(ComplexF64, n_levels^n_qubits)  
    ψ_initial[1 + parse(Int, initial_state, base=n_levels)] = one(ComplexF64) 


    H_static = static_hamiltonian(device, n_levels)
    #eigenvalues and eigenvectors of the static Hamiltonian
    drives =a_fullspace(n_qubits, n_levels)
    eigvalues, eigvecs = eigen(Hermitian(H_static))  # Ensures real eigenvalues
    println("Eignvalues of our static Hamiltonian")
    display(eigvalues)

    tol_ode=1e-6
    Λ, U = eigen(Cost_ham)
    E_actual = Λ[1]
    println("Actual energy: $E_actual") 
    # display(drives[1])
    display(eigvecs)
    for i in 1:n_qubits
        drives[i] = eigvecs' * drives[i] * eigvecs
    end




    # we have to optimize the samples in the signal
    n_samples_grad = n_samples
    δΩ_ = Matrix{Float64}(undef, n_samples+1, n_qubits)
    ∂Ω0 = Matrix{Float64}(undef, n_samples_grad+1, n_qubits)
    τ = T/n_samples_grad
    a=a_q(n_levels)
    tol_ode=1e-6




    @time energy,ϕ = costfunction_ode(ψ_initial, eigvalues, signals, n_qubits, drives,eigvecs, T,Cost_ham;basis="qubitbasis",tol_ode=1e-10)   
    println("ode evolved energy is ",energy)

    # trotter direct exponentiation evolution
    n_trotter_steps = n_samples
    @time energy2,ψ_d = costfunction_direct_exponentiation(ψ_initial, eigvalues,eigvecs, signals, n_qubits, drives,Cost_ham, T;basis="qubitbasis", n_trotter_steps=n_trotter_steps)
    println("direct evolved energy is ",energy2)


    @time energy3,ψ_t = costfunction_trotter(ψ_initial, eigvalues,eigvecs,signals, n_qubits,n_levels, a,Cost_ham,T;basis="qubitbasis",  n_trotter_steps=n_trotter_steps) 
    println("trotter evolved energy is ",energy3)
    println("infidelity between the ode and direct exponentiation")
    display(infidelity(ϕ,ψ_d))
    println("infidelity between the ode and trotter exponentiation")
    display(infidelity(ϕ,ψ_t))
    println("infidelity between the direct and trotter exponentiation")
    display(infidelity(ψ_d,ψ_t))
end
energy()