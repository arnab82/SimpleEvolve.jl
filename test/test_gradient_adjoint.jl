using SimpleEvolve
using LinearAlgebra
using NPZ
using Random
using Test

# Regression tests for the adjoint (costate) gradient, gradientsignal_ODE.
#
# 1. Shift invariance (no finite differences): for normalized states, H -> H + c*I shifts the energy
#    by c only, so the gradient must not change. Normalizing the costate C|psi(T)> breaks this.
# 2. Element-wise agreement with central finite differences of costfunction_ode at tight ODE
#    tolerance. The analytic gradient evaluates the gradient signal at the sample points (it does
#    not integrate it against the interpolation hat functions), so agreement is ~1%, not 1e-8.

function setup_lih(; T=10.0, seed=1)
    Cost_ham = npzread(joinpath(@__DIR__, "lih15.npy"))
    n_qubits = round(Int, log2(size(Cost_ham, 1)))
    freqs = 2π * collect(4.8 .+ (0.02 * (1:n_qubits)))
    coupling_map = Dict{QubitCoupling,Float64}()
    for p in 1:(n_qubits-1)
        coupling_map[QubitCoupling(p, p + 1)] = 2π * 0.02
    end
    device = Transmon(freqs, 2π * 0.3 * ones(n_qubits), coupling_map, n_qubits)
    ψ0 = zeros(ComplexF64, 2^n_qubits)
    ψ0[1 + parse(Int, "0011", base=2)] = 1
    eigvalues, eigvectors = eigen(Hermitian(static_hamiltonian(device, 2)))
    drives = a_fullspace(n_qubits, 2)
    for i in 1:n_qubits
        drives[i] = eigvectors' * drives[i] * eigvectors
    end
    N = Int(T)
    δt = T / N
    Random.seed!(seed)
    S = 2π * 0.01 .* ((2 .* rand(N + 1, n_qubits) .- 1) .+ im .* (2 .* rand(N + 1, n_qubits) .- 1))
    signals(S) = MultiChannelSignal([DigitizedSignal(S[:, i], δt, freqs[i]) for i in 1:n_qubits])
    return (; Cost_ham, n_qubits, ψ0, eigvalues, eigvectors, drives, N, T, S, signals)
end

function analytic_gradient(p, H)
    gR, gI, _, _ = SimpleEvolve.gradientsignal_ODE(p.ψ0, p.T, p.signals(p.S), p.n_qubits, p.drives,
                                                   p.eigvalues, p.eigvectors, H, p.N;
                                                   basis="qubitbasis", tol_ode=1e-12)
    return vcat(vec(gR), vec(gI))
end

@testset "Adjoint gradient (gradientsignal_ODE)" begin
    p = setup_lih()

    @testset "invariance under H -> H + c*I" begin
        g0 = analytic_gradient(p, p.Cost_ham)
        for c in (-5.0, 5.0, 20.0)
            gc = analytic_gradient(p, p.Cost_ham + c * I)
            @test norm(gc - g0) / norm(g0) < 1e-6
        end
    end

    @testset "agreement with finite differences" begin
        E(S) = SimpleEvolve.costfunction_ode(p.ψ0, p.eigvalues, p.signals(S), p.n_qubits, p.drives,
                                             p.eigvectors, p.T, p.Cost_ham;
                                             basis="qubitbasis", tol_ode=1e-12)[1]
        an = analytic_gradient(p, p.Cost_ham)
        fd = similar(an)
        h = 1e-6
        k = 0
        for part in (1.0, im), j in 1:p.n_qubits, n in 1:p.N+1
            k += 1
            D = zeros(ComplexF64, p.N + 1, p.n_qubits)
            D[n, j] = part
            fd[k] = (E(p.S .+ h .* D) - E(p.S .- h .* D)) / (2h)
        end
        rel = norm(an - fd) / norm(fd)
        scale = dot(an, fd) / dot(an, an)
        @info "finite-difference check" rel_error = rel best_scale = scale
        @test rel < 3e-2
        @test abs(scale - 1) < 1e-2
    end
end
