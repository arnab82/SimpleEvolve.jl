# SimpleEvolve

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nmayhall-vt.github.io/SimpleEvolve.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nmayhall-vt.github.io/SimpleEvolve.jl/dev/)
[![Build Status](https://github.com/nmayhall-vt/SimpleEvolve.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nmayhall-vt/SimpleEvolve.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nmayhall-vt/SimpleEvolve.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/SimpleEvolve.jl)

SimpleEvolve.jl is a small Julia package for pulse-level quantum evolution and
control optimization. It provides utilities for building transmon device
Hamiltonians, representing time-dependent multi-channel control signals,
evolving quantum states, evaluating variational pulse objectives, and computing
analytic pulse gradients.

The main workflow is:

1. Build a device Hamiltonian, usually a coupled transmon model.
2. Define one complex drive envelope per qubit/channel.
3. Evolve an initial state under the driven Hamiltonian.
4. Evaluate an energy or fidelity-style cost.
5. Optimize pulse parameters using analytic gradients.

## Installation

SimpleEvolve.jl targets Julia 1.11. From the repository root, start a Julia
1.11 session with the project active:

```bash
julia +1.11 --project=.
```

Then instantiate the dependencies:

```julia
using Pkg
Pkg.instantiate()
```

For development from another Julia 1.11 environment:

```julia
using Pkg
Pkg.develop(path="/path/to/SimpleEvolve.jl")
using SimpleEvolve
```

Run the tests with:

```bash
julia +1.11 --project=. -e 'using Pkg; Pkg.test()'
```

## Main Components

- `Transmon`, `QubitCoupling`, and `static_hamiltonian` construct coupled
  transmon device Hamiltonians.
- `a_q`, `a_fullspace`, and `projector` build local/full-space annihilation
  operators and Hilbert-space projectors.
- `DigitizedSignal`, `WindowedSquareWave`, `WindowedGaussianPulse`, and
  `MultiChannelSignal` represent drive envelopes.
- `evolve_ODE`, `evolve_direct_exponentiation`, and `trotter_evolve` propagate
  states under time-dependent drives.
- `costfunction_ode`, `costfunction_ode_with_penalty`, and related routines
  evaluate final-time objectives.
- `gradientsignal_ODE` computes analytic gradients of the cost with respect to
  the real and imaginary pulse samples.
- `validate_and_expand` and the signal reconstruction helpers interpolate or
  upsample reduced control/gradient signals.

## Minimal Example

The following sketch builds a two-qubit transmon device, creates one digitized
drive per qubit, and evolves a computational-basis state.

```julia
using SimpleEvolve
using LinearAlgebra

n_qubits = 2
n_levels = 2
T = 20.0
n_samples = 100
dt = T / n_samples

device = Transmon(
    2π .* [4.82, 4.84],
    2π .* [0.30, 0.30],
    Dict(QubitCoupling(1, 2) => 2π * 0.02),
)

H_static = static_hamiltonian(device, n_levels)
eigvalues, eigvectors = eigen(Hermitian(H_static))

drives = a_fullspace(n_qubits, n_levels)
for q in 1:n_qubits
    drives[q] = eigvectors' * drives[q] * eigvectors
end

times = range(0, T; length=n_samples + 1)
carrier_freqs = 2π .* [4.82, 4.84]

signals = MultiChannelSignal([
    DigitizedSignal(ComplexF64.(2π * 1e-4 .* sin.(2π .* times ./ T)), dt, carrier_freqs[q])
    for q in 1:n_qubits
])

ψ0 = zeros(ComplexF64, n_levels^n_qubits)
ψ0[1] = 1.0

ψT = evolve_ODE(
    ψ0, T, signals, n_qubits, drives, eigvalues, eigvectors;
    basis="qubitbasis",
    tol_ode=1e-8,
)
```

## Pulse Optimization Workflow

The notebook
`example/slepian/slepian_LiH_VERIFICATION.ipynb` gives a complete worked
example for LiH. It uses a bandlimited Slepian/DPSS basis to optimize pulse
envelopes:

```julia
using DSP

fs = 2.0          # samples/ns
B = 0.4           # GHz bandwidth
N = Int(round(T * fs))
W = B / fs
K = Int(round(2 * N * W - 3))

dpss_basis = dpss(N + 1, N * W, K)
Φ = hcat([dpss_basis[:, k] / norm(dpss_basis[:, k]) for k in 1:K]...)
```

Each complex pulse is parameterized as

```julia
Ω_real = Φ * coeffs_real
Ω_imag = Φ * coeffs_imag
Ω = Ω_real .+ im .* Ω_imag
```

The optimizer sees the Slepian coefficients, while `SimpleEvolve` evolves the
corresponding digitized pulses. A typical cost wrapper is:

```julia
function costfunction_coeffs(coeffs)
    samples_complex = SimpleEvolve.coeffs_to_samples_matrix(coeffs, Φ, K, n_qubits)
    signals = MultiChannelSignal([
        DigitizedSignal(samples_complex[:, q], dt, carrier_freqs[q])
        for q in 1:n_qubits
    ])

    energy, ψT = costfunction_ode_with_penalty(
        ψ_initial, eigvalues, signals, n_qubits, drives, eigvectors, T, Cost_ham;
        basis="qubitbasis",
        tol_ode=1e-8,
    )

    return energy
end
```

The analytic gradient is first computed with respect to time samples and then
projected back to Slepian coefficients:

```julia
∂Ω_real, ∂Ω_imag, ψT, σT = gradientsignal_ODE(
    ψ_initial, T, signals, n_qubits, drives, eigvalues, eigvectors,
    Cost_ham, n_samples;
    basis="qubitbasis",
    tol_ode=1e-8,
)

grad_real = Φ' * ∂Ω_real
grad_imag = Φ' * ∂Ω_imag
grad = vcat(vec(grad_real), vec(grad_imag))
```

For optimization, pass the scalar cost and an in-place gradient function to
`Optim.jl`. The Slepian example uses limited-memory BFGS with a More-Thuente
line search; for smaller parameter vectors, `Optim.BFGS` can be used instead.

```julia
using Optim
using LineSearches

tol_ode = 1e-6

function gradient_coeffs!(G, coeffs)
    samples_complex = SimpleEvolve.coeffs_to_samples_matrix(coeffs, Φ, K, n_qubits)
    signals = MultiChannelSignal([
        DigitizedSignal(samples_complex[:, q], dt, carrier_freqs[q])
        for q in 1:n_qubits
    ])

    ∂Ω_real, ∂Ω_imag, ψT, σT = gradientsignal_ODE(
        ψ_initial, T, signals, n_qubits, drives, eigvalues, eigvectors,
        Cost_ham, n_samples;
        basis="qubitbasis",
        tol_ode=tol_ode,
    )

    grad_real = Φ' * ∂Ω_real
    grad_imag = Φ' * ∂Ω_imag
    G .= vcat(vec(grad_real), vec(grad_imag))
    return G
end

function safe_cost(coeffs)
    value = costfunction_coeffs(coeffs)
    return isfinite(value) ? value : Inf
end

function safe_gradient!(G, coeffs)
    gradient_coeffs!(G, coeffs)
    return G
end

linesearch = LineSearches.MoreThuente()
optimizer = Optim.LBFGS(linesearch=linesearch)
# For smaller coefficient vectors, use:
# optimizer = Optim.BFGS(linesearch=linesearch)

options = Optim.Options(
    show_trace = true,
    show_every = 1,
    f_reltol = 1e-15,
    g_tol = 1e-6,
    iterations = 1000,
)

optimization = Optim.optimize(
    safe_cost,
    safe_gradient!,
    coeffs_initial,
    optimizer,
    options,
)
coeffs_final = Optim.minimizer(optimization)
```

The LiH verification notebook also warm-restarts the optimizer while tightening
the ODE tolerance, for example `tol_ode = 1e-6`, then `1e-8`, then `1e-10`.

This gives an efficient optimization loop: one forward and one backward ODE
solve compute the full pulse gradient, independent of the number of pulse
parameters.

## Examples

- `example/slepian/slepian_LiH_VERIFICATION.ipynb`: detailed LiH Slepian-basis
  optimization tutorial.
- `example/lih15/`: LiH pulse optimization examples without the Slepian basis.
  For Slepian-basis pulse optimization, follow
  `example/slepian/slepian_LiH_VERIFICATION.ipynb`.
- `example/H2O_stretching/`, `example/N2_stretching/`, and
  `example/F2_stretching/`: molecular Hamiltonian examples across geometries.
- `test/`: regression tests and small optimization checks.

## Notes

- Most routines assume Julia's 1-based indexing and dense state vectors.
- Use `basis="qubitbasis"` when the input state and cost Hamiltonian are in the
  computational basis. Use the default `basis="eigenbasis"` when they are
  already expressed in the static-Hamiltonian eigenbasis.
- Drive operators are usually rotated into the eigenbasis of `H_static` before
  calling the evolution and gradient routines.
