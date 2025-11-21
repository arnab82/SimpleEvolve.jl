using Random
using SimpleEvolve
using NPZ
using Plots
using LinearAlgebra
# Parameters
n_points = 100
T = 10.0  # Total duration in seconds

# Generate random complex amplitudes (real/imag parts between -1 and 1)
Random.seed!(42)  
real_parts = 2 .* rand(n_points) .- 1
imag_parts = 2 .* rand(n_points) .- 1
random_amplitudes = complex.(real_parts, imag_parts)

# Equal window durations (total_time/n_points per window)
window_durations = fill(T/n_points, n_points)

# Create windowed square wave signal
sw = WindowedSquareWave(
    100.0,            # Frequency (Hz)
    0.5,              # Duty cycle (50%)
    random_amplitudes,
    window_durations
)

# Generate samples
δt = T/(10*n_points)  # 10x oversampling
t = 0:δt:T
samples_initial = [SimpleEvolve.amplitude(sw, τ) for τ in t]

# Plot (real and imaginary components)
using Plots
plot(t, real.(samples_initial), label="Real", linewidth=1.5)
plot!(t, imag.(samples_initial), label="Imaginary", linewidth=1.5, linestyle=:dash)
xlabel!("Time (s)")
ylabel!("Amplitude")
title!("Random Complex Windowed Square Pulse")
# Create DigitizedSignal
complex_signal = DigitizedSignal(
    samples_initial,
    δt,
    sw.frequency  
)

# Parameters
n_windows = 5
T = 10.0
centers = LinRange(T/(n_windows+1), T-T/(n_windows+1), n_windows)
widths = fill(T/(6n_windows), n_windows)
amplitudes = complex.(2 .* rand(n_windows) .- 1, 2 .* rand(n_windows) .- 1)  
frequencies_1 = 100.0 * ones(n_windows)  # Constant frequency for all windows
frequencies_2 = 100.80 * ones(n_windows)  # Different constant frequency for second pulse
pulse_1 = SimpleEvolve.WindowedGaussianPulse(amplitudes, centers, widths, frequencies_1)
pulse_2 = SimpleEvolve.WindowedGaussianPulse(amplitudes, centers, widths, frequencies_2)
δt = 0.01
t = 0:δt:T
samples_1 = [SimpleEvolve.amplitude(pulse_1, τ) for τ in t]
samples_2 = [SimpleEvolve.amplitude(pulse_2, τ) for τ in t]
plot(t, real.(samples_1), label="Real pulse 1",color=:blue)
plot!(t, imag.(samples_1), label="Imag pulse 1", color=:orange)
plot!(t, real.(samples_2), label="Real Pulse 2", color=:red, linewidth=1.5)
plot!(t, imag.(samples_2), label="Imag Pulse 2", color=:green)
xlabel!("Time")
ylabel!("Amplitude")
title!("Windowed Gaussian Pulse with Frequency")


