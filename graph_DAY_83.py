"""Day 83 — Diffusion: forward noising.

Visualizes the DDPM forward process q(x_t | x_0) on a synthetic 1D signal:
  1. The signal at several timesteps t as noise is injected.
  2. The alpha_bar_t (signal-retention) schedule and the resulting SNR decay.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- synthetic 1D "signal" (stand-in for a flattened toy image row) ---
n_points = 80
t_axis = np.linspace(0, 4 * np.pi, n_points)
x0 = np.sin(t_axis) + 0.3 * np.sin(3 * t_axis)  # a band-limited toy signal

# --- linear beta schedule (standard DDPM choice) ---
T = 200
beta_start, beta_end = 1e-4, 0.02
betas = np.linspace(beta_start, beta_end, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)  # alpha_bar_t = prod_{s<=t} alpha_s

# reparameterized forward sample: x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps
def forward_sample(x0, t_idx, alpha_bars, rng):
    eps = rng.standard_normal(x0.shape)
    ab = alpha_bars[t_idx]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps

rng = np.random.default_rng(42)
snapshot_ts = [0, 20, 60, 120, 199]

fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Panel 1: signal at increasing noise levels
ax = axes[0]
offset = 0
for t_idx in snapshot_ts:
    xt = forward_sample(x0, t_idx, alpha_bars, rng)
    ax.plot(t_axis, xt + offset, lw=1.5, label=f"t={t_idx}  (ᾱ_t={alpha_bars[t_idx]:.2f})")
    offset += 4
ax.set_title("Forward noising q(x_t | x_0): signal buried in Gaussian noise")
ax.set_xlabel("position (toy 1D signal)")
ax.set_yticks([])
ax.legend(loc="upper right", fontsize=8)

# Panel 2: alpha_bar_t schedule and SNR decay
ax2 = axes[1]
ax2.plot(np.arange(T), alpha_bars, color="tab:blue", label=r"$\bar{\alpha}_t$ (signal retained)")
ax2.set_xlabel("timestep t")
ax2.set_ylabel(r"$\bar{\alpha}_t$", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

snr = alpha_bars / (1 - alpha_bars)
ax3 = ax2.twinx()
ax3.plot(np.arange(T), 10 * np.log10(snr), color="tab:red", label="SNR (dB)")
ax3.set_ylabel("SNR (dB)", color="tab:red")
ax3.tick_params(axis="y", labelcolor="tab:red")
ax2.set_title("Variance-preserving schedule: signal power vs. noise power over t")

fig.tight_layout()
fig.savefig("graph_DAY_83.png", dpi=120, bbox_inches="tight")
