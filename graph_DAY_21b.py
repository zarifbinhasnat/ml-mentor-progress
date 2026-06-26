import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_layers = 15
n_units  = 32

def exploding_grad_norms(n_layers, n_units, spectral_norm):
    """Approximate gradient norms when weight spectral norm > 1."""
    norms = [spectral_norm ** k for k in range(n_layers, -1, -1)]
    return norms

# Normal network (spectral norm ≈ 0.9 → vanishing)
norms_ok  = exploding_grad_norms(n_layers, n_units, 0.9)
# Exploding (spectral norm ≈ 1.2)
norms_exp = exploding_grad_norms(n_layers, n_units, 1.2)

# Simulate gradient clipping effect
clip_threshold = 5.0
def clip_norm(v, max_norm):
    n = np.linalg.norm(v) if not np.isscalar(v) else abs(v)
    return min(n, max_norm) / max(n, 1e-9) * v if not np.isscalar(v) else min(abs(v), max_norm) * np.sign(v)

# Simulate training loss with and without clipping
steps = 80
np.random.seed(0)
base_grads = np.random.randn(steps) * 2.0
spike_steps = [20, 21, 22, 50, 51]
for s in spike_steps:
    base_grads[s] = 40.0   # simulated gradient spikes

theta_no_clip, theta_clip = 5.0, 5.0
loss_no_clip, loss_clip = [], []
for g in base_grads:
    theta_no_clip -= 0.01 * g
    loss_no_clip.append(0.5 * theta_no_clip**2)
    g_clipped = clip_norm(g, clip_threshold)
    theta_clip -= 0.01 * g_clipped
    loss_clip.append(0.5 * theta_clip**2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Day 21b — Exploding Gradients & Gradient Clipping", fontsize=13, fontweight="bold")

ax = axes[0]
layers = list(range(n_layers + 1))
ax.semilogy(layers, norms_ok,  "o-", color="#98c379", linewidth=2, markersize=5,
            label="Spectral norm < 1 (safe, vanishes)")
ax.semilogy(layers, norms_exp, "s-", color="#e06c75", linewidth=2, markersize=5,
            label="Spectral norm > 1 (explodes!)")
ax.set_xlabel("Layer (0=input, 15=output)")
ax.set_ylabel("‖gradient‖ (log scale)")
ax.set_title("Gradient explosion: ρ(W) > 1 amplifies per layer")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.axhline(1e4, color="darkred", linewidth=0.8, linestyle=":", label="NaN territory")

ax2 = axes[1]
ax2.plot(loss_no_clip, color="#e06c75", linewidth=2, label="No clipping — spikes destroy training")
ax2.plot(loss_clip,    color="#98c379", linewidth=2, label=f"Gradient clipping (max_norm={clip_threshold})")
for s in spike_steps[:2]:
    ax2.axvline(s, color="#e06c75", linewidth=0.8, linestyle=":", alpha=0.6)
ax2.set_xlabel("Training step"); ax2.set_ylabel("Loss")
ax2.set_title("Gradient clipping prevents spike-induced divergence")
ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3)
ax2.set_ylim(-1, min(200, max(loss_no_clip) * 1.1))

plt.tight_layout()
plt.savefig("graph_DAY_21b.png", dpi=120, bbox_inches="tight")
print("graph_DAY_21b.png saved")
