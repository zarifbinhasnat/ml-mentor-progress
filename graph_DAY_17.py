import numpy as np
import matplotlib.pyplot as plt

T = 200    # total steps
Tw = 20    # warmup steps
eta0 = 1e-3
eta_min = 1e-6

steps = np.arange(T)

# Step decay: halve every 50 steps
def step_decay(t, gamma=0.5, k=50):
    return eta0 * gamma ** (t // k)

# Cosine annealing
def cosine(t):
    return eta_min + 0.5 * (eta0 - eta_min) * (1 + np.cos(np.pi * t / T))

# Warmup + Cosine
def warmup_cosine(t):
    if t < Tw:
        return eta0 * t / Tw
    return eta_min + 0.5 * (eta0 - eta_min) * (1 + np.cos(np.pi * (t - Tw) / (T - Tw)))

lr_step    = np.array([step_decay(t) for t in steps])
lr_cosine  = np.array([cosine(t)    for t in steps])
lr_warmup  = np.array([warmup_cosine(t) for t in steps])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Day 17 — LR Schedules: Step Decay, Cosine, Warmup+Cosine", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(steps, lr_step,   color="#e06c75", linewidth=2, label="Step decay (γ=0.5, every 50 steps)")
ax.plot(steps, lr_cosine, color="#61afef", linewidth=2, label="Cosine annealing")
ax.plot(steps, lr_warmup, color="#98c379", linewidth=2, label=f"Warmup ({Tw} steps) + Cosine")
ax.axhline(eta_min, color="k", linewidth=0.6, linestyle=":", alpha=0.5, label=f"η_min={eta_min:.0e}")
ax.set_xlabel("Training step"); ax.set_ylabel("Learning rate")
ax.set_title("LR schedules over 200 steps")
ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.semilogy(steps, lr_step,   color="#e06c75", linewidth=2, label="Step decay")
ax2.semilogy(steps, lr_cosine, color="#61afef", linewidth=2, label="Cosine")
ax2.semilogy(steps, lr_warmup, color="#98c379", linewidth=2, label="Warmup + Cosine")
ax2.axvspan(0, Tw, alpha=0.15, color="#98c379", label=f"Warmup zone (0–{Tw})")
# annotate the step decay discontinuities
for t_drop in [50, 100, 150]:
    ax2.axvline(t_drop, color="#e06c75", linewidth=0.8, linestyle=":", alpha=0.7)
ax2.set_xlabel("Training step"); ax2.set_ylabel("Learning rate (log)")
ax2.set_title("Log scale — step decay staircase vs smooth cosine")
ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_17.png", dpi=120, bbox_inches="tight")
print("graph_DAY_17.png saved")
