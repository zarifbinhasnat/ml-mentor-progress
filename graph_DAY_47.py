"""
Day 47 — Backprop Through Time (BPTT)
Shows two costs of BPTT: (1) memory grows linearly with sequence length
because every timestep's activations must be cached for the backward pass,
and truncated BPTT caps that; (2) the gradient signal reaching early
timesteps scales like a^T (same pole `a` as Day 46's forward IIR filter,
now driving the backward pass instead).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(46)

T_max = 60
timesteps = np.arange(1, T_max + 1)

fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# --- Top: cached-activation memory, full BPTT vs truncated BPTT (window k) ---
ax = axes[0]
full_bptt_memory = timesteps  # one cached activation set per timestep, grows without bound
k = 10
truncated_memory = np.minimum(timesteps, k)  # capped at window size k

ax.plot(timesteps, full_bptt_memory, color="#d62728", label="full BPTT: O(T) cached activations")
ax.plot(timesteps, truncated_memory, color="#1f77b4", label=f"truncated BPTT (k={k}): O(k), capped")
ax.axhline(k, color="#1f77b4", linestyle="--", linewidth=0.8)
ax.set_xlabel("sequence length T (timesteps processed so far)")
ax.set_ylabel("cached activation sets")
ax.set_title("Memory cost: full BPTT grows with T, truncated BPTT is capped at k")
ax.legend(fontsize=8)

# --- Bottom: gradient magnitude reaching early timesteps, scales like a^(T-t) ---
ax2 = axes[1]
a_values = [0.5, 0.9, 1.0, 1.05]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
steps_back = np.arange(0, 40)

for a, c in zip(a_values, colors):
    grad_ratio = np.abs(a) ** steps_back
    ax2.semilogy(steps_back, grad_ratio, color=c, marker=".", markersize=3, label=f"a={a}")

ax2.set_xlabel("steps backward from the loss (T - t)")
ax2.set_ylabel(r"$|\partial L/\partial h_t|$ relative scale (log)")
ax2.set_title("Gradient magnitude through BPTT: same a^(T-t) scaling as the forward impulse response")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_47.png", dpi=120, bbox_inches="tight")
