"""
Day 46 — RNN Cell
Shows the RNN's recurrence as an IIR (infinite impulse response) filter:
unlike a CNN's fixed-width FIR kernel, a linear RNN's impulse response
h_t = a*h_{t-1} + x_t keeps ringing forever, with |a| controlling whether
that ring decays, sustains, or blows up.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(45)

T = 40
impulse = np.zeros(T)
impulse[0] = 1.0

a_values = [0.5, 0.9, 1.0, 1.05]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# --- Top: impulse response of the linear recurrence h_t = a*h_{t-1} + x_t ---
ax = axes[0]
for a, c in zip(a_values, colors):
    h = np.zeros(T)
    for t in range(T):
        prev = h[t - 1] if t > 0 else 0.0
        h[t] = a * prev + impulse[t]
    label = f"a={a}" + ("  (decays)" if a < 1 else " (sustains)" if a == 1 else " (explodes)")
    ax.plot(h, color=c, marker=".", markersize=3, label=label)

ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("timestep t")
ax.set_ylabel("h_t")
ax.set_title("RNN recurrence = IIR filter: impulse response never truly ends")
ax.legend(fontsize=8)

# --- Bottom: an FIR (CNN-style fixed window) impulse response for contrast ---
ax2 = axes[1]
kernel = np.array([1, 2, 3, 2, 1], dtype=float)
kernel /= kernel.sum()
fir_response = np.convolve(impulse, kernel)[:T]
ax2.stem(fir_response, linefmt="#9467bd", markerfmt="o", basefmt=" ")
ax2.set_xlabel("timestep t")
ax2.set_ylabel("y_t")
ax2.set_title("CNN kernel = FIR filter: impulse response is exactly kernel-width, then zero")
ax2.set_xlim(-1, T)

plt.tight_layout()
plt.savefig("graph_DAY_46.png", dpi=120, bbox_inches="tight")
