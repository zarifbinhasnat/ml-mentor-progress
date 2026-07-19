"""
Day 49 — LSTM Gates & Cell State
Shows why the LSTM's cell-state path survives long sequences where the vanilla
RNN's hidden-state path doesn't: the cell state's backward path is an additive
product of forget gates ONLY (no tanh-derivative multiplied in every step),
and a forget gate that saturates near 1 keeps that product close to 1 for a
long time.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(48)

T = 60
steps = np.arange(T)

fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# --- Top: LSTM cell-state gradient path (product of forget gates) vs vanilla RNN ---
ax = axes[0]

# LSTM: cell state gradient is just the running product of forget gate values
forget_gate_scenarios = {
    "LSTM, forget gate ~0.98 (learned to remember)": 0.98,
    "LSTM, forget gate ~0.9": 0.90,
    "LSTM, forget gate ~0.5 (learned to forget fast)": 0.50,
}
colors_lstm = ["#2ca02c", "#1f77b4", "#d62728"]
for (label, f), c in zip(forget_gate_scenarios.items(), colors_lstm):
    gate_product = f ** steps
    ax.semilogy(steps, gate_product, color=c, label=label)

# Vanilla RNN: reuse Day 48's compounding factor (tanh saturation x spectral radius ~0.9)
vanilla_factor = 0.9 * (1 - np.tanh(1.5) ** 2)  # ~0.9 * 0.19 = 0.17 per step
vanilla_decay = vanilla_factor ** steps
ax.semilogy(steps, vanilla_decay, color="black", linestyle="--",
            label="vanilla RNN (spectral radius 0.9 x tanh saturation)")

ax.set_xlabel("steps backward through time")
ax.set_ylabel("gradient scale along this path (log)")
ax.set_title("LSTM cell-state gradient path: no tanh' multiplied in, just forget gates")
ax.legend(fontsize=7)

# --- Bottom: sigmoid, showing how a gate saturates toward a clean 0/1 switch ---
ax2 = axes[1]
x = np.linspace(-8, 8, 400)
sigmoid = 1 / (1 + np.exp(-x))
ax2.plot(x, sigmoid, color="#9467bd")
ax2.axhline(1.0, color="black", linewidth=0.5)
ax2.axhline(0.0, color="black", linewidth=0.5)
ax2.set_xlabel("gate pre-activation")
ax2.set_ylabel(r"$\sigma(x)$ = gate value")
ax2.set_title("Sigmoid gates saturate near 0 (fully closed) or 1 (fully open) -- a learned switch")

plt.tight_layout()
plt.savefig("graph_DAY_49.png", dpi=120, bbox_inches="tight")
