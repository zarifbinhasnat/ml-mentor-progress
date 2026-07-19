"""
Day 50 — GRU
Shows the two structural differences from LSTM: (1) GRU's update gate couples
"how much to forget" and "how much to add" into a single convex combination
(f + i = 1 always), a strict subset of the LSTM's independently-free (f, i)
plane; (2) GRU needs 3 gate weight sets instead of 4, a real parameter saving
that grows with hidden size.
"""
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# --- Left: LSTM's free (f, i) plane vs GRU's constrained line ---
ax = axes[0]
ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#1f77b4", alpha=0.15,
                            edgecolor="#1f77b4", label="LSTM: (f, i) any point in [0,1]x[0,1]"))
z = np.linspace(0, 1, 100)
ax.plot(1 - z, z, color="#d62728", linewidth=2.5, label="GRU: (1-z, z), forced onto this line")

corners = {
    "(0,0): forget all, add nothing\n(cell resets to 0)": (0, 0),
    "(1,0): keep all, add nothing\n(pure memory)": (1, 0),
    "(0,1): forget all, add all\n(full replace)": (0, 1),
    "(1,1): keep all AND add all\n(grows without bound)": (1, 1),
}
for label, (fx, fy) in corners.items():
    ax.plot(fx, fy, "ko", markersize=4)

ax.set_xlabel("f (forget gate / keep-old weight)")
ax.set_ylabel("i (input gate / add-new weight)")
ax.set_title("LSTM's independent gates vs GRU's coupled update gate")
ax.legend(fontsize=7, loc="lower left")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_aspect("equal")

# --- Right: parameter count, LSTM (4 gates) vs GRU (3 gates), vs hidden size ---
ax2 = axes[1]
input_size = 32
hidden_sizes = np.arange(8, 257, 8)

def gate_params(n_gates, d_in, d_h):
    # each gate: W_x (d_h x d_in) + W_h (d_h x d_h) + bias (d_h)
    return n_gates * (d_h * d_in + d_h * d_h + d_h)

lstm_params = gate_params(4, input_size, hidden_sizes)
gru_params = gate_params(3, input_size, hidden_sizes)

ax2.plot(hidden_sizes, lstm_params, color="#1f77b4", label="LSTM (4 gate sets)")
ax2.plot(hidden_sizes, gru_params, color="#d62728", label="GRU (3 gate sets)")
ax2.fill_between(hidden_sizes, gru_params, lstm_params, alpha=0.15, color="gray")
ax2.set_xlabel("hidden size")
ax2.set_ylabel("recurrent layer parameter count")
ax2.set_title(f"Parameter cost (input_size={input_size}): GRU is ~25% smaller")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_50.png", dpi=120, bbox_inches="tight")
