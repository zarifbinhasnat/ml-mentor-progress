"""
Graph for Day 114 — Concept 50: Bidirectional RNNs.

DSP framing: a causal (forward-only) one-pole IIR smoother behaves exactly
like a unidirectional RNN's hidden state -- it can only react to a step AFTER
it happens, so its response lags. Running the same filter backward is
"anticausal" and reacts BEFORE the step (it already knows the future).
Combining the forward and backward passes -- exactly what concatenating
h_t^forward and h_t^backward does in a BiRNN -- cancels that lag and centers
the response exactly on the true transition, at the cost of needing the
whole sequence up front (no longer real-time / causal).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- synthetic noisy step signal: flat at 0, steps up to 1 at n=30 ---
n = 60
true_signal = np.zeros(n)
true_signal[30:] = 1.0
noise = np.random.normal(0, 0.08, size=n)
x = true_signal + noise

alpha = 0.25  # smoothing factor for the one-pole IIR filter


def causal_filter(signal):
    """Forward-only one-pole IIR: y[t] = alpha*x[t] + (1-alpha)*y[t-1]."""
    y = np.zeros_like(signal)
    y[0] = signal[0]
    for t in range(1, len(signal)):
        y[t] = alpha * signal[t] + (1 - alpha) * y[t - 1]
    return y


# forward pass: only ever sees x[0..t], mirrors h_t^forward
forward_pass = causal_filter(x)

# backward pass: run the identical filter on the reversed signal, then
# flip the result back -- mirrors h_t^backward, which is built by scanning
# t = T..1
backward_pass = causal_filter(x[::-1])[::-1]

# combined: mirrors concatenating [h_t^forward ; h_t^backward] -- here we
# average the two scalar traces to visualize what "using both directions"
# buys you at every single timestep
combined = 0.5 * (forward_pass + backward_pass)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, color="#999999", lw=1, alpha=0.6, label="noisy input")
ax.plot(true_signal, color="black", lw=1, ls="--", label="true step")
ax.plot(forward_pass, color="#d62728", lw=2, label="forward-only (causal) — lags the step")
ax.plot(backward_pass, color="#1f77b4", lw=2, label="backward-only (anticausal) — leads the step")
ax.plot(combined, color="#2ca02c", lw=2.5, label="forward+backward combined — centered, no lag")

ax.axvline(30, color="black", lw=0.8, alpha=0.4)
ax.set_xlabel("timestep")
ax.set_ylabel("filtered value")
ax.set_title("Causal vs. bidirectional smoothing = unidirectional vs. bidirectional RNN")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("graph_DAY_114.png", dpi=120, bbox_inches="tight")
