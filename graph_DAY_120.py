"""
Day 120 - Consolidation Phase 5: a linear RNN is a one-pole IIR filter.

h_t = a*h_{t-1} + x_t has impulse response a^t (memory) and a low-pass magnitude
response 1/|1 - a e^{-jw}|. BPTT multiplies the gradient by ~a every step, so the
same a^t curve is the gradient reaching t steps back. An LSTM forget gate near 1
is a pole near the unit circle: long memory, narrow low-pass.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

T = 60
t = np.arange(T)
poles = {"a=0.5 (forgets fast)": 0.5, "a=0.9 (vanilla RNN)": 0.9, "a=0.99 (forget gate ~ 1)": 0.99}

# --- Impulse response computed by actually running the recurrence ---
def run_rnn(a, x):
    h, out = 0.0, []
    for xt in x:
        h = a * h + xt       # linear RNN cell, no nonlinearity
        out.append(h)
    return np.array(out)

impulse = np.zeros(T)
impulse[0] = 1.0

w = np.linspace(1e-3, np.pi, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for label, a in poles.items():
    resp = run_rnn(a, impulse)
    assert np.allclose(resp, a ** t)   # recurrence == closed-form a^t
    axes[0].plot(t, resp, label=label)
    H = 1.0 / np.abs(1 - a * np.exp(-1j * w))
    axes[1].plot(w / np.pi, 20 * np.log10(H * (1 - a)), label=label)

axes[0].set_title("Memory of an input at t=0 (= gradient reaching t steps back)")
axes[0].set_xlabel("time steps")
axes[0].set_ylabel("a^t")
axes[0].legend()

axes[1].set_title("Frequency response: a longer memory is a narrower low-pass")
axes[1].set_xlabel("normalized frequency (x pi rad/sample)")
axes[1].set_ylabel("gain relative to DC (dB)")
axes[1].set_ylim(-40, 3)
axes[1].legend()

# half-life printout
for label, a in poles.items():
    print(f"{label}: half-life = {np.log(0.5) / np.log(a):.1f} steps")

plt.savefig("graph_DAY_120.png", dpi=120, bbox_inches="tight")
