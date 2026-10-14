"""
Day 135 - Consolidation: the EMA thread (momentum -> Adam -> BatchNorm running stats -> DINO teacher -> DQN target net).

Left : EMA of a noisy 1-D "gradient" signal for beta = 0.5, 0.9, 0.99 (larger beta = smoother but laggier),
       plus the Adam bias-corrected estimate for beta = 0.99 which fixes the cold-start shrinkage toward 0.
Right: the EMA is a one-pole IIR low-pass filter. |H(w)| = (1-b)/sqrt(1 - 2 b cos w + b^2), drawn for the same betas;
       the dots are measured gains from filtering pure sinusoids.
All data synthetic, seed 42, 80 samples.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
T = 80
t = np.arange(T)
true = np.where(t < 40, 1.0, 2.0)                     # step in the underlying signal
x = true + 0.8 * np.random.randn(T)                   # noisy observations


def ema(sig, b, correct=False):
    m, out = 0.0, []
    for i, v in enumerate(sig, start=1):
        m = b * m + (1 - b) * v
        out.append(m / (1 - b ** i) if correct else m)
    return np.array(out)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
ax1.plot(t, x, ".", c="lightgray", label="noisy samples")
ax1.plot(t, true, "k--", lw=1, label="true signal")
for b, c in [(0.5, "tab:green"), (0.9, "tab:blue"), (0.99, "tab:red")]:
    ax1.plot(t, ema(x, b), c=c, lw=2, label=f"EMA b={b}")
ax1.plot(t, ema(x, 0.99, correct=True), c="tab:red", lw=1.5, ls=":", label="b=0.99 bias-corrected")
ax1.set_xlabel("step")
ax1.set_title("Bigger beta: smoother but laggier")
ax1.legend(fontsize=7)

w = np.linspace(0, np.pi, 400)
n = np.arange(2000)
for b, c in [(0.5, "tab:green"), (0.9, "tab:blue"), (0.99, "tab:red")]:
    H = (1 - b) / np.sqrt(1 - 2 * b * np.cos(w) + b ** 2)
    ax2.plot(w / (2 * np.pi), H, c=c, lw=2, label=f"b={b}")
    ws = np.array([0.05, 0.15, 0.3]) * 2 * np.pi
    meas = [np.abs(ema(np.sin(wi * n), b)[1000:]).max() for wi in ws]
    ax2.plot(ws / (2 * np.pi), meas, "o", c=c, ms=5)
    print(f"b={b}: time constant 1/(1-b) = {1/(1-b):.0f} steps, DC gain = 1.0, gain at Nyquist = {(1-b)/(1+b):.4f}")
ax2.set_xlabel("frequency (cycles/step)")
ax2.set_ylabel("|H(f)|")
ax2.set_title("EMA = one-pole IIR low-pass (dots: measured)")
ax2.legend(fontsize=8)

print("step-0 raw EMA (b=0.99) of first sample:", round(ema(x, 0.99)[0], 4), "vs x[0] =", round(x[0], 4),
      "| bias-corrected:", round(ema(x, 0.99, True)[0], 4))
plt.savefig("graph_DAY_135.png", dpi=120, bbox_inches="tight")
