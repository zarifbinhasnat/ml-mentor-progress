"""
DINO teacher = EMA(student) is literally a first-order IIR low-pass filter.
This script simulates a noisy 'student' training signal and shows how two
different EMA momenta (lambda) act as low-pass filters with different
cutoff frequencies / time constants on that same signal.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 100
steps = np.arange(n)

# Synthetic "student" trajectory: a slow rising trend (real learning signal,
# low-frequency) plus step-to-step noise (SGD jitter, high-frequency).
true_trend = 5 * (1 - np.exp(-steps / 30))
noise = np.random.normal(0, 0.6, n)
student = true_trend + noise


def ema(signal, lam):
    out = np.zeros_like(signal, dtype=float)
    out[0] = signal[0]
    for i in range(1, len(signal)):
        out[i] = lam * out[i - 1] + (1 - lam) * signal[i]
    return out


teacher_fast = ema(student, 0.9)     # low momentum -> wide passband, noisy, low lag
teacher_dino = ema(student, 0.996)   # DINO-typical momentum -> narrow passband, smooth, laggy

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, student, color="gray", alpha=0.5, linewidth=1, label="Student (raw, per-step)")
ax.plot(steps, teacher_fast, color="tab:orange", linewidth=1.8,
        label=r"Teacher EMA, $\lambda=0.9$  ($\tau\approx10$ steps)")
ax.plot(steps, teacher_dino, color="tab:blue", linewidth=2.2,
        label=r"Teacher EMA, $\lambda=0.996$  ($\tau\approx250$ steps)")
ax.plot(steps, true_trend, color="black", linestyle="--", linewidth=1,
        label="True underlying trend (unobservable)")

ax.set_xlabel("Training step")
ax.set_ylabel("Representative output statistic")
ax.set_title("Teacher = EMA(Student): a time-varying low-pass filter")
ax.legend(loc="lower right", fontsize=9)
fig.savefig("graph_DAY_93.png", dpi=120, bbox_inches="tight")
