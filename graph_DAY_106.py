"""
Day 106 -- Policy gradients / REINFORCE: variance reduction by averaging,
shown side-by-side in the RL domain and the frequency domain.

Left panel: a 2-armed logistic bandit. REINFORCE's gradient estimate is a
single score-function sample g = r(a) * d/dtheta[log pi_theta(a)]. We
Monte-Carlo the variance of the *batch-averaged* estimator (mean of N
i.i.d. single-trajectory samples) for N = 1..128 and overlay the
theoretical Var(g)/N line -- this is exactly why REINFORCE needs many
trajectories per update, and exactly the "more samples -> lower-variance
estimate" trade every gradient-based RL method makes.

Right panel: the *same* variance-vs-averaging trade, but in the frequency
domain. A single periodogram of a noisy sinusoid is a high-variance PSD
estimate; Welch's method splits the signal into K segments and averages
their periodograms, cutting variance the same way averaging K independent
REINFORCE samples does. Same law, two fields.

Synthetic data only, np.random.seed(42), no downloads.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------------
# Left panel: REINFORCE gradient-estimator variance vs batch size
# ---------------------------------------------------------------------------
THETA = 0.4           # fixed policy parameter (logit); pi(a=0) = sigmoid(theta)
REWARD = {0: 1.0, 1: 0.0}  # asymmetric rewards so the score-function estimator has real variance
N_TRIALS = 3000       # repeated batches per N, to estimate Var(batch mean)
BATCH_SIZES = np.array([1, 2, 4, 8, 16, 32, 64, 128])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_reinforce_grad(theta, n):
    """n i.i.d. single-trajectory REINFORCE gradient samples for a 2-armed bandit."""
    p0 = sigmoid(theta)
    actions = (np.random.rand(n) >= p0).astype(int)  # 0 with prob p0, else 1
    score = np.where(actions == 0, 1 - p0, -p0)       # d/dtheta log pi_theta(a)
    rewards = np.where(actions == 0, REWARD[0], REWARD[1])
    return rewards * score  # per-sample gradient estimate g_i


empirical_var = []
for N in BATCH_SIZES:
    batch_means = np.array([
        sample_reinforce_grad(THETA, N).mean() for _ in range(N_TRIALS)
    ])
    empirical_var.append(batch_means.var())
empirical_var = np.array(empirical_var)

single_sample_var = sample_reinforce_grad(THETA, 200_000).var()  # Var(g), N=1 reference
theory_var = single_sample_var / BATCH_SIZES  # Var(mean of N i.i.d.) = Var(g) / N

# ---------------------------------------------------------------------------
# Right panel: single periodogram vs Welch-averaged periodogram
# ---------------------------------------------------------------------------
FS = 1.0
N_SAMPLES = 4096
F0 = 0.08
t = np.arange(N_SAMPLES) / FS
x = np.sin(2 * np.pi * F0 * t) + 1.2 * np.random.randn(N_SAMPLES)


def periodogram(seg, fs=1.0):
    n = len(seg)
    X = np.fft.rfft(seg * np.hanning(n))
    Pxx = (1.0 / (fs * n)) * np.abs(X) ** 2
    Pxx[1:-1] *= 2.0  # one-sided spectrum
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, Pxx


def welch(x, fs=1.0, k=8):
    n = len(x)
    seg_len = n // k
    acc = None
    for i in range(k):
        seg = x[i * seg_len:(i + 1) * seg_len]
        freqs, Pxx = periodogram(seg, fs)
        acc = Pxx if acc is None else acc + Pxx
    return freqs, acc / k


freqs_single, Pxx_single = periodogram(x, FS)
freqs_welch, Pxx_welch = welch(x, FS, k=8)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

ax1.loglog(BATCH_SIZES, empirical_var, "o", color="#3b82f6", markersize=6,
           label="empirical Var(batch-mean grad)")
ax1.loglog(BATCH_SIZES, theory_var, "--", color="#ef4444", linewidth=1.5,
           label=r"theory: $\mathrm{Var}(g)/N$")
ax1.set_xlabel("N (trajectories per update)")
ax1.set_ylabel("variance of gradient estimate")
ax1.set_title("REINFORCE: variance $\\propto 1/N$")
ax1.legend(fontsize=8, loc="lower left")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.semilogy(freqs_single, Pxx_single, color="#94a3b8", linewidth=1.0,
             label="single periodogram (K=1)")
ax2.semilogy(freqs_welch, Pxx_welch, color="#10b981", linewidth=2.0,
             label="Welch-averaged (K=8)")
ax2.axvline(F0, color="#ef4444", linestyle="--", linewidth=1.0, label=f"true f0={F0}")
ax2.set_xlabel("frequency")
ax2.set_ylabel("PSD estimate")
ax2.set_title("Welch's method: same $1/K$ variance trade")
ax2.legend(fontsize=8, loc="upper right")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("graph_DAY_106.png", dpi=120, bbox_inches="tight")
print(f"single-sample Var(g) = {single_sample_var:.4f}")
print(f"empirical var at N=128: {empirical_var[-1]:.6f}, theory: {theory_var[-1]:.6f}")
