"""
Day 147 - Consolidation: the evaluation thread (calibration, benchmark noise, why metrics lie).
Left : reliability diagram of an overconfident classifier before/after temperature scaling.
Right: 95% CI half-width of accuracy vs test-set size -- a 2-point gap needs thousands of examples.
Synthetic, deterministic (seed 42), nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: binary classifier whose logits are 3x too large (overconfident)
n = 4000
z_true = np.random.randn(n) * 1.5                 # well-calibrated logits
y = (np.random.rand(n) < 1 / (1 + np.exp(-z_true))).astype(int)
z_over = 3.0 * z_true                             # overconfident model
def conf_acc(z, T):
    p = 1 / (1 + np.exp(-z / T))
    conf = np.maximum(p, 1 - p); pred = (p > 0.5).astype(int)
    return conf, (pred == y).astype(float)
def ece(conf, acc, bins=10):
    edges = np.linspace(0.5, 1.0, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): tot += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return tot
def curve(conf, acc, bins=10):
    edges = np.linspace(0.5, 1.0, bins + 1); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 10: xs.append(conf[m].mean()); ys.append(acc[m].mean())
    return xs, ys
c0, a0 = conf_acc(z_over, 1.0)
Ts = np.linspace(0.5, 6, 111)
nll = [-np.mean(y * np.log(1 / (1 + np.exp(-z_over / T)) + 1e-12) +
                (1 - y) * np.log(1 - 1 / (1 + np.exp(-z_over / T)) + 1e-12)) for T in Ts]
T_best = Ts[int(np.argmin(nll))]                  # temperature fit by minimizing NLL
c1, a1 = conf_acc(z_over, T_best)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot([0.5, 1], [0.5, 1], "k--", label="perfect calibration")
ax[0].plot(*curve(c0, a0), "o-", color="tab:red", label=f"raw (ECE={ece(c0, a0):.3f})")
ax[0].plot(*curve(c1, a1), "o-", color="tab:blue", label=f"T={T_best:.1f} (ECE={ece(c1, a1):.3f})")
ax[0].set_xlabel("mean confidence"); ax[0].set_ylabel("empirical accuracy")
ax[0].set_title("Overconfidence fixed by one scalar"); ax[0].legend(); ax[0].grid(alpha=0.3)

# ---- right: binomial 95% CI half-width at 80% accuracy; paired difference needs even more
ns = np.array([100, 300, 1000, 3000, 10000, 30000])
hw = 1.96 * np.sqrt(0.8 * 0.2 / ns) * 100
ax[1].loglog(ns, hw, "o-", color="tab:green", label="95% CI half-width of accuracy")
ax[1].axhline(2.0, color="tab:red", ls="--", label="a '2-point improvement'")
ax[1].set_xlabel("test-set size n"); ax[1].set_ylabel("percentage points")
ax[1].set_title("Small benchmarks cannot resolve small gains"); ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

print(f"raw ECE={ece(c0, a0):.3f}  accuracy={a0.mean():.3f}  mean conf={c0.mean():.3f}")
print(f"best T={T_best:.2f}  ECE after={ece(c1, a1):.3f}  mean conf={c1.mean():.3f}")
print("CI half-width (pts):", {int(k): round(float(v), 2) for k, v in zip(ns, hw)})
plt.savefig("graph_DAY_147.png", dpi=120, bbox_inches="tight")
