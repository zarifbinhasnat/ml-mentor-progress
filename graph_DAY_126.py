"""
Day 126 - Consolidation Phase 11: three ideas that recur through RL, agents and long context.

Left  : PPO clipped surrogate vs probability ratio r (A>0 and A<0): the clip removes the incentive
        to move the policy far from the data-collecting policy.
Middle: REINFORCE gradient noise on a 3-armed bandit: subtracting a baseline keeps the mean
        gradient but shrinks its variance.
Right : RoPE wavelengths per frequency pair (d=64, base 10000) vs a 4096 context: low-frequency
        pairs never complete a cycle, high-frequency pairs alias like an under-sampled signal.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: PPO clip ----
eps = 0.2
r = np.linspace(0.4, 1.6, 200)
obj_pos = np.minimum(r * 1.0, np.clip(r, 1 - eps, 1 + eps) * 1.0)      # A = +1
obj_neg = np.minimum(r * -1.0, np.clip(r, 1 - eps, 1 + eps) * -1.0)    # A = -1

# ---- middle: REINFORCE variance with / without baseline ----
rewards = np.array([10.0, 11.0, 12.0])      # large common offset -> noisy raw gradient
logits = np.zeros(3)
pi = np.exp(logits) / np.exp(logits).sum()

def grad_sample(baseline):
    a = np.random.choice(3, p=pi)
    R = rewards[a] + np.random.randn()
    g = -pi.copy(); g[a] += 1.0              # grad of log pi(a) wrt logits
    return (R - baseline) * g

N = 2000
g_raw = np.array([grad_sample(0.0) for _ in range(N)])
g_base = np.array([grad_sample(float(pi @ rewards)) for _ in range(N)])
var_raw, var_base = g_raw.var(axis=0).sum(), g_base.var(axis=0).sum()
print("mean grad (no baseline):", g_raw.mean(axis=0).round(3))
print("mean grad (baseline)   :", g_base.mean(axis=0).round(3))
print(f"total variance: no baseline {var_raw:.2f}, with baseline {var_base:.2f}")

# ---- right: RoPE wavelengths ----
d, base, ctx = 64, 10000.0, 4096
i = np.arange(d // 2)
theta = base ** (-2 * i / d)
wavelen = 2 * np.pi / theta
print("pairs with wavelength > ctx:", int((wavelen > ctx).sum()), "of", d // 2)
print("pairs with wavelength < 4 tokens (aliasing risk):", int((wavelen < 4).sum()))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(r, obj_pos, label="A = +1", color="tab:green")
ax.plot(r, obj_neg, label="A = -1", color="tab:red")
ax.axvline(1 - eps, color="gray", ls="--", lw=0.8)
ax.axvline(1 + eps, color="gray", ls="--", lw=0.8)
ax.set_xlabel("ratio r = pi_new / pi_old")
ax.set_ylabel("clipped surrogate")
ax.set_title("PPO clip: flat where moving further earns nothing")
ax.legend(fontsize=8)

ax = axes[1]
bins = np.linspace(-6, 6, 40)
ax.hist(g_raw[:, 2], bins=bins, alpha=0.6, label=f"no baseline (var sum {var_raw:.1f})", color="tab:red")
ax.hist(g_base[:, 2], bins=bins, alpha=0.6, label=f"baseline (var sum {var_base:.1f})", color="tab:blue")
ax.set_xlabel("gradient component for best arm")
ax.set_ylabel("count")
ax.set_title("REINFORCE: baseline cuts variance")
ax.legend(fontsize=8)

ax = axes[2]
ax.semilogy(i, wavelen, "o-", ms=3, color="black")
ax.axhline(ctx, color="tab:orange", ls="--", label="context = 4096")
ax.axhline(2, color="tab:red", ls="--", label="Nyquist: 2 tokens")
ax.set_xlabel("RoPE frequency pair index")
ax.set_ylabel("wavelength (tokens)")
ax.set_title("RoPE as a filter bank")
ax.legend(fontsize=8)

plt.savefig("graph_DAY_126.png", dpi=120, bbox_inches="tight")
