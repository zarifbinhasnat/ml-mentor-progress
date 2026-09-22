"""
Phase 11 consolidation — Connection 1: why does an actor-critic baseline
reduce policy-gradient variance without biasing it?

Two "states" produce very different typical returns (a stand-in for two very
different starting situations in an environment). REINFORCE's raw-return
weight R_i mixes both distributions together, so its spread is dominated by
*which state you happened to start in*, not by whether the action taken was
actually good. Subtracting a state-dependent baseline (exactly what a critic
V(s) learns to estimate) removes that between-state variance and leaves only
the local, action-relevant signal.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 60
mu0, mu1 = 2.0, 9.0   # true expected return of state 0 vs state 1
sigma = 1.5           # within-state noise (the part that's actually informative)

states = rng.integers(0, 2, size=n)                 # 0 or 1, ~half each
true_mean = np.where(states == 0, mu0, mu1)
returns = true_mean + rng.normal(0, sigma, size=n)   # R_i: the REINFORCE weight

baseline = true_mean                                  # V(s): a perfect critic's baseline
advantage = returns - baseline                         # A_i: the actor-critic weight

var_R, var_A = returns.var(), advantage.var()

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

colors = np.where(states == 0, "#4C72B0", "#DD8452")
axes[0].scatter(np.arange(n), returns, c=colors, s=28, alpha=0.85)
axes[0].axhline(mu0, color="#4C72B0", ls="--", lw=1, label=f"state 0 mean = {mu0}")
axes[0].axhline(mu1, color="#DD8452", ls="--", lw=1, label=f"state 1 mean = {mu1}")
axes[0].set_ylabel("REINFORCE weight $R_i$")
axes[0].set_title(f"Raw return (REINFORCE) — Var = {var_R:.2f}")
axes[0].legend(loc="upper left", fontsize=8)

axes[1].scatter(np.arange(n), advantage, c=colors, s=28, alpha=0.85)
axes[1].axhline(0.0, color="black", ls="--", lw=1)
axes[1].set_ylabel("Actor-critic weight $A_i = R_i - V(s_i)$")
axes[1].set_xlabel("sample $i$")
axes[1].set_title(f"Return minus state baseline (Actor-Critic) — Var = {var_A:.2f}")

fig.suptitle("Subtracting a state-dependent baseline removes between-state variance,\n"
             f"leaving a {var_R / var_A:.1f}x tighter gradient signal", y=1.02)
fig.tight_layout()
fig.savefig("graph_DAY_113.png", dpi=120, bbox_inches="tight")
print(f"Var(R) = {var_R:.3f}   Var(A) = {var_A:.3f}   ratio = {var_R/var_A:.2f}x")
