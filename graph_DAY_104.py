"""
Day 104 — RL basics (MDP, reward, policy): discount factor as a filter's memory.

The discounted return G_t = sum_k gamma^k * r_{t+k+1} is a first-order IIR
filter applied backward over the reward stream (see SIGNAL LAB). This script
plots the filter's impulse response weight gamma^k vs. lag k for several
discount factors, and marks each curve's effective memory horizon
1 / (1 - gamma) — the "time constant" of the filter, i.e. how many steps
into the future the agent actually "listens to" before the weight decays
below 1/e.

Synthetic data only, no downloads.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

LAGS = np.arange(0, 60)  # k = 0..59 steps into the future
GAMMAS = [0.5, 0.9, 0.97]
COLORS = ["#3b82f6", "#f97316", "#10b981"]

fig, ax = plt.subplots(figsize=(7, 5))

for gamma, color in zip(GAMMAS, COLORS):
    weights = gamma ** LAGS  # impulse response h[k] = gamma^k of the return filter
    horizon = 1.0 / (1.0 - gamma)  # effective memory (steps until weight ~ 1/e)
    ax.plot(LAGS, weights, color=color, linewidth=2,
            label=f"$\\gamma$={gamma}  (horizon $\\approx${horizon:.0f} steps)")
    ax.axvline(horizon, color=color, linestyle="--", linewidth=1, alpha=0.6)

ax.axhline(1 / np.e, color="gray", linestyle=":", linewidth=1)
ax.text(LAGS[-1], 1 / np.e, " 1/e", va="bottom", ha="right", color="gray", fontsize=9)

ax.set_xlabel("steps into the future, k")
ax.set_ylabel("return weight $\\gamma^k$")
ax.set_title("Discount factor = pole of a first-order IIR filter on rewards")
ax.legend(loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig("graph_DAY_104.png", dpi=120, bbox_inches="tight")
for gamma in GAMMAS:
    print(f"gamma={gamma}: effective horizon = {1/(1-gamma):.1f} steps")
