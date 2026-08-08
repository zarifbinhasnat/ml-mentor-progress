import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# Synthetic RLHF "overoptimization" curve: as PPO pushes the policy further
# from the reference model (larger KL), the learned reward model's score
# (proxy reward) keeps climbing, but true human preference (gold reward)
# peaks and then falls off -- classic Goodhart's-law behavior in RLHF.
kl = np.linspace(0, 12, 60)  # KL(policy || reference), grows over PPO steps

# proxy reward model score: monotonically increasing, saturating slowly
proxy_reward = 3.2 * (1 - np.exp(-0.35 * kl)) + np.random.normal(0, 0.03, size=kl.shape)

# true/gold reward: rises with the proxy early on, then the policy starts
# exploiting reward-model blind spots and true quality decays
true_reward = 2.6 * (1 - np.exp(-0.5 * kl)) - 0.018 * kl**2 + np.random.normal(0, 0.03, size=kl.shape)

peak_idx = np.argmax(true_reward)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(kl, proxy_reward, color="#1f77b4", lw=2.2, label="proxy reward (reward model score)")
ax.plot(kl, true_reward, color="#d62728", lw=2.2, label="true reward (actual human preference)")
ax.axvline(kl[peak_idx], color="gray", ls="--", lw=1.2)
ax.annotate(
    "optimum: stop PPO here\n(early-stopping / KL penalty target)",
    xy=(kl[peak_idx], true_reward[peak_idx]),
    xytext=(kl[peak_idx] + 1.0, true_reward[peak_idx] - 0.55),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="gray"),
)
ax.fill_between(kl, true_reward, proxy_reward, where=(kl > kl[peak_idx]),
                 color="#d62728", alpha=0.08, label="reward hacking gap")

ax.set_xlabel(r"KL$(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$  (distance PPO has pushed the policy)")
ax.set_ylabel("reward")
ax.set_title("RLHF reward-model overoptimization: proxy vs. true reward")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.25)

plt.savefig("graph_DAY_72.png", dpi=120, bbox_inches="tight")
