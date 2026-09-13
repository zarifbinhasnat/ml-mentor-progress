"""
Day 106 -- Policy gradients / REINFORCE
Visualizes why a baseline slashes REINFORCE gradient-estimate variance.

Toy setup: one state, two actions, softmax policy pi_theta(a).
  a0 -> reward ~ N(1, 1)
  a1 -> reward ~ N(3, 1)
For each of many independent trials we sample one action, observe one
reward, and form the single-sample REINFORCE gradient estimate for the
theta component that pushes probability mass toward a1:

    grad_hat = (R - b) * (1{a=a1} - pi(a1))

with b = 0 (no baseline) and b = running mean reward (baseline).
Both estimators are unbiased (E[grad_hat] is identical either way,
since E[(b_const) * (1{a=a1} - pi(a1))] = 0 for any constant b), but
their variance is very different -- that's the plotted quantity.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

theta = np.array([0.0, 0.5])          # logits for [a0, a1] -- fixed policy for this illustration
probs = np.exp(theta) / np.exp(theta).sum()
pi_a1 = probs[1]

n_trials = 400
rewards_mean = np.array([1.0, 3.0])   # true mean reward per action
reward_std = 1.0

grad_no_baseline = np.zeros(n_trials)
grad_with_baseline = np.zeros(n_trials)

running_reward_sum = 0.0
for t in range(n_trials):
    a = 0 if np.random.rand() > pi_a1 else 1        # sample action ~ pi
    r = np.random.normal(rewards_mean[a], reward_std)  # observe stochastic reward

    indicator = 1.0 if a == 1 else 0.0
    score = indicator - pi_a1                        # d/dtheta_1 log pi(a) at this sample

    running_reward_sum += r
    baseline = running_reward_sum / (t + 1)           # running mean reward as baseline

    grad_no_baseline[t] = r * score
    grad_with_baseline[t] = (r - baseline) * score

var_no_baseline = np.var(grad_no_baseline)
var_with_baseline = np.var(grad_with_baseline)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(grad_no_baseline, bins=30, color="#d95f5f", alpha=0.85)
axes[0].set_title(f"No baseline\nVar = {var_no_baseline:.2f}")
axes[0].set_xlabel(r"$\hat{g} = R \cdot \mathrm{score}$")
axes[0].set_ylabel("count")

axes[1].hist(grad_with_baseline, bins=30, color="#4c8bf5", alpha=0.85)
axes[1].set_title(f"With baseline $b_t$ = running mean reward\nVar = {var_with_baseline:.2f}")
axes[1].set_xlabel(r"$\hat{g} = (R - b_t) \cdot \mathrm{score}$")
axes[1].set_ylabel("count")

fig.suptitle("REINFORCE gradient estimator: baseline shrinks variance, not bias")
plt.tight_layout()
plt.savefig("graph_DAY_106.png", dpi=120, bbox_inches="tight")

print(f"mean(no baseline)   = {grad_no_baseline.mean():.4f}")
print(f"mean(with baseline) = {grad_with_baseline.mean():.4f}")
print(f"var(no baseline)    = {var_no_baseline:.4f}")
print(f"var(with baseline)  = {var_with_baseline:.4f}")
print(f"variance reduction factor = {var_no_baseline / var_with_baseline:.2f}x")
