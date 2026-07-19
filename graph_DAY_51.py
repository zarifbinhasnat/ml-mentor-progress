"""
Day 51 — Seq2seq Encoder-Decoder
Shows "exposure bias": training with teacher forcing (decoder always sees the
TRUE previous token) vs inference (decoder must feed back its OWN previous
prediction) are different feedback loops -- closed-loop (error-corrected every
step) vs open-loop (errors compound) in control-theory terms. Same per-step
noise, wildly different accumulated error over a decode sequence.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(50)

T = 50
n_trials = 30
per_step_noise_std = 0.15

fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# true reference sequence the decoder is trying to reproduce
true_seq = np.cumsum(np.random.randn(T) * 0.3)

# --- Top: example trajectories, teacher forcing vs free-running ---
ax = axes[0]
ax.plot(true_seq, color="black", linewidth=2, label="ground truth sequence")

# teacher forcing: prediction at t is always true_seq[t-1] + noise (error never compounds)
tf_pred = np.zeros(T)
tf_pred[0] = true_seq[0]
for t in range(1, T):
    tf_pred[t] = true_seq[t - 1] + np.random.randn() * per_step_noise_std
ax.plot(tf_pred, color="#1f77b4", linestyle="--", label="teacher forcing (fed TRUE prev token)")

# free-running: prediction at t is based on the model's OWN previous prediction
fr_pred = np.zeros(T)
fr_pred[0] = true_seq[0]
for t in range(1, T):
    fr_pred[t] = fr_pred[t - 1] + np.random.randn() * per_step_noise_std
ax.plot(fr_pred, color="#d62728", label="free-running (fed OWN prev prediction)")

ax.set_xlabel("decode step")
ax.set_ylabel("value")
ax.set_title("Same per-step noise, different feedback loop: open-loop drifts, closed-loop doesn't")
ax.legend(fontsize=8)

# --- Bottom: accumulated |error| averaged over many trials ---
ax2 = axes[1]
tf_errors = np.zeros((n_trials, T))
fr_errors = np.zeros((n_trials, T))

for trial in range(n_trials):
    true_t = np.cumsum(np.random.randn(T) * 0.3)

    tf = np.zeros(T)
    tf[0] = true_t[0]
    for t in range(1, T):
        tf[t] = true_t[t - 1] + np.random.randn() * per_step_noise_std
    tf_errors[trial] = np.abs(tf - true_t)

    fr = np.zeros(T)
    fr[0] = true_t[0]
    for t in range(1, T):
        fr[t] = fr[t - 1] + np.random.randn() * per_step_noise_std
    fr_errors[trial] = np.abs(fr - true_t)

ax2.plot(tf_errors.mean(axis=0), color="#1f77b4", label="teacher forcing: bounded error")
ax2.plot(fr_errors.mean(axis=0), color="#d62728", label="free-running: error compounds (random walk)")
ax2.set_xlabel("decode step")
ax2.set_ylabel("mean |prediction error| over 30 trials")
ax2.set_title("Exposure bias: train/test feedback-loop mismatch, not a training bug")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_51.png", dpi=120, bbox_inches="tight")
