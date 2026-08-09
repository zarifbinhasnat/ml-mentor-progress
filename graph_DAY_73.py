"""Why LoRA works: weight-update deltas are effectively low-rank.

Synthesizes a plausible fine-tuning weight delta as a sum of a few dominant
rank-1 directions plus noise, then shows (a) its singular value spectrum
decays fast and (b) a rank-r truncation captures almost all of its energy
with r << min(d, k). This is the empirical hypothesis LoRA is built on.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

d, k = 64, 64
true_rank = 4

# Build a delta W as a sum of `true_rank` dominant rank-1 outer products
# (the "real" adaptation signal) plus small isotropic noise (numerical / optimization noise).
delta_w = np.zeros((d, k))
for i in range(true_rank):
    u = np.random.randn(d, 1)
    v = np.random.randn(1, k)
    scale = (true_rank - i) * 2.0  # earlier components carry more energy
    delta_w += scale * (u @ v)
delta_w += 0.3 * np.random.randn(d, k)  # noise floor

U, S, Vt = np.linalg.svd(delta_w, full_matrices=False)

# Reconstruction error vs. truncation rank r
ranks = np.arange(1, 21)
fro_total = np.linalg.norm(delta_w, ord="fro")
rel_errors = []
for r in ranks:
    approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
    rel_errors.append(np.linalg.norm(delta_w - approx, ord="fro") / fro_total)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].semilogy(np.arange(1, len(S) + 1), S, marker="o", markersize=3, color="#2b6cb0")
axes[0].axvline(true_rank, color="#e53e3e", linestyle="--", linewidth=1, label=f"true rank = {true_rank}")
axes[0].set_xlabel("singular value index")
axes[0].set_ylabel("singular value (log scale)")
axes[0].set_title("Spectrum of a synthetic fine-tuning ΔW")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(ranks, rel_errors, marker="o", markersize=4, color="#2f855a")
axes[1].axvline(true_rank, color="#e53e3e", linestyle="--", linewidth=1, label=f"true rank = {true_rank}")
axes[1].set_xlabel("LoRA rank r (truncation)")
axes[1].set_ylabel("relative Frobenius reconstruction error")
axes[1].set_title("Reconstruction error vs. rank r")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("LoRA's bet: task-adaptation deltas concentrate in a few directions", fontsize=11)
plt.savefig("graph_DAY_73.png", dpi=120, bbox_inches="tight")
