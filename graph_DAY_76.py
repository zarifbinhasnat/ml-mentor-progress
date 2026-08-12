"""
Day 76 — Mixture of Experts: router load balancing.

Simulates routing 512 tokens to 8 experts under top-2 gating, comparing
a router trained WITHOUT a load-balancing auxiliary loss (collapses onto
a few "winner" experts) against one trained WITH it (near-uniform load).
Synthetic data only — no downloads, no internet.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_tokens = 512
n_experts = 8
top_k = 2

# --- Router WITHOUT load-balancing loss: a few experts dominate the logits,
# so softmax repeatedly favors the same small subset (collapse). ---
collapse_bias = np.array([3.0, 2.6, 0.1, 0.0, -0.2, -0.4, -0.6, -0.8])
logits_collapsed = collapse_bias[None, :] + 0.3 * np.random.randn(n_tokens, n_experts)

# --- Router WITH load-balancing loss: bias pushed toward uniform, so
# per-token noise dominates and traffic spreads across all experts. ---
logits_balanced = 0.1 * np.random.randn(n_tokens, n_experts)


def route_counts(logits, k):
    counts = np.zeros(logits.shape[1], dtype=int)
    for row in logits:
        top = np.argpartition(row, -k)[-k:]  # top-k expert indices for this token
        counts[top] += 1
    return counts


counts_collapsed = route_counts(logits_collapsed, top_k)
counts_balanced = route_counts(logits_balanced, top_k)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
experts = np.arange(n_experts)

axes[0].bar(experts, counts_collapsed, color="#c0392b")
axes[0].set_title("No aux loss: expert collapse")
axes[0].set_xlabel("Expert ID")
axes[0].set_ylabel("Tokens routed")
axes[0].axhline(n_tokens * top_k / n_experts, color="black", linestyle="--", linewidth=1, label="ideal uniform")
axes[0].legend(fontsize=8)

axes[1].bar(experts, counts_balanced, color="#27ae60")
axes[1].set_title("With aux loss: balanced load")
axes[1].set_xlabel("Expert ID")
axes[1].axhline(n_tokens * top_k / n_experts, color="black", linestyle="--", linewidth=1)

fig.suptitle("Mixture of Experts: top-2 routing over 512 tokens, 8 experts")
plt.tight_layout()
plt.savefig("graph_DAY_76.png", dpi=120, bbox_inches="tight")
