"""
Day 53 -- Attention Intuition (soft lookup)
Two panels: (1) a toy "dictionary lookup" -- one query dot-producted against 8
keys, softmax'd into attention weights, contrasted with what a HARD lookup
(argmax-only, like a Python dict) would do; (2) the resulting soft-lookup
output is a weighted blend of ALL values, not just the winner, and that blend
sits strictly between the value vectors it draws from.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Toy "soft dictionary": 8 keys/values, 1 query ---
n_keys = 8
d = 4
keys = np.random.randn(n_keys, d)
values = np.random.randn(n_keys, 1) * 3  # scalar "value" per key, for easy plotting
query = keys[2] + 0.3 * np.random.randn(d)  # query is close to key 2, but not identical

scores = keys @ query  # raw dot-product similarity, pre-softmax
weights = np.exp(scores - scores.max())
weights /= weights.sum()  # softmax -> attention weights, sum to 1

hard_choice = np.argmax(scores)  # what a hash-map / argmax lookup would return
soft_output = weights @ values.flatten()  # weighted blend -- attention's actual output
hard_output = values[hard_choice, 0]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- Left: attention weights vs a hard one-hot lookup ---
ax = axes[0]
x = np.arange(n_keys)
onehot = np.zeros(n_keys)
onehot[hard_choice] = 1.0
width = 0.38
ax.bar(x - width / 2, onehot, width, color="#7f7f7f", label="hard lookup (argmax, like dict[key])")
ax.bar(x + width / 2, weights, width, color="#1f77b4", label="soft lookup (softmax attention weights)")
ax.set_xticks(x)
ax.set_xticklabels([f"key {i}" for i in range(n_keys)])
ax.set_ylabel("weight assigned")
ax.set_title("Hard lookup picks ONE key; soft lookup blends ALL of them")
ax.legend(fontsize=8)

# --- Right: resulting output value -- soft blend sits between the values it drew from ---
ax2 = axes[1]
ax2.scatter(x, values.flatten(), color="#2ca02c", zorder=3, label="value$_i$ stored at each key")
ax2.axhline(hard_output, color="#7f7f7f", linestyle="--", label=f"hard lookup output = value[{hard_choice}]")
ax2.axhline(soft_output, color="#1f77b4", linestyle="-", label="soft lookup output = $\\sum_i w_i \\cdot$ value$_i$")
for xi, wi in zip(x, weights):
    ax2.plot([xi, xi], [0, wi * 0], alpha=0)  # keep axes consistent, no-op draw
ax2.set_xticks(x)
ax2.set_xticklabels([f"key {i}" for i in range(n_keys)])
ax2.set_ylabel("value")
ax2.set_title("Soft output is a weighted average -- it can use partial evidence\nfrom every key, not just the single closest one")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_53.png", dpi=120, bbox_inches="tight")
