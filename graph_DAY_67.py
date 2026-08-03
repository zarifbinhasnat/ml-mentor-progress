"""
Day 67 - BERT & Masked LM
Visualizes the single structural fact that separates BERT from GPT (Day 68
preview): the attention mask. BERT's masked-LM objective only works because
every token -- including the [MASK] position -- is allowed to attend in both
directions. A causal (GPT-style) mask would make predicting a masked middle
token from its right-context literally impossible (future positions are -inf).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

tokens = ["[CLS]", "the", "cat", "[MASK]", "on", "the", "mat", "[SEP]"]
n = len(tokens)
mask_pos = tokens.index("[MASK]")

# ---- BERT: bidirectional attention mask -- every position sees every position
bidirectional_mask = np.ones((n, n))

# ---- GPT (Day 68 preview): causal attention mask -- position i only sees j <= i
causal_mask = np.tril(np.ones((n, n)))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, mask, title in zip(
    axes,
    [bidirectional_mask, causal_mask],
    ["BERT: bidirectional mask\n(masked-LM, Day 67)", "GPT: causal mask\n(next-token LM, Day 68 preview)"],
):
    ax.imshow(mask, cmap="Greys", vmin=0, vmax=1, alpha=0.85)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("key position (attended TO)")
    ax.set_ylabel("query position (attending FROM)")
    ax.set_title(title, fontsize=10)

    # highlight the [MASK] row: what can it legally attend to?
    ax.add_patch(plt.Rectangle((-0.5, mask_pos - 0.5), n, 1, fill=False, edgecolor="crimson", linewidth=2.2))

    for i in range(n):
        for j in range(n):
            if mask[i, j] == 0:
                ax.text(j, i, "X", ha="center", va="center", color="crimson", fontsize=7)

axes[0].text(n - 0.5, mask_pos, "  <- sees left AND right context", fontsize=8, color="crimson", va="center")
axes[1].text(n - 0.5, mask_pos, "  <- can only see left (\"the cat\")", fontsize=8, color="crimson", va="center")

fig.suptitle('Predicting "[MASK]" = "sat": full context (BERT) vs. left-only context (causal)', fontsize=11)
plt.tight_layout()
plt.savefig("graph_DAY_67.png", dpi=120, bbox_inches="tight")
