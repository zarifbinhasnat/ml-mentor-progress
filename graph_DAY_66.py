"""
Day 66 - Embeddings -> Contextual Embeddings
Visualizes why a single static embedding for a polysemous word ("bank") is a
compromise: a self-attention-style weighted average over context words pulls
the *same* query word to different points in space depending on what
sentence it appears in.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Toy 2D "embedding space" ----------------------------------------
# Two semantic clusters that "bank" can belong to: river-related and
# finance-related. Coordinates are hand-placed cluster centers + small noise.
river_words = ["river", "stream", "water", "shore"]
finance_words = ["loan", "money", "finance", "deposit"]

river_center = np.array([-2.2, 1.4])
finance_center = np.array([2.4, -1.2])

river_vecs = river_center + np.random.randn(len(river_words), 2) * 0.35
finance_vecs = finance_center + np.random.randn(len(finance_words), 2) * 0.35

context_words = river_words + finance_words
context_vecs = np.vstack([river_vecs, finance_vecs])

# ---- Static embedding: one fixed vector for "bank", independent of context
bank_static = context_vecs.mean(axis=0)

# ---- Contextual embeddings: attention-weighted average of the words that
# actually co-occur with "bank" in a given sentence (self-attention, single
# query). Score = dot product between the static query and each context
# word's vector; softmax turns scores into attention weights.
def contextual_embedding(query, keys):
    scores = keys @ query                      # dot-product attention scores
    weights = np.exp(scores - scores.max())     # softmax (numerically stable)
    weights /= weights.sum()
    return weights @ keys, weights               # weighted sum = contextual vector

sentences = {
    "river bank\n(fish/shore)": river_vecs,
    "bank loan\n(deposit/finance)": finance_vecs,
}

fig, ax = plt.subplots(figsize=(7, 6))

# Context words
ax.scatter(river_vecs[:, 0], river_vecs[:, 1], c="#2f9e44", s=70, label="river-context words", zorder=3)
ax.scatter(finance_vecs[:, 0], finance_vecs[:, 1], c="#1971c2", s=70, label="finance-context words", zorder=3)
for w, (x, y) in zip(context_words, context_vecs):
    ax.annotate(w, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9, color="dimgray")

# Static embedding: one fixed point, no matter the sentence
ax.scatter(*bank_static, marker="X", s=220, c="black", zorder=5, label='static embedding of "bank"')
ax.annotate('"bank" (static, word2vec-style)', bank_static, textcoords="offset points",
            xytext=(10, -14), fontsize=9, fontweight="bold")

colors = ["#2f9e44", "#1971c2"]
for (label, keys), color in zip(sentences.items(), colors):
    ctx_vec, _ = contextual_embedding(bank_static, keys)
    ax.scatter(*ctx_vec, marker="*", s=320, c=color, edgecolors="black", linewidths=0.8, zorder=6)
    ax.annotate(f'"bank" in:\n{label}', ctx_vec, textcoords="offset points",
                xytext=(10, 8), fontsize=9, fontweight="bold", color=color)
    ax.annotate("", xy=ctx_vec, xytext=bank_static,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.6, alpha=0.8))

ax.set_title('Static vs. contextual embeddings of the word "bank"')
ax.set_xlabel("embedding dim 1 (toy 2D projection)")
ax.set_ylabel("embedding dim 2 (toy 2D projection)")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.25)
ax.set_aspect("equal")

plt.savefig("graph_DAY_66.png", dpi=120, bbox_inches="tight")
