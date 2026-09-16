"""
Day 108 - RAG (Retrieval-Augmented Generation)

Visualizes the retrieval step of RAG: a query embedding dropped into a
2D "embedding space" of document embeddings, with the top-k nearest
documents (by cosine similarity) highlighted and connected.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 40 synthetic "document embeddings" clustered into 4 topics
n_per_cluster = 10
centers = np.array([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])
docs = np.vstack([
    c + 0.6 * np.random.randn(n_per_cluster, 2) for c in centers
])
topic_ids = np.repeat(np.arange(4), n_per_cluster)

# Query embedding: lands near the top-right cluster but off-center
query = np.array([1.3, 2.6])

# Cosine similarity between query and every doc embedding
def cosine_sim(q, D):
    q_norm = q / np.linalg.norm(q)
    D_norm = D / np.linalg.norm(D, axis=1, keepdims=True)
    return D_norm @ q_norm

sims = cosine_sim(query, docs)
k = 5
top_k_idx = np.argsort(-sims)[:k]

fig, ax = plt.subplots(figsize=(6, 6))

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for cluster in range(4):
    mask = topic_ids == cluster
    ax.scatter(docs[mask, 0], docs[mask, 1], c=colors[cluster], s=60,
               alpha=0.55, label=f"topic {cluster}", edgecolors="none")

# Highlight retrieved top-k docs
ax.scatter(docs[top_k_idx, 0], docs[top_k_idx, 1], s=220,
           facecolors="none", edgecolors="black", linewidths=2.0,
           label=f"top-{k} retrieved", zorder=5)

# Draw similarity lines from query to retrieved docs
for idx in top_k_idx:
    ax.plot([query[0], docs[idx, 0]], [query[1], docs[idx, 1]],
            color="black", linewidth=0.8, alpha=0.6, zorder=4)

ax.scatter(*query, marker="*", s=400, c="gold", edgecolors="black",
           linewidths=1.2, label="query", zorder=6)

ax.set_title("RAG retrieval: query vs. document embedding space")
ax.set_xlabel("embedding dim 1")
ax.set_ylabel("embedding dim 2")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.set_aspect("equal")

plt.savefig("graph_DAY_108.png", dpi=120, bbox_inches="tight")
