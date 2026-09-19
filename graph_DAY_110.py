import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N = 8   # number of (image, text) pairs
D = 32  # embedding dimension before projection into the shared space

# "Text" embeddings act as the anchors for each concept.
text_emb = np.random.randn(N, D)

# "Image" embeddings are a noisy view of their matching text embedding, mimicking
# a vision encoder that has been trained (via the contrastive loss) to land near
# the right anchor -- but not exactly on it.
image_emb = 0.75 * text_emb + 0.55 * np.random.randn(N, D)


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


t = normalize(text_emb)
v = normalize(image_emb)

sim = v @ t.T  # cosine similarity matrix: rows = images, cols = text

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)

ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xlabel("Text embedding index")
ax.set_ylabel("Image embedding index")
ax.set_title("Contrastive alignment: image-text cosine similarity")

for i in range(N):
    for j in range(N):
        color = "white" if sim[i, j] < 0.5 else "black"
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", color=color, fontsize=7)

# Highlight the diagonal: these are the positive pairs the contrastive loss pulls together.
for i in range(N):
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))

fig.colorbar(im, ax=ax, label="cosine similarity")
plt.savefig("graph_DAY_110.png", dpi=120, bbox_inches="tight")
