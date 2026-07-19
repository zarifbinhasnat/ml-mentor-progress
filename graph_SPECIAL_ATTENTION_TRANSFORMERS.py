"""
Special Day -- Attention & Transformers exam-prep capstone.
Two panels reusing the lesson's own worked numbers: (1) scaled vs unscaled
softmax attention weights on the same Q/K/V toy example, showing exactly how
skipping the sqrt(d_k) scale sharpens (over-saturates) the distribution;
(2) sinusoidal positional encoding values across dimensions and positions,
showing the multi-frequency structure (fast-varying early dims, slow-varying
later dims) that makes relative position linearly recoverable.
"""
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- Left: scaled vs unscaled attention weights on the worked Q/K/V example ---
ax = axes[0]
K = np.array([[1., 0.], [0., 1.], [1., 1.]])
V = np.array([[1., 0.], [0., 1.], [2., 2.]])
q = np.array([1., 1.])
d_k = 2

scores = K @ q
scaled = scores / np.sqrt(d_k)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

w_scaled = softmax(scaled)
w_unscaled = softmax(scores)

labels = ["key 1", "key 2", "key 3"]
x_pos = np.arange(3)
width = 0.35
ax.bar(x_pos - width/2, w_unscaled, width, color="#d62728", label="unscaled softmax(QK^T)")
ax.bar(x_pos + width/2, w_scaled, width, color="#1f77b4", label=r"scaled softmax(QK^T/$\sqrt{d_k}$)")
ax.set_xticks(x_pos, labels)
ax.set_ylabel("attention weight")
ax.set_title("Same Q,K -- scaling redistributes weight away from the top key")
ax.legend(fontsize=8)

# --- Right: sinusoidal positional encoding, multiple dims x positions ---
ax2 = axes[1]
d_model = 16
positions = np.arange(0, 50)
dims = np.arange(d_model)

pe = np.zeros((len(positions), d_model))
for pos in positions:
    for i in range(0, d_model, 2):
        denom = 10000 ** (i / d_model)
        pe[pos, i] = np.sin(pos / denom)
        if i + 1 < d_model:
            pe[pos, i + 1] = np.cos(pos / denom)

im = ax2.imshow(pe.T, aspect="auto", cmap="RdBu", origin="lower",
                 extent=[positions[0], positions[-1], 0, d_model])
ax2.set_xlabel("position")
ax2.set_ylabel("encoding dimension")
ax2.set_title("Positional encoding: low dims oscillate fast, high dims slow")
fig.colorbar(im, ax=ax2, label="PE value")

plt.tight_layout()
plt.savefig("graph_SPECIAL_ATTENTION_TRANSFORMERS.png", dpi=120, bbox_inches="tight")
