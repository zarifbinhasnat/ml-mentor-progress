"""
Day 61 — Encoder-only vs decoder-only vs encoder-decoder attention masks.

Visualizes the additive attention mask (0 = attend, -inf shown as black/blocked)
for three architectures, applied to QK^T/sqrt(d) before softmax:
  1. Encoder self-attention: fully bidirectional, no mask.
  2. Decoder self-attention: causal mask, token i only sees j <= i.
  3. Decoder cross-attention: decoder queries (m tokens) attend freely over
     the FINISHED encoder output (n tokens) -- no causal restriction needed
     because the whole encoder sequence is already known.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_enc = 10   # encoder sequence length
m_dec = 7    # decoder sequence length

# 1. Encoder self-attention mask: fully open (all zeros = fully allowed)
encoder_mask = np.zeros((n_enc, n_enc))

# 2. Decoder self-attention mask: causal (upper triangle, excluding diagonal, is blocked)
decoder_mask = np.triu(np.ones((m_dec, m_dec)), k=1)  # 1 = blocked, 0 = allowed

# 3. Cross-attention mask: decoder queries (rows) x encoder keys (cols), fully open
cross_mask = np.zeros((m_dec, n_enc))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

titles = [
    "Encoder self-attention\n(bidirectional, no mask)",
    "Decoder self-attention\n(causal mask)",
    "Decoder cross-attention\n(queries -> finished encoder output)",
]
data = [encoder_mask, decoder_mask, cross_mask]
xlabels = ["key position (encoder)", "key position (decoder, past+self)", "key position (encoder)"]
ylabels = ["query position (encoder)", "query position (decoder)", "query position (decoder)"]

for ax, mat, title, xl, yl in zip(axes, data, titles, xlabels, ylabels):
    # cmap: white = allowed (0), black = blocked (1)
    ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.tick_params(labelsize=7)

fig.suptitle(
    "Three attention wiring patterns: who is allowed to look at whom (white = attend, black = blocked)",
    fontsize=11,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("graph_DAY_61.png", dpi=120, bbox_inches="tight")
