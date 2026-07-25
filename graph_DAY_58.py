"""
Day 58 — Sinusoidal Positional Encoding
Left panel:  heatmap of the PE(pos, dim) matrix — the pattern actually added to token embeddings.
Right panel: a handful of individual dimension pairs plotted across position, showing how
             frequency drops geometrically as the dimension index increases (fast "seconds hand"
             low dims vs slow "hour hand" high dims).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

d_model = 64
max_pos = 50

positions = np.arange(max_pos)[:, None]                       # (max_pos, 1)
dims = np.arange(d_model)[None, :]                             # (1, d_model)

# angle_rates[i] = 1 / 10000^(2*floor(i/2)/d_model) — even/odd dim pairs share a frequency
angle_rates = 1.0 / np.power(10000, (2 * (dims // 2)) / d_model)
angles = positions * angle_rates                                # (max_pos, d_model)

pe = np.zeros((max_pos, d_model))
pe[:, 0::2] = np.sin(angles[:, 0::2])                            # even dims -> sin
pe[:, 1::2] = np.cos(angles[:, 1::2])                            # odd dims  -> cos

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

im = axes[0].imshow(pe.T, aspect="auto", cmap="RdBu", origin="lower")
axes[0].set_xlabel("position")
axes[0].set_ylabel("embedding dimension")
axes[0].set_title("PE(pos, dim) heatmap")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

sample_dims = [0, 8, 24, 48]
for d in sample_dims:
    axes[1].plot(positions[:, 0], pe[:, d], label=f"dim {d}")
axes[1].set_xlabel("position")
axes[1].set_ylabel("PE value")
axes[1].set_title("Frequency drops as dim index rises")
axes[1].legend(fontsize=8)

fig.suptitle("Day 58 — Sinusoidal Positional Encoding")
fig.tight_layout()
plt.savefig("graph_DAY_58.png", dpi=120, bbox_inches="tight")
