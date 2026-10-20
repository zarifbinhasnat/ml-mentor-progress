"""
Day 140 - Consolidation: the dot-product similarity thread
(neuron w.x -> attention q.k/sqrt(d) -> CLIP/SimCLR InfoNCE with temperature -> RAG retrieval by cosine).
Left : std of a dot product of two random d-dim vectors grows like sqrt(d); dividing by sqrt(d) (or unit-normalising) keeps it flat.
Right: softmax over 8 candidate cosine similarities at different temperatures (the InfoNCE / retrieval sharpness knob).
Synthetic, seed 42, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---- Left: dot-product scale vs dimension ------------------------------------------------
dims = np.array([4, 16, 64, 256, 1024])
raw_std, scaled_std = [], []
for d in dims:
    q = rng.standard_normal((400, d))
    k = rng.standard_normal((400, d))
    dots = (q * k).sum(axis=1)                       # 400 independent q.k samples
    raw_std.append(dots.std())
    scaled_std.append((dots / np.sqrt(d)).std())
print("d      std(q.k)   std(q.k/sqrt(d))")
for d, r, s in zip(dims, raw_std, scaled_std):
    print(f"{d:<6d} {r:8.2f}   {s:8.2f}")

# ---- Right: temperature on cosine similarities -------------------------------------------
cos = np.array([0.62, 0.55, 0.48, 0.30, 0.22, 0.10, 0.05, -0.10])   # 1 positive-ish, rest distractors
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
temps = [1.0, 0.1, 0.05]
probs = {t: softmax(cos / t) for t in temps}
for t in temps:
    print(f"tau={t:<5} p(top1)={probs[t][0]:.3f}  p(top2)={probs[t][1]:.3f}  entropy={-(probs[t]*np.log(probs[t])).sum():.2f} nats")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].loglog(dims, raw_std, "o-", label="std(q·k)")
ax[0].loglog(dims, scaled_std, "s-", label="std(q·k / √d)")
ax[0].loglog(dims, np.sqrt(dims), "k--", lw=1, label="√d reference")
ax[0].set_xlabel("dimension d"); ax[0].set_ylabel("standard deviation"); ax[0].set_title("Raw dot products grow as √d")
ax[0].legend()

w = 0.27
x = np.arange(len(cos))
for i, t in enumerate(temps):
    ax[1].bar(x + (i - 1) * w, probs[t], w, label=f"τ = {t}")
ax[1].set_xlabel("candidate (sorted by cosine)"); ax[1].set_ylabel("softmax probability")
ax[1].set_title("Temperature sharpens the similarity softmax")
ax[1].legend()
plt.savefig("graph_DAY_140.png", dpi=120, bbox_inches="tight")
