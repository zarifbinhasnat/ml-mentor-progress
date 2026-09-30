"""
Day 121 - Consolidation Phase 6: why sqrt(d) and why sinusoids are relative.

Left: for random q, k with unit-variance entries, q.k has variance d. Unscaled softmax
saturates (max weight -> 1, gradient -> 0) as d grows; dividing by sqrt(d) keeps it stable.
Right: the sinusoidal positional-encoding dot product PE(p).PE(p+k) depends only on the
offset k (a relative-position kernel), and looks like a low-pass bump around k=0.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

n_keys, n_trials = 16, 64
dims = np.array([4, 16, 64, 256, 1024])
max_plain, max_scaled = [], []
for d in dims:
    q = np.random.randn(n_trials, 1, d)
    K = np.random.randn(n_trials, n_keys, d)
    logits = (q * K).sum(-1)                       # (trials, keys)
    max_plain.append(softmax(logits).max(-1).mean())
    max_scaled.append(softmax(logits / np.sqrt(d)).max(-1).mean())

def pe(pos, d_model=64):
    i = np.arange(d_model // 2)
    w = 1.0 / (10000 ** (2 * i / d_model))
    return np.concatenate([np.sin(pos * w), np.cos(pos * w)])

offsets = np.arange(-40, 41)
for p in (10, 50, 90):
    kern = np.array([pe(p) @ pe(p + k) for k in offsets])
    # translation invariance: same curve for every start position p
    if p == 10:
        ref = kern
    assert np.allclose(kern, ref)                  # depends on offset only

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(dims, max_plain, "o-", label="softmax(q.k)  (no scaling)")
axes[0].plot(dims, max_scaled, "s-", label="softmax(q.k / sqrt(d))")
axes[0].axhline(1 / n_keys, ls=":", c="gray", label="uniform = 1/16")
axes[0].set_xscale("log", base=2)
axes[0].set_xlabel("head dimension d")
axes[0].set_ylabel("mean max attention weight")
axes[0].set_title("Unscaled softmax saturates as d grows")
axes[0].legend()

axes[1].plot(offsets, ref)
axes[1].set_xlabel("relative offset k")
axes[1].set_ylabel("PE(p) . PE(p+k)")
axes[1].set_title("Sinusoidal PE: same kernel for p=10, 50, 90")

print("max weight plain :", np.round(max_plain, 3))
print("max weight scaled:", np.round(max_scaled, 3))
plt.savefig("graph_DAY_121.png", dpi=120, bbox_inches="tight")
