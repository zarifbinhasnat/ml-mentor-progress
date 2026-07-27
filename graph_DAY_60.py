"""
Residual stream RMS magnitude vs. transformer block depth,
with and without Pre-LN normalization on the block's input.

No normalization: block output scale is proportional to the current
residual stream's scale -> variance compounds MULTIPLICATIVELY each layer
-> exponential blowup.

Pre-LN: LayerNorm resets the block's *input* to unit variance before every
block, so each block's output variance is roughly constant regardless of
depth -> residual variance grows ADDITIVELY (Var(x_L) ~ Var(x_0) + L*sigma^2)
-> RMS grows only like sqrt(depth), not exponentially.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

d = 64          # residual stream (model) dimension
hidden = 4 * d  # FFN expansion, matches the 4x convention
depth = 24      # number of stacked transformer blocks

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

# one distinct pair of FFN weight matrices per block, fixed for reproducibility
Ws1 = [np.random.randn(hidden, d) * 0.1 for _ in range(depth)]
Ws2 = [np.random.randn(d, hidden) * 0.1 for _ in range(depth)]

def ffn_block(x, W1, W2):
    return W2 @ gelu(W1 @ x)

def layernorm(x, eps=1e-5):
    mu = x.mean()
    var = x.var()
    return (x - mu) / np.sqrt(var + eps)

def rms(x):
    return np.sqrt(np.mean(x ** 2))

x0 = np.random.randn(d)

# --- no normalization: block sees the raw, ever-growing residual stream ---
x = x0.copy()
norms_noln = [rms(x)]
for l in range(depth):
    x = x + ffn_block(x, Ws1[l], Ws2[l])
    norms_noln.append(rms(x))

# --- Pre-LN: block sees a normalized copy of x, residual add uses raw x ---
x = x0.copy()
norms_preln = [rms(x)]
for l in range(depth):
    x = x + ffn_block(layernorm(x), Ws1[l], Ws2[l])
    norms_preln.append(rms(x))

layers = np.arange(depth + 1)

plt.figure(figsize=(7.5, 4.8))
plt.plot(layers, norms_noln, marker="o", ms=4, label="No normalization (multiplicative growth)")
plt.plot(layers, norms_preln, marker="s", ms=4, label="Pre-LN block input (additive growth)")
plt.yscale("log")
plt.xlabel("Block depth (layer index)")
plt.ylabel("Residual stream RMS (log scale)")
plt.title("LayerNorm as AGC: residual stream magnitude vs. depth")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.savefig("graph_DAY_60.png", dpi=120, bbox_inches="tight")
