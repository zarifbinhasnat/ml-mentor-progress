"""
Day 132 - Consolidation: the scale-control thread.

Left : activation std per layer in a 40-layer ReLU MLP (width 64, 32 random inputs, seed 42)
       for three inits: too small (std 0.05), Xavier-like (1/sqrt(fan_in)), He (sqrt(2/fan_in)).
Right: attention softmax with d = 64: logit std and mean max-probability, with and without the 1/sqrt(d) scale.
All data synthetic, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
width, depth, batch = 64, 40, 32


def layer_stds(w_std):
    h = np.random.randn(batch, width)
    out = [h.std()]
    for _ in range(depth):
        W = np.random.randn(width, width) * w_std(width)
        h = np.maximum(h @ W, 0.0)                       # ReLU layer
        out.append(h.std())
    return np.array(out)


inits = {
    "too small (std 0.05)": lambda n: 0.05,
    "Xavier-like 1/sqrt(n)": lambda n: np.sqrt(1.0 / n),
    "He sqrt(2/n)": lambda n: np.sqrt(2.0 / n),
}
stds = {name: layer_stds(f) for name, f in inits.items()}

d, T = 64, 16
Q = np.random.randn(T, d)
K = np.random.randn(T, d)
raw = Q @ K.T                                            # unscaled logits, std ~ sqrt(d)
scaled = raw / np.sqrt(d)                                # std ~ 1


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


stats = {
    "logit std": (raw.std(), scaled.std()),
    "mean max prob": (softmax(raw).max(axis=1).mean(), softmax(scaled).max(axis=1).mean()),
}

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
for name, s in stds.items():
    a.semilogy(s, label=name)
a.set_xlabel("layer")
a.set_ylabel("activation std (log)")
a.set_title("Signal scale through 40 ReLU layers")
a.legend()
a.grid(alpha=0.3)

xs = np.arange(len(stats))
b.bar(xs - 0.2, [v[0] for v in stats.values()], 0.4, label="no scaling")
b.bar(xs + 0.2, [v[1] for v in stats.values()], 0.4, label="divide by sqrt(d)")
b.set_xticks(xs)
b.set_xticklabels(list(stats))
b.set_title("Attention logits, d = 64")
b.legend()
b.grid(alpha=0.3, axis="y")

print({k: s[-1] for k, s in stds.items()})
print({k: (round(float(v[0]), 3), round(float(v[1]), 3)) for k, v in stats.items()})
plt.savefig("graph_DAY_132.png", dpi=120, bbox_inches="tight")
