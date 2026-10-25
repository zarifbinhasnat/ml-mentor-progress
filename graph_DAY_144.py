"""
Day 144 - Consolidation: the optimisation thread (SGD -> momentum -> Adam -> AdamW -> warmup/cosine).
Left : loss on an ill-conditioned 2-D quadratic for SGD, SGD+momentum, Adam.
Right: warmup + cosine learning-rate schedule, and the Adam effective step it produces.
Synthetic, deterministic, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
H = np.array([100.0, 1.0])                       # curvature per axis: condition number 100
loss = lambda w: 0.5 * np.sum(H * w ** 2)
grad = lambda w: H * w
w0 = np.array([1.0, 1.0])
T = 100

def run(opt, lr):
    w, m, v, hist = w0.copy(), np.zeros(2), np.zeros(2), []
    for t in range(1, T + 1):
        g = grad(w)
        if opt == "sgd":
            w = w - lr * g
        elif opt == "mom":
            m = 0.9 * m + g
            w = w - lr * m
        else:                                     # adam
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g ** 2
            w = w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        hist.append(loss(w))
    return np.array(hist)

curves = {"SGD lr=0.019": run("sgd", 0.019),      # lr must stay below 2/100 or the stiff axis diverges
          "SGD+momentum lr=0.01": run("mom", 0.01),
          "Adam lr=0.1": run("adam", 0.1)}
for k, c in curves.items():
    print(f"{k:24s} final loss after {T} steps: {c[-1]:.2e}")

# schedule: linear warmup then cosine decay
steps = np.arange(1000)
warm, peak = 100, 1.0
lr_t = np.where(steps < warm, peak * steps / warm,
                0.5 * peak * (1 + np.cos(np.pi * (steps - warm) / (len(steps) - warm))))
print(f"lr at step 0/50/100/550/999: {lr_t[0]:.2f} {lr_t[50]:.2f} {lr_t[100]:.2f} {lr_t[550]:.2f} {lr_t[999]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, c in curves.items():
    ax[0].semilogy(c, label=k)
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss (log)")
ax[0].set_title("Condition number 100: momentum and Adam beat plain SGD"); ax[0].legend()
ax[1].plot(steps, lr_t)
ax[1].axvline(warm, ls=":", color="gray")
ax[1].set_xlabel("step"); ax[1].set_ylabel("lr / peak lr")
ax[1].set_title("Warmup then cosine decay")
plt.tight_layout()
plt.savefig("graph_DAY_144.png", dpi=120, bbox_inches="tight")
