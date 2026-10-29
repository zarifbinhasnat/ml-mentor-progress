"""
Day 148 - Consolidation: the optimisation-failure thread (init scale -> gradient flow -> learning rate -> loss curves).
Left : backprop gradient norm per layer in a 30-layer ReLU MLP for three init scales (vanish / He / explode).
Right: gradient descent on a quadratic with curvature h=10 -- stable iff lr < 2/h = 0.2.
Synthetic, deterministic (seed 42), numpy only, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: gradient norm vs depth for different init gains (std = gain * sqrt(1/fan_in))
L, width = 30, 64
x = np.random.randn(width)
def grad_norms(gain):
    Ws = [np.random.randn(width, width) * gain / np.sqrt(width) for _ in range(L)]
    h, masks = x, []
    for W in Ws:                                   # forward, remembering ReLU masks
        z = W @ h; masks.append(z > 0); h = np.maximum(z, 0)
    g = np.ones(width) / np.sqrt(width)            # unit-norm gradient arriving at the top
    norms = [np.linalg.norm(g)]
    for W, m in zip(Ws[::-1], masks[::-1]):        # backward through the layers
        g = W.T @ (g * m); norms.append(np.linalg.norm(g))
    return norms[::-1]                             # index 0 = input layer
gains = {"Xavier-ish (gain 1.0): vanishes": 1.0, "He (gain sqrt 2): stable": np.sqrt(2), "too big (gain 2.0): explodes": 2.0}

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for (name, g), c in zip(gains.items(), ["tab:red", "tab:green", "tab:purple"]):
    n = grad_norms(g)
    ax[0].semilogy(range(L + 1), n, "o-", ms=3, color=c, label=name)
    print(f"{name:34s} grad norm at input layer = {n[0]:.2e}")
ax[0].set_xlabel("layer (0 = input, 30 = output)"); ax[0].set_ylabel("||dL/dh|| (log scale)")
ax[0].set_title("Init scale decides if gradients survive depth"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

# ---- right: GD on L(w) = 0.5 * h * w^2, h = 10  ->  w <- (1 - lr*h) w, stable iff |1 - lr*h| < 1
h = 10.0
for lr, c in [(0.005, "tab:blue"), (0.05, "tab:green"), (0.19, "tab:orange"), (0.21, "tab:red")]:
    w, ls = 1.0, []
    for _ in range(40):
        ls.append(0.5 * h * w * w); w -= lr * h * w
    ax[1].semilogy(ls, color=c, label=f"lr={lr}  (rate |1-lr*h|={abs(1-lr*h):.2f})")
    print(f"lr={lr}: loss after 40 steps = {ls[-1]:.2e}")
ax[1].set_ylim(1e-12, 1e8)
ax[1].set_xlabel("step"); ax[1].set_ylabel("loss (log scale)")
ax[1].set_title("Too small: slow. Near 2/h: ringing. Past 2/h: divergence"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

plt.savefig("graph_DAY_148.png", dpi=120, bbox_inches="tight")
