"""
Day 42 — ResNet & residual connections
Shows how a skip connection changes the gradient magnitude flowing back
through a deep stack of layers, vs a plain (non-residual) stack.

Toy setup: a chain of N "blocks". Each plain block multiplies the incoming
gradient by a small random factor (simulating vanishing/exploding gradient
through weight matrices + activation derivatives). A residual block instead
adds 1.0 (the identity path) to that same random factor before multiplying,
which is exactly what happens in backprop through y = x + F(x):
dL/dx = dL/dy * (1 + dF/dx).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N_LAYERS = 40
N_TRIALS = 30

# Per-layer local Jacobian scale: centered below 1 to mimic the classic
# vanishing-gradient regime (small weights / saturating activations).
def sample_local_grads(n_layers, n_trials):
    return np.random.normal(loc=0.75, scale=0.15, size=(n_trials, n_layers))

plain_grad_norms = np.zeros((N_TRIALS, N_LAYERS))
resnet_grad_norms = np.zeros((N_TRIALS, N_LAYERS))

for t in range(N_TRIALS):
    local = sample_local_grads(N_LAYERS, 1).flatten()

    # Plain net: gradient at layer k is the product of all local Jacobians
    # from the output back to layer k -> dL/dx_k = prod(local[k:])
    plain_cum = 1.0
    resnet_cum = 1.0
    for k in range(N_LAYERS - 1, -1, -1):
        plain_cum *= local[k]                 # y = F(x)      -> dy/dx = F'(x)
        resnet_cum *= (1.0 + local[k])         # y = x + F(x)  -> dy/dx = 1 + F'(x)
        plain_grad_norms[t, k] = abs(plain_cum)
        resnet_grad_norms[t, k] = abs(resnet_cum)

plain_mean = plain_grad_norms.mean(axis=0)
resnet_mean = resnet_grad_norms.mean(axis=0)
depth = np.arange(1, N_LAYERS + 1)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(depth, plain_mean, label="Plain stack: grad ~ prod(F')", color="#d62728", linewidth=2)
ax.plot(depth, resnet_mean, label="Residual stack: grad ~ prod(1+F')", color="#1f77b4", linewidth=2)
ax.set_yscale("log")
ax.set_xlabel("Layers back-propagated through (depth)")
ax.set_ylabel("|gradient| reaching that layer (log scale)")
ax.set_title("Skip connections keep gradient magnitude alive across depth")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

plt.savefig("graph_DAY_42.png", dpi=120, bbox_inches="tight")
