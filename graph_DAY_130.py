"""
Day 130 - Consolidation: the identity-path (skip connection) thread.

Left : gradient norm reaching the input vs depth. Plain stack h' = tanh(W h) versus residual stack
       h' = h + 0.2*tanh(W h). Same random weights, same width, seed 42.
Right: frequency response. A block F that is a low-pass filter kills high frequencies when stacked
       (|H|^L), but the residual block x + a*F(x) has gain |1 + a*H| >= 1 at every frequency.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: backward gradient norm vs depth ----
n, depth = 32, 40
Ws = [np.random.randn(n, n) / np.sqrt(n) for _ in range(depth)]
x0 = np.random.randn(n)

def grad_norms(residual):
    h, Js = x0.copy(), []
    for W in Ws:
        a = np.tanh(W @ h)
        Jf = (1 - a ** 2)[:, None] * W            # d tanh(Wh) / dh
        if residual:
            Js.append(np.eye(n) + 0.2 * Jf)
            h = h + 0.2 * a
        else:
            Js.append(Jf)
            h = a
    g = np.ones(n) / np.sqrt(n)                   # unit upstream gradient at the output
    norms = [np.linalg.norm(g)]
    for J in reversed(Js):
        g = J.T @ g                               # one backward step
        norms.append(np.linalg.norm(g))
    return np.array(norms)

plain, res = grad_norms(False), grad_norms(True)
layers = np.arange(depth + 1)

# ---- right: frequency response of a stacked low-pass block ----
w = np.linspace(0, np.pi, 200)
H = (2 + 2 * np.cos(w)) / 4                       # 3-tap [1,2,1]/4 low-pass: H(0)=1, H(pi)=0
L = 10
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(layers, plain, label="plain (tanh stack)")
ax[0].semilogy(layers, res, label="residual (x + 0.2 F(x))")
ax[0].set_xlabel("layers back-propagated through")
ax[0].set_ylabel("gradient norm")
ax[0].set_title("Identity path keeps the gradient alive")
ax[0].legend()

alpha = 0.3
ax[1].semilogy(w / np.pi, np.abs(H) ** L + 1e-16, label=f"plain block F: |H|^{L}")
ax[1].semilogy(w / np.pi, np.abs(1 + alpha * H) ** L, label=f"residual x+{alpha}F(x): |1+{alpha}H|^{L}")
ax[1].set_xlabel("normalised frequency (1 = Nyquist)")
ax[1].set_ylabel("gain after 10 blocks (log)")
ax[1].set_title("Low-pass blocks erase high frequencies; skips keep them")
ax[1].legend(fontsize=8)

plt.savefig("graph_DAY_130.png", dpi=120, bbox_inches="tight")
print("grad @ depth40  plain %.2e  residual %.2e" % (plain[-1], res[-1]))
print("gain at Nyquist after %d blocks: plain %.1e  residual %.3f" % (L, abs(H[-1]) ** L, abs(1 + alpha * H[-1]) ** L))
print("gain at w=pi/2: plain %.1e  residual %.3f" % (abs(H[100]) ** L, abs(1 + alpha * H[100]) ** L))
print("gain at DC: plain %.2f  residual %.2f" % (abs(H[0]) ** L, abs(1 + alpha * H[0]) ** L))
