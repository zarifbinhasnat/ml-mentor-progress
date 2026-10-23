"""
Day 142 - Consolidation: the identity-path thread
(ResNet skip -> LSTM cell state -> Transformer residual stream -> U-Net skips -> vanishing gradients).
Left : gradient norm through L layers, plain stack of Jacobians W vs residual stack (I + W).
Right: frequency response of a block y = F(x) vs y = x + F(x) where F is a 5-tap low-pass FIR.
Synthetic, seed 42, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---- Left: backward signal through depth ------------------------------------------------
width, depth = 16, 60
v0 = rng.standard_normal(width)
v0 /= np.linalg.norm(v0)
Ws = [0.8 * rng.standard_normal((width, width)) / np.sqrt(width) * 0.9 for _ in range(depth)]  # spectral radius < 1
plain, resid = [1.0], [1.0]
vp, vr = v0.copy(), v0.copy()
for W in Ws:
    vp = W.T @ vp                  # plain layer: grad <- J^T grad
    vr = (np.eye(width) + 0.1 * W).T @ vr   # residual layer: grad <- (I + J)^T grad
    plain.append(np.linalg.norm(vp))
    resid.append(np.linalg.norm(vr))
print(f"after {depth} layers: plain grad norm = {plain[-1]:.2e}, residual grad norm = {resid[-1]:.3f}")

# ---- Right: |H(w)| of plain block vs residual block --------------------------------------
h = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0      # F: binomial low-pass, DC gain 1
w = np.linspace(0, np.pi, 257)
G = np.array([np.sum(h * np.exp(-1j * wi * np.arange(5))) for wi in w])
F_plain = 0.5 * G                    # block learns a scaled low-pass: kills high frequencies
F_res = 1.0 + 0.5 * G                # identity + residual branch
for name, wi in (("DC", 0), ("mid", 128), ("Nyquist", 256)):
    print(f"{name:8s} |plain|={abs(F_plain[wi]):.4f}  |residual|={abs(F_res[wi]):.4f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(plain, label="plain: $\\prod W$")
ax[0].semilogy(resid, label="residual: $\\prod (I+0.1W)$")
ax[0].set_xlabel("layers traversed"); ax[0].set_ylabel("gradient norm"); ax[0].set_title("Backward signal vs depth"); ax[0].legend()
ax[1].plot(w / np.pi, np.abs(F_plain), label="plain block $0.5\\,G(\\omega)$")
ax[1].plot(w / np.pi, np.abs(F_res), label="residual block $1+0.5\\,G(\\omega)$")
ax[1].set_xlabel("normalised frequency ($\\times\\pi$ rad/sample)"); ax[1].set_ylabel("|H|"); ax[1].set_title("Skip keeps every frequency"); ax[1].legend()
plt.savefig("graph_DAY_142.png", dpi=120, bbox_inches="tight")
