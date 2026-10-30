"""
Day 149 - Consolidation: regularization as constraining the function class.
Left : degree-12 polynomial fit to 20 noisy points, ridge (L2) with three penalty strengths.
Right: coefficient magnitudes for L2 (ridge) vs L1 (lasso via ISTA) at matched training fit -- L1 zeroes coefficients.
Synthetic, deterministic (seed 42), numpy only, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n, deg = 20, 12
x = np.sort(np.random.uniform(-1, 1, n))
y = np.sin(2.5 * x) + 0.2 * np.random.randn(n)
xt = np.linspace(-1, 1, 200)
yt = np.sin(2.5 * xt)
Phi = np.vander(x, deg + 1, increasing=True)
Pt = np.vander(xt, deg + 1, increasing=True)

def ridge(lam):
    return np.linalg.solve(Phi.T @ Phi + lam * np.eye(deg + 1), Phi.T @ y)

def lasso(lam, iters=20000):                       # ISTA: gradient step then soft-threshold
    w = np.zeros(deg + 1)
    step = 1.0 / np.linalg.norm(Phi, 2) ** 2
    for _ in range(iters):
        w = w - step * Phi.T @ (Phi @ w - y)
        w = np.sign(w) * np.maximum(np.abs(w) - step * lam, 0)
    return w

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(x, y, c="k", s=18, zorder=3, label="train points")
ax[0].plot(xt, yt, "k--", lw=1, label="true function")
for lam, c in [(1e-9, "tab:red"), (1e-3, "tab:green"), (1.0, "tab:blue")]:
    w = ridge(lam)
    err = np.mean((Pt @ w - yt) ** 2)
    ax[0].plot(xt, np.clip(Pt @ w, -2.5, 2.5), color=c, label=f"lambda={lam:g}  test MSE={err:.3f}")
    print(f"ridge lambda={lam:g}: test MSE={err:.4f}  ||w||2={np.linalg.norm(w):.2f}")
ax[0].set_ylim(-2.5, 2.5); ax[0].set_title("L2 shrinks the function class: smoother fits"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

wr, wl = ridge(1e-3), lasso(0.05)
idx = np.arange(deg + 1)
ax[1].bar(idx - 0.2, np.abs(wr), 0.4, color="tab:blue", label="L2 (ridge)")
ax[1].bar(idx + 0.2, np.abs(wl), 0.4, color="tab:orange", label="L1 (lasso)")
ax[1].set_yscale("symlog", linthresh=1e-2); ax[1].set_xlabel("polynomial coefficient index"); ax[1].set_ylabel("|w|")
ax[1].set_title("L1 sets coefficients exactly to zero"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
print(f"nonzeros: ridge={np.sum(np.abs(wr) > 1e-8)}  lasso={np.sum(np.abs(wl) > 1e-8)} of {deg + 1}")
print(f"train MSE: ridge={np.mean((Phi @ wr - y) ** 2):.4f}  lasso={np.mean((Phi @ wl - y) ** 2):.4f}")
print(f"test  MSE: ridge={np.mean((Pt @ wr - yt) ** 2):.4f}  lasso={np.mean((Pt @ wl - yt) ** 2):.4f}")

plt.savefig("graph_DAY_149.png", dpi=120, bbox_inches="tight")
