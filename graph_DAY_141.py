"""
Day 141 - Consolidation: the loss-is-a-likelihood thread
(MSE = Gaussian NLL -> cross-entropy = categorical NLL -> VAE ELBO -> diffusion eps-MSE -> RLHF KL penalty).
Left : gradient of the loss w.r.t. the logit z for a confidently WRONG sigmoid unit (y=1): MSE vs binary cross-entropy.
Right: Gaussian NLL with fixed sigma is an affine function of MSE (same minimiser), checked on 40 synthetic points.
Synthetic, seed 42, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---- Left: gradients wrt logit z, target y = 1 --------------------------------------------
z = np.linspace(-8, 4, 241)
p = 1 / (1 + np.exp(-z))
grad_mse = (p - 1) * p * (1 - p)        # d/dz of 0.5*(p-y)^2 : the sigmoid derivative p(1-p) kills it
grad_bce = p - 1                        # d/dz of -log p      : the sigmoid derivative cancels
for zz in (-6.0, -2.0, 0.0):
    pp = 1 / (1 + np.exp(-zz))
    print(f"z={zz:+.0f}  p={pp:.4f}  |grad MSE|={abs((pp-1)*pp*(1-pp)):.4f}  |grad BCE|={abs(pp-1):.4f}")

# ---- Right: Gaussian NLL vs MSE on a tiny regression set ----------------------------------
x = np.linspace(-1, 1, 40)
y = 2.0 * x + 0.3 * rng.standard_normal(40)
slopes = np.linspace(0, 4, 81)
sigma = 0.3
mse = np.array([np.mean((y - a * x) ** 2) for a in slopes])
nll = np.array([np.mean(0.5 * ((y - a * x) / sigma) ** 2 + np.log(sigma * np.sqrt(2 * np.pi))) for a in slopes])
affine_err = np.max(np.abs(nll - (mse / (2 * sigma ** 2) + np.log(sigma * np.sqrt(2 * np.pi)))))
print(f"argmin MSE slope = {slopes[mse.argmin()]:.2f}, argmin NLL slope = {slopes[nll.argmin()]:.2f}, max |NLL - affine(MSE)| = {affine_err:.2e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(z, np.abs(grad_mse), label="|dL/dz|  MSE on sigmoid")
ax[0].plot(z, np.abs(grad_bce), label="|dL/dz|  cross-entropy")
ax[0].set_xlabel("logit z (target y = 1, so z << 0 is confidently wrong)")
ax[0].set_ylabel("gradient magnitude")
ax[0].set_title("Likelihood loss keeps the gradient alive")
ax[0].legend()

ax[1].plot(slopes, mse, label="MSE")
ax2 = ax[1].twinx()
ax2.plot(slopes, nll, "C1--", label="Gaussian NLL (σ=0.3)")
ax[1].axvline(slopes[mse.argmin()], color="k", lw=1, ls=":")
ax[1].set_xlabel("model slope a"); ax[1].set_ylabel("MSE"); ax2.set_ylabel("NLL")
ax[1].set_title("MSE and Gaussian NLL share one minimiser")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[1].legend(h1 + h2, l1 + l2, loc="upper center")
plt.savefig("graph_DAY_141.png", dpi=120, bbox_inches="tight")
