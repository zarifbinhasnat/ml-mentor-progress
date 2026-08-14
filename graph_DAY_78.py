"""
Day 78 — Autoencoders.
Left panel: reconstruction MSE vs bottleneck size k, PCA/linear-autoencoder
(data-adapted basis) vs a fixed 2D-DCT basis (generic, data-agnostic),
on synthetic 8x8 patches whose true structure lives in 5 random directions
(not aligned with low frequencies) -> the data-adapted basis wins early.
Right panel: one example patch reconstructed by both bases at k=5.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- Synthetic 8x8 "image" patches with a true rank-5 subspace ----------
P = 8                     # patch side
D = P * P                 # 64-dim flattened patch
n_samples = 60
true_rank = 5

# random orthonormal basis for the TRUE underlying subspace (not frequency-aligned)
raw = np.random.randn(D, true_rank)
U_true, _ = np.linalg.qr(raw)          # D x true_rank, orthonormal columns
U_true = U_true.T                      # true_rank x D

latent_std = np.array([3.2, 2.4, 1.6, 1.0, 0.5])   # decaying eigenvalue-like spectrum
Z = np.random.randn(n_samples, true_rank) * latent_std
X = Z @ U_true + 0.15 * np.random.randn(n_samples, D)   # n_samples x D

mean = X.mean(axis=0)
Xc = X - mean

# ---------- PCA / optimal linear autoencoder basis (Eckart-Young) ----------
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)   # rows of Vt: principal directions, variance-sorted

# ---------- Fixed 2D-DCT-II orthonormal basis, ordered low-frequency first ----------
def dct_1d_basis(n):
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi / n * (x + 0.5) * k)      # n x n, row k = k-th DCT frequency
    basis[0] *= np.sqrt(1.0 / n)
    basis[1:] *= np.sqrt(2.0 / n)
    return basis                                    # orthonormal rows

phi = dct_1d_basis(P)                               # P x P
dct_basis = np.zeros((D, D))
freqs = np.zeros(D, dtype=int)
idx = 0
for u in range(P):
    for v in range(P):
        dct_basis[idx] = np.outer(phi[u], phi[v]).ravel()   # 2D basis vector, flattened
        freqs[idx] = u + v                                   # total frequency, for zigzag ordering
        idx += 1
order = np.argsort(freqs, kind="stable")
dct_basis = dct_basis[order]                        # D x D, low-frequency rows first

# ---------- Reconstruction MSE vs bottleneck size k ----------
ks = np.arange(1, 21)
mse_pca, mse_dct = [], []
for k in ks:
    Bp = Vt[:k]                                      # k x D, orthonormal
    Xp = (Xc @ Bp.T) @ Bp
    mse_pca.append(np.mean((Xc - Xp) ** 2))

    Bd = dct_basis[:k]
    Xd = (Xc @ Bd.T) @ Bd
    mse_dct.append(np.mean((Xc - Xd) ** 2))

# ---------- Example patch reconstruction at k=5 ----------
k_show = 5
example = Xc[0]
rec_pca = mean + (example @ Vt[:k_show].T) @ Vt[:k_show]
rec_dct = mean + (example @ dct_basis[:k_show].T) @ dct_basis[:k_show]
orig = X[0]

# ---------- Plot ----------
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2),
                          gridspec_kw={"width_ratios": [2.4, 1, 1, 1, 1]})

axes[0].plot(ks, mse_pca, "o-", label="PCA / linear autoencoder (data-adapted)")
axes[0].plot(ks, mse_dct, "s-", label="Fixed 2D-DCT basis (data-agnostic)")
axes[0].axvline(true_rank, color="gray", ls="--", lw=1, label=f"true rank = {true_rank}")
axes[0].set_xlabel("Bottleneck size k")
axes[0].set_ylabel("Reconstruction MSE")
axes[0].set_yscale("log")
axes[0].set_title("Reconstruction error vs bottleneck size")
axes[0].legend(fontsize=7)

for ax, img, title in zip(
    axes[1:],
    [orig, rec_pca, rec_dct, orig - rec_pca],
    [f"original", f"PCA recon (k={k_show})", f"DCT recon (k={k_show})", "PCA residual"],
):
    im = ax.imshow(img.reshape(P, P), cmap="gray")
    ax.set_title(title, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("graph_DAY_78.png", dpi=120, bbox_inches="tight")
