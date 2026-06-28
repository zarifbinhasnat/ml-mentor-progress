import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

# Simulate "internal covariate shift": distributions of pre-activation values
# across 3 training snapshots, before and after BatchNorm
fig = plt.figure(figsize=(13, 8))
fig.patch.set_facecolor("#0f0f1a")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

snapshot_labels = ["Early training", "Mid training", "Late training"]
means_before   = [0.0, 1.8, 3.5]
stds_before    = [1.0, 2.4, 0.4]

gamma, beta = 1.5, 0.3   # learnable affine params

for col, (label, mu, sigma) in enumerate(zip(snapshot_labels, means_before, stds_before)):
    x_raw  = np.random.normal(mu, sigma, 2000)

    # BatchNorm forward
    eps    = 1e-5
    mu_b   = x_raw.mean()
    var_b  = x_raw.var()
    x_hat  = (x_raw - mu_b) / np.sqrt(var_b + eps)
    x_bn   = gamma * x_hat + beta

    # ── Top row: before BN ──────────────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, col])
    ax_top.hist(x_raw, bins=60, color="#e05c5c", edgecolor="none", alpha=0.85, density=True)
    ax_top.set_title(f"{label}\nμ={mu:.1f}  σ={sigma:.1f}",
                     color="white", fontsize=9, pad=6)
    ax_top.set_facecolor("#1a1a2e")
    ax_top.tick_params(colors="gray", labelsize=7)
    for sp in ax_top.spines.values():
        sp.set_edgecolor("#333355")
    ax_top.set_xlim(-8, 12)
    ax_top.set_xlabel("activation value", color="gray", fontsize=7)
    ax_top.set_ylabel("density", color="gray", fontsize=7)

    # ── Bottom row: after BN ─────────────────────────────────────────────
    ax_bot = fig.add_subplot(gs[1, col])
    ax_bot.hist(x_bn, bins=60, color="#5ce0b8", edgecolor="none", alpha=0.85, density=True)
    ax_bot.axvline(x_bn.mean(), color="white", lw=1.2, ls="--", alpha=0.7)
    ax_bot.set_title(f"After BN  (γ={gamma}, β={beta})\nμ≈{x_bn.mean():.2f}  σ≈{x_bn.std():.2f}",
                     color="white", fontsize=9, pad=6)
    ax_bot.set_facecolor("#1a1a2e")
    ax_bot.tick_params(colors="gray", labelsize=7)
    for sp in ax_bot.spines.values():
        sp.set_edgecolor("#333355")
    ax_bot.set_xlim(-6, 6)
    ax_bot.set_xlabel("activation value", color="gray", fontsize=7)
    ax_bot.set_ylabel("density", color="gray", fontsize=7)

# Row labels on the left
fig.text(0.01, 0.75, "Before\nBatch Norm", color="#e05c5c",
         fontsize=10, fontweight="bold", va="center", rotation=90)
fig.text(0.01, 0.28, "After\nBatch Norm", color="#5ce0b8",
         fontsize=10, fontweight="bold", va="center", rotation=90)

fig.suptitle("Batch Normalization: Taming Internal Covariate Shift",
             color="white", fontsize=13, fontweight="bold", y=0.98)

plt.savefig("graph_DAY_28.png", dpi=120, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved graph_DAY_28.png")
