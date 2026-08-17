"""
Day 81 - GANs (minimax game)

Toy 1D GAN trained with plain NumPy (no autodiff, no torch) to make the
minimax dynamics visible:

  D(x)    = sigmoid(w*x + b)                  -- discriminator (logistic)
  G(z)    = a*z + c,  z ~ N(0,1)               -- generator (affine)
  x_real  ~ N(4, 1.25^2)                       -- tiny fixed real dataset

We alternate: a few gradient-ASCENT steps on D to maximize
    V = E[log D(x)] + E[log(1 - D(G(z)))]
then one gradient-ASCENT step on G on the non-saturating objective
    E[log D(G(z))]   (instead of minimizing E[log(1 - D(G(z)))]).

Left panel: D's and G's loss curves over training (classic "wobble" of
adversarial training). Right panel: real vs. generated histograms at the
end of training, showing G's distribution converging onto p_data.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)  # fixed seed -> reproducible run

# ---- tiny fixed "real data" dataset (64 points, < 100) ----
x_real = rng.normal(loc=4.0, scale=1.25, size=64)

# ---- discriminator params: D(x) = sigmoid(w*x + b) ----
w, b = 0.05, 0.0
# ---- generator params: G(z) = a*z + c ----
a, c = 3.0, -2.0  # starts wide and centered at -2, far from the real mean of 4

sigmoid = lambda s: 1.0 / (1.0 + np.exp(-s))

lr_d, lr_g = 0.05, 0.3
d_steps_per_g_step = 1
n_iters = 800
batch = 32

d_loss_hist, g_loss_hist = [], []

for it in range(n_iters):
    # ---- D steps: ascend V(D, G) ----
    for _ in range(d_steps_per_g_step):
        xb = rng.choice(x_real, size=batch, replace=True)
        zb = rng.normal(size=batch)
        gz = a * zb + c

        Dx = sigmoid(w * xb + b)
        Dg = sigmoid(w * gz + b)

        # dV/dw, dV/db (derived from d/ds log(sigmoid(s)) = 1 - sigmoid(s))
        dV_dw = np.mean((1 - Dx) * xb) + np.mean(-Dg * gz)
        dV_db = np.mean(1 - Dx) + np.mean(-Dg)

        w += lr_d * dV_dw
        b += lr_d * dV_db

    # ---- G step: ascend the non-saturating objective E[log D(G(z))] ----
    zb = rng.normal(size=batch)
    gz = a * zb + c
    Dg = sigmoid(w * gz + b)

    dLogD_da = np.mean((1 - Dg) * w * zb)
    dLogD_dc = np.mean((1 - Dg) * w)

    a += lr_g * dLogD_da
    c += lr_g * dLogD_dc

    # ---- bookkeeping for the loss curves ----
    xb = rng.choice(x_real, size=batch, replace=True)
    zb = rng.normal(size=batch)
    gz = a * zb + c
    Dx = sigmoid(w * xb + b)
    Dg = sigmoid(w * gz + b)
    eps = 1e-8
    d_loss = -(np.mean(np.log(Dx + eps)) + np.mean(np.log(1 - Dg + eps)))
    g_loss = -np.mean(np.log(Dg + eps))
    d_loss_hist.append(d_loss)
    g_loss_hist.append(g_loss)

# ---- final samples for the histogram panel ----
z_final = rng.normal(size=500)
x_generated = a * z_final + c
x_real_big = rng.normal(loc=4.0, scale=1.25, size=500)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].plot(d_loss_hist, label="D loss", color="#d62728", linewidth=1.5)
axes[0].plot(g_loss_hist, label="G loss", color="#1f77b4", linewidth=1.5)
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("loss")
axes[0].set_title("Minimax game: D and G loss curves")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(x_real_big, bins=25, alpha=0.55, label="real $p_{data}$", color="#2ca02c", density=True)
axes[1].hist(x_generated, bins=25, alpha=0.55, label="generated $p_g$", color="#1f77b4", density=True)
axes[1].set_xlabel("x")
axes[1].set_ylabel("density")
axes[1].set_title(f"After training: G(z) = {a:.2f}z + {c:.2f}  (real: N(4, 1.25))")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("Day 81 -- GANs: the minimax game (note G's variance undershoot)", y=1.02)
plt.savefig("graph_DAY_81.png", dpi=120, bbox_inches="tight")
print(f"final generator: a={a:.3f}, c={c:.3f} (target: a~1.25, c~4.0)")
