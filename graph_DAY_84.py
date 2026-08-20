"""Day 84 — Diffusion: reverse denoising & objective.

Trains a tiny 1-hidden-layer MLP (plain NumPy, manual backprop — no
framework, no download) to predict the noise eps added to a toy 1D
signal family at a random diffusion timestep t. Then visualizes:
  1. The training loss curve (the epsilon-prediction MSE objective).
  2. A reconstruction: true x0 vs. noisy x_t vs. x0 recovered from the
     network's predicted eps at a fixed, fairly noisy t.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# --- toy 1D "signal" family: sum of two sinusoids with random phase/amp ---
n_points = 40
axis = np.linspace(0, 4 * np.pi, n_points)

def sample_signal(rng):
    a1, a2 = rng.uniform(0.6, 1.0), rng.uniform(0.1, 0.4)
    p1, p2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    return a1 * np.sin(axis + p1) + a2 * np.sin(3 * axis + p2)

# --- linear beta schedule (same closed-form machinery as Day 83) ---
T = 50
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = np.cumprod(alphas)

def forward_sample(x0, t_idx, rng):
    eps = rng.standard_normal(x0.shape)
    ab = alpha_bars[t_idx]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps, eps

# --- tiny 1-hidden-layer MLP: eps_theta(x_t, t) -> predicted eps ---
in_dim, hidden_dim, out_dim = n_points + 1, 64, n_points
scale = np.sqrt(2.0 / in_dim)
W1 = rng.standard_normal((in_dim, hidden_dim)) * scale
b1 = np.zeros(hidden_dim)
W2 = rng.standard_normal((hidden_dim, out_dim)) * np.sqrt(2.0 / hidden_dim)
b2 = np.zeros(out_dim)

def forward(x_in):
    z1 = x_in @ W1 + b1
    h = np.tanh(z1)
    out = h @ W2 + b2
    return out, h, z1

lr = 0.05
batch_size = 32
n_iters = 1500
losses = []

for it in range(n_iters):
    x0_batch = np.stack([sample_signal(rng) for _ in range(batch_size)])
    t_batch = rng.integers(0, T, size=batch_size)
    xt_batch = np.zeros_like(x0_batch)
    eps_batch = np.zeros_like(x0_batch)
    for i in range(batch_size):
        xt_batch[i], eps_batch[i] = forward_sample(x0_batch[i], t_batch[i], rng)

    t_norm = (t_batch / T).reshape(-1, 1)
    x_in = np.concatenate([xt_batch, t_norm], axis=1)

    eps_pred, h, z1 = forward(x_in)
    diff = eps_pred - eps_batch
    loss = np.mean(diff ** 2)
    losses.append(loss)

    # manual backprop (MSE -> linear -> tanh -> linear), Day 9's math in code
    d_out = 2 * diff / diff.size          # dL/d(eps_pred)
    dW2 = h.T @ d_out
    db2 = d_out.sum(axis=0)
    dh = d_out @ W2.T
    dz1 = dh * (1 - np.tanh(z1) ** 2)     # tanh'(z) = 1 - tanh(z)^2
    dW1 = x_in.T @ dz1
    db1 = dz1.sum(axis=0)

    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

# --- reconstruction demo at a fixed, fairly noisy timestep ---
t_demo = 35
x0_demo = sample_signal(rng)
xt_demo, eps_true = forward_sample(x0_demo, t_demo, rng)
x_in_demo = np.concatenate([xt_demo, [t_demo / T]]).reshape(1, -1)
eps_pred_demo, _, _ = forward(x_in_demo)
eps_pred_demo = eps_pred_demo.ravel()

ab = alpha_bars[t_demo]
x0_hat = (xt_demo - np.sqrt(1 - ab) * eps_pred_demo) / np.sqrt(ab)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))

ax = axes[0]
ax.plot(losses, color="tab:blue", lw=1)
ax.set_title(r"Training objective: $\mathbb{E}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$")
ax.set_xlabel("training iteration")
ax.set_ylabel("MSE loss (predicted vs. true noise)")

ax2 = axes[1]
ax2.plot(axis, x0_demo, color="black", lw=2, label="true $x_0$")
ax2.plot(axis, xt_demo, color="tab:red", lw=1, alpha=0.6, label=f"noisy $x_t$ (t={t_demo}, ᾱ_t={ab:.2f})")
ax2.plot(axis, x0_hat, color="tab:green", lw=1.5, ls="--", label=r"recovered $\hat{x}_0$ from $\epsilon_\theta$")
ax2.set_title("One reverse-step estimate: recovering x0 from the predicted noise")
ax2.set_xlabel("position (toy 1D signal)")
ax2.legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig("graph_DAY_84.png", dpi=120, bbox_inches="tight")
