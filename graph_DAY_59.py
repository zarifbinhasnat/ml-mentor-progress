import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

d = 8  # embedding dim -> 4 rotation pairs
base = 10000.0
n_pairs = d // 2
theta = base ** (-2 * np.arange(n_pairs) / d)  # same freq schedule as sinusoidal PE (Day 58)


def rope_rotate(x, pos):
    """Rotate each (x0, x1) pair of x by angle pos * theta_i."""
    x = x.reshape(*x.shape[:-1], n_pairs, 2)
    angles = pos * theta
    cos, sin = np.cos(angles), np.sin(angles)
    x0, x1 = x[..., 0], x[..., 1]
    rot0 = x0 * cos - x1 * sin
    rot1 = x0 * sin + x1 * cos
    return np.stack([rot0, rot1], axis=-1).reshape(*x.shape[:-2], d)


q = np.random.randn(d)
k = np.random.randn(d)

deltas = np.arange(0, 65)


def score_curve(m_base):
    scores = []
    for delta in deltas:
        n = m_base + delta
        rq = rope_rotate(q, m_base)
        rk = rope_rotate(k, n)
        scores.append(rq @ rk)
    return np.array(scores)


curve_m0 = score_curve(0)
curve_m1000 = score_curve(1000)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(deltas, curve_m0, label="absolute position m=0", lw=2.5, color="#2563eb")
ax.plot(deltas, curve_m1000, "--", label="absolute position m=1000", lw=2.5, color="#f97316")
ax.set_xlabel("relative offset  Δ = n − m")
ax.set_ylabel("RoPE(q, m) · RoPE(k, n)")
ax.set_title("RoPE attention score depends only on relative position (m=0 vs m=1000 overlap)")
ax.legend()
ax.grid(alpha=0.3)
plt.savefig("graph_DAY_59.png", dpi=120, bbox_inches="tight")
