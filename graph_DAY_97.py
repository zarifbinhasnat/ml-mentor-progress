"""
Day 97 -- Gradient accumulation & checkpointing

Left panel: memory footprint M(c) = L/c + c for activation checkpointing
every c layers, minimized at c* = sqrt(L) (the classic sqrt(L)-checkpointing
result). Right panel: total recompute cost is invariant to c -- no matter
how you carve L layers into segments of size c, every layer's forward gets
recomputed exactly once overall, so compute overhead is a flat one extra
full forward pass, independent of the memory-optimal choice of c.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

L = 144  # number of sequential layers (chosen so sqrt(L) = 12 is a clean integer)
c = np.arange(1, L + 1)  # candidate checkpoint intervals

memory = L / c + c          # M(c) = L/c checkpoints stored + c activations recomputed live
c_star = int(round(np.sqrt(L)))  # calculus optimum: dM/dc = -L/c^2 + 1 = 0 -> c = sqrt(L)
m_star = L / c_star + c_star

recompute_layers = np.full_like(c, L, dtype=float)  # (L/c segments) * (c layers/segment) = L, always

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(c, memory, color="#2b8cbe", lw=2, label="M(c) = L/c + c")
ax.scatter([c_star], [m_star], color="#de2d26", zorder=5, s=50,
           label=f"minimum at c*=sqrt(L)={c_star}\n(M={m_star:.0f})")
ax.axvline(c_star, color="#de2d26", ls=":", lw=1, alpha=0.7)
ax.set_xlabel("checkpoint interval c (layers per segment)")
ax.set_ylabel("activation memory (units)")
ax.set_title(f"Memory vs checkpoint interval (L={L} layers)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(c, recompute_layers, color="#31a354", lw=2, label="total recomputed layer-forwards = L (constant)")
ax.axhline(L, color="#31a354", ls=":", lw=1, alpha=0.5)
ax.set_ylim(0, L * 1.5)
ax.set_xlabel("checkpoint interval c (layers per segment)")
ax.set_ylabel("recomputed layer-forwards")
ax.set_title("Recompute cost is flat: +1 full forward pass, any c")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("Day 97: sqrt(L) checkpointing minimizes memory at no extra compute cost", y=1.02)
plt.savefig("graph_DAY_97.png", dpi=120, bbox_inches="tight")
print(f"L={L}: memory-optimal c*={c_star}, M(c*)={m_star:.1f} (vs M(1)={memory[0]:.0f} and M(L)={memory[-1]:.0f})")
