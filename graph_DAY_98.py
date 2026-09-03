"""
Day 98 -- Data / model / pipeline parallelism

Pipeline-parallel bubble fraction as a function of micro-batch count m,
for a few pipeline depths p (number of stages / GPUs in the pipeline).

Exact bubble fraction = (p-1) / (m + p - 1)   (fill+drain slots over total slots)
Common large-m approximation quoted in papers = (p-1) / m

Both are plotted to show they converge as m grows -- i.e. more micro-batches
(finer-grained pipelining) shrinks the idle "bubble" time toward zero,
independent of how many stages p you split the model into.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

m = np.arange(1, 65)  # number of micro-batches per pipeline flush
stage_counts = [2, 4, 8, 16]
colors = ["#2b8cbe", "#31a354", "#de2d26", "#756bb1"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
for p, color in zip(stage_counts, colors):
    exact = (p - 1) / (m + p - 1)
    ax.plot(m, exact, color=color, lw=2, label=f"p={p} stages")
ax.axhline(0.1, color="gray", ls=":", lw=1, alpha=0.6)
ax.text(45, 0.11, "10% bubble", fontsize=8, color="gray")
ax.set_xlabel("micro-batches per flush (m)")
ax.set_ylabel("bubble fraction = (p-1)/(m+p-1)")
ax.set_title("Pipeline idle time shrinks as m grows")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
p = 8
exact = (p - 1) / (m + p - 1)
approx = (p - 1) / m
ax.plot(m, exact, color="#de2d26", lw=2, label="exact: (p-1)/(m+p-1)")
ax.plot(m, approx, color="#de2d26", lw=2, ls="--", alpha=0.6, label="approx: (p-1)/m  (m >> p)")
ax.set_xlabel("micro-batches per flush (m)")
ax.set_ylabel("bubble fraction")
ax.set_title(f"Exact vs. textbook approximation (p={p})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle("Day 98: more micro-batches -> smaller pipeline bubble, any stage count p", y=1.02)
plt.savefig("graph_DAY_98.png", dpi=120, bbox_inches="tight")

for p in stage_counts:
    frac_at_8 = (p - 1) / (8 + p - 1)
    print(f"p={p}: bubble fraction at m=8 is {frac_at_8:.3f}")
