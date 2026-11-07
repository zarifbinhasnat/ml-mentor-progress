import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
# Idealised model: SM has 65536 32-bit registers and max 2048 resident threads.
regs = np.arange(16, 257, 8)
threads = np.minimum(2048, (65536 // regs) // 32 * 32)
occ = 100 * threads / 2048

# Warp lane utilisation when lane i loops (i+1) times (data-dependent trip count)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].step(regs, occ, where="post", color="tab:blue")
ax[0].set_xlabel("registers per thread")
ax[0].set_ylabel("resident threads / 2048 (%)")
ax[0].set_title("Idealised occupancy vs register use (65536 regs/SM)")
ax[0].grid(alpha=0.3)

it = np.arange(1, 33)
active_lanes = 33 - it
ax[1].bar(it, active_lanes, color="tab:orange")
ax[1].axhline(32, color="gray", ls="--")
ax[1].set_xlabel("loop iteration (warp issues all 32)")
ax[1].set_ylabel("active lanes")
ax[1].set_title("Divergent loop: lane i runs i+1 iterations\nuseful 528 / issued 1024 = 51.6%")
ax[1].grid(alpha=0.3)
plt.savefig("graph_DAY_159.png", dpi=120, bbox_inches="tight")
