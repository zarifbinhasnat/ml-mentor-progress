import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
# Stride s (in 4-byte words): a warp's 32 loads touch distinct 32-byte sectors.
strides = np.array([1, 2, 3, 4, 6, 8, 16, 32])
eff = []
for s in strides:
    sectors = len({(s * i) // 8 for i in range(32)})   # 4*s*i bytes / 32 B
    eff.append(128.0 / (32 * sectors) * 100)
eff = np.array(eff)

# Bank conflict degree for column read of a [32][W] float tile.
widths = np.arange(32, 40)
deg = []
for W in widths:
    banks = [(i * W) % 32 for i in range(32)]
    deg.append(max(banks.count(b) for b in set(banks)))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar([str(s) for s in strides], eff, color="#3b6fb6")
ax[0].set_xlabel("stride between neighbouring threads (4-byte words)")
ax[0].set_ylabel("useful bytes per byte fetched (%)")
ax[0].set_title("Coalescing: global-memory efficiency")
ax[1].bar([str(w) for w in widths], deg, color="#c0503a")
ax[1].set_xlabel("tile row width W (floats), tile[32][W]")
ax[1].set_ylabel("k-way bank conflict on column read")
ax[1].set_title("Shared-memory padding (W=33 fixes it)")
plt.savefig("graph_DAY_160.png", dpi=120, bbox_inches="tight")
