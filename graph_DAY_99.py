"""
Day 99 -- ZeRO & sharded training
Per-GPU memory footprint (in units of Psi = param count) for the ZeRO stages,
vs. plain data-parallel replication, as a function of world size N.

Formulas (mixed-precision Adam, standard ZeRO paper accounting):
  fp16 params  = 2*Psi
  fp16 grads   = 2*Psi
  fp32 optim   = 12*Psi   (fp32 master weights 4*Psi + momentum 4*Psi + variance 4*Psi)
  total        = 16*Psi   <- classic "16*Psi bytes per parameter" baseline

  Baseline DP : every GPU holds all 16*Psi, unsharded.
  ZeRO-1      : shard optimizer states only -> 2*Psi + 2*Psi + 12*Psi/N
  ZeRO-2      : also shard gradients        -> 2*Psi + (2*Psi + 12*Psi)/N
  ZeRO-3      : also shard params           -> 16*Psi/N
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N = np.array([1, 2, 4, 8, 16, 32, 64])  # world size (number of GPUs)

baseline = np.full_like(N, 16, dtype=float)
zero1 = 2 + 2 + 12 / N
zero2 = 2 + (2 + 12) / N
zero3 = 16 / N

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(N, baseline, marker="o", label="Baseline DP (no sharding)", color="#888888", linestyle="--")
ax.plot(N, zero1, marker="o", label="ZeRO-1 (shard optimizer states)", color="#4C72B0")
ax.plot(N, zero2, marker="o", label="ZeRO-2 (+ shard gradients)", color="#DD8452")
ax.plot(N, zero3, marker="o", label="ZeRO-3 (+ shard parameters)", color="#55A868")

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks(N)
ax.set_xticklabels(N)
ax.set_xlabel(r"World size $N$ (number of GPUs)")
ax.set_ylabel(r"Per-GPU memory (units of $\Psi$, i.e. multiples of param count)")
ax.set_title(r"ZeRO stages: per-GPU memory vs. $N$ (mixed-precision Adam, $16\Psi$ baseline)")
ax.legend(frameon=False)
ax.grid(True, which="both", alpha=0.3)

plt.savefig("graph_DAY_99.png", dpi=120, bbox_inches="tight")
