"""
Day 137 - Consolidation: the recompute-vs-store thread (BPTT truncation -> FlashAttention -> grad checkpointing -> KV cache).
Left : activation memory of an L-layer net when you keep only every k-th activation (checkpointing):
       memory(k) = L/k + k, minimised at k = sqrt(L); extra cost = one more forward pass.
Right: streaming FIR filtering. Recompute-everything does O(n*M) work per new sample's history;
       a cached state of the last M-1 samples (the signal-processing "KV cache") does O(M) per sample.
Synthetic, seed 42, 64 samples, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Left: checkpointing memory curve -------------------------------------------------
L = 64
ks = np.arange(1, L + 1)
mem = L / ks + ks
k_opt = int(np.sqrt(L))
print(f"L={L}: store-all memory={L}, k=1 memory={mem[0]:.0f}, best k={ks[np.argmin(mem)]} "
      f"(sqrt(L)={k_opt}) memory={mem.min():.0f} -> {L / mem.min():.1f}x smaller, +1 forward pass")

# ---- Right: streaming FIR, recompute vs cached state ---------------------------------
N, M = 64, 16
x = np.sin(2 * np.pi * 5 * np.arange(N) / N) + 0.3 * np.random.randn(N)
h = np.hanning(M) / np.hanning(M).sum()

y_full = np.convolve(x, h)[:N]

# recompute: for each new sample, redo the dot product over the whole history so far
ops_recompute = np.zeros(N)
y_re = np.zeros(N)
for t in range(N):
    acc = 0.0
    for j in range(t + 1):                       # touches every past sample, even those with h = 0
        if t - j < M:
            acc += h[t - j] * x[j]
        ops_recompute[t] += 1
    y_re[t] = acc

# cached: keep only the last M samples, O(M) per step
state = np.zeros(M)
ops_cached = np.zeros(N)
y_ca = np.zeros(N)
for t in range(N):
    state = np.roll(state, 1)
    state[0] = x[t]
    y_ca[t] = h @ state
    ops_cached[t] = M

assert np.allclose(y_re, y_full) and np.allclose(y_ca, y_full)
print(f"N={N}, M={M}: total ops recompute={int(ops_recompute.sum())}, cached={int(ops_cached.sum())} "
      f"-> {ops_recompute.sum() / ops_cached.sum():.2f}x fewer; outputs identical")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
ax1.plot(ks, mem, "tab:blue", lw=2, label="memory(k) = L/k + k")
ax1.axhline(L, color="k", ls="--", lw=1, label=f"store everything = {L}")
ax1.plot([k_opt], [mem[k_opt - 1]], "ro", label=f"k = sqrt(L) = {k_opt}, memory {mem[k_opt - 1]:.0f}")
ax1.set_xlabel("checkpoint spacing k (layers)")
ax1.set_ylabel("activations held (layer units)")
ax1.set_title("Gradient checkpointing: memory vs spacing")
ax1.legend(fontsize=8)

ax2.plot(np.arange(N), ops_recompute, "tab:red", lw=2, label="recompute from history")
ax2.plot(np.arange(N), ops_cached, "tab:green", lw=2, label=f"cached state (M={M})")
ax2.set_xlabel("time step t")
ax2.set_ylabel("multiply-adds visited at step t")
ax2.set_title("Streaming FIR: why a cache wins")
ax2.legend(fontsize=8)

plt.savefig("graph_DAY_137.png", dpi=120, bbox_inches="tight")
