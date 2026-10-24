"""
Day 143 - Consolidation: the normalisation thread (BatchNorm -> LayerNorm -> RMSNorm).
Left : how much a fixed sample's BatchNorm output wobbles depending on who else is in the batch (vs LayerNorm: none).
Right: magnitude spectrum of a feature vector with a big DC offset, raw vs LayerNorm vs RMSNorm.
Synthetic, seed 42, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---- Left: batch-statistics noise ---------------------------------------------------------
pop = rng.normal(2.0, 3.0, size=5000)           # one feature over a "dataset"
x0 = 5.0                                         # the fixed sample we track
sizes = [2, 4, 8, 16, 32, 64]
bn_std = []
for B in sizes:
    outs = []
    for _ in range(300):
        batch = np.append(rng.choice(pop, B - 1), x0)
        outs.append((x0 - batch.mean()) / np.sqrt(batch.var() + 1e-5))
    bn_std.append(np.std(outs))
print("BN output std for the same sample, by batch size:", dict(zip(sizes, np.round(bn_std, 3))))

# ---- Right: spectrum of one token's feature vector ----------------------------------------
d = 64
n = np.arange(d)
v = 4.0 + np.sin(2 * np.pi * 5 * n / d) + 0.5 * rng.standard_normal(d)   # big DC + one tone + noise
ln = (v - v.mean()) / np.sqrt(v.var() + 1e-5)
rms = v / np.sqrt(np.mean(v ** 2) + 1e-5)
spec = lambda s: np.abs(np.fft.rfft(s)) / d
print(f"DC bin: raw={spec(v)[0]:.3f}  LN={spec(ln)[0]:.1e}  RMS={spec(rms)[0]:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(sizes, bn_std, "o-", label="BatchNorm")
ax[0].axhline(0, color="tab:orange", label="LayerNorm / RMSNorm (per-sample)")
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("batch size B"); ax[0].set_ylabel("std of normalised output (same input)")
ax[0].set_title("Batch statistics make outputs depend on neighbours"); ax[0].legend()
k = np.arange(len(spec(v)))
ax[1].plot(k, spec(v), label="raw")
ax[1].plot(k, spec(ln), label="LayerNorm")
ax[1].plot(k, spec(rms), "--", label="RMSNorm")
ax[1].set_xlabel("frequency bin k"); ax[1].set_ylabel("|FFT| / d")
ax[1].set_title("LN removes DC; RMSNorm only rescales"); ax[1].legend()
plt.tight_layout()
plt.savefig("graph_DAY_143.png", dpi=120, bbox_inches="tight")
