import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- Build a toy 1D "image row" with both structure and fine texture ----------
N = 128
t = np.arange(N)

edge = 1.5 * np.tanh((t - 64) / 8.0)                    # a smooth low-frequency edge (like an object boundary)
mid_texture = 0.5 * np.sin(2 * np.pi * t * 0.12)         # mid-frequency texture
high_texture = 0.25 * np.sin(2 * np.pi * t * 0.40)       # near-Nyquist high-frequency texture (fine detail)
noise = 0.05 * np.random.randn(N)
signal = edge + mid_texture + high_texture + noise

# ---------- Patchify = non-overlapping box-average (the simplest view of what a
# strided-conv patch embedding does to local intensity before any learned weights
# are even applied: it collapses each P-pixel patch to one summary value) ----------
def patchify_boxcar(x, P):
    tokens = x.reshape(-1, P).mean(axis=1)
    reconstruction = np.repeat(tokens, P)   # "what the token remembers", spread back over its patch
    return tokens, reconstruction

P_small, P_large = 16, 32
_, recon_small = patchify_boxcar(signal, P_small)
_, recon_large = patchify_boxcar(signal, P_large)

# ---------- Frequency response of a boxcar (patch-average) filter of length P ----------
def boxcar_response(P, n_fft=4096):
    kernel = np.ones(P) / P
    H = np.fft.rfft(kernel, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0)   # cycles/sample, 0 to 0.5 (Nyquist)
    return freqs, np.abs(H)

freqs8, H8 = boxcar_response(8)
freqs16, H16 = boxcar_response(16)
freqs32, H32 = boxcar_response(32)

# ---------- FFT magnitude spectra: original vs. patch-reconstructed signals ----------
def mag_spectrum(x):
    F = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    return freqs, np.abs(F)

f_orig, S_orig = mag_spectrum(signal)
f_small, S_small = mag_spectrum(recon_small)
f_large, S_large = mag_spectrum(recon_large)

# ---------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

ax = axes[0]
ax.plot(t, signal, color="black", linewidth=1, label="original row")
ax.plot(t, recon_small, color="steelblue", linewidth=1.5, label=f"patch={P_small} reconstruction")
for x in range(0, N, P_small):
    ax.axvline(x, color="steelblue", alpha=0.15, linewidth=0.8)
ax.set_title("signal vs. patch-16 tokens (spatial domain)")
ax.set_xlabel("pixel position")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(freqs8, H8, label="patch=8 filter", color="seagreen")
ax.plot(freqs16, H16, label="patch=16 filter", color="steelblue")
ax.plot(freqs32, H32, label="patch=32 filter", color="firebrick")
ax.set_title("frequency response of patch-average filter")
ax.set_xlabel("normalized frequency (cycles/pixel)")
ax.set_ylabel("|H(f)|")
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(f_orig, S_orig, color="black", linewidth=1, label="original spectrum")
ax.plot(f_small, S_small, color="steelblue", linewidth=1.5, linestyle="--", label="patch=16 spectrum")
ax.plot(f_large, S_large, color="firebrick", linewidth=1.5, linestyle="--", label="patch=32 spectrum")
ax.set_title("what patchify keeps vs. discards")
ax.set_xlabel("normalized frequency (cycles/pixel)")
ax.set_ylabel("|FFT|")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_89.png", dpi=120, bbox_inches="tight")
