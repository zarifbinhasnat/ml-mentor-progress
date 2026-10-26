"""
Day 145 - Consolidation: the generative thread (AE -> VAE -> GAN -> diffusion) in the frequency domain.
Left : average power spectrum of 1/f-style "real" signals vs a blurry (MSE/VAE-like) reconstruction
       vs a zero-insertion upsampler (transposed-conv-like) that leaves a spectral replica.
Right: forward diffusion SNR per frequency -- high frequencies drown first (cut-off frequency vs t).
Synthetic, deterministic, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N, B = 64, 40                                     # 64 samples per signal, 40 signals
f = np.fft.rfftfreq(N)                            # cycles/sample, 0 .. 0.5

def make_real():
    amp = 1.0 / np.maximum(np.arange(N // 2 + 1), 1)          # amplitude ~ 1/k  (power ~ 1/k^2)
    ph = np.exp(2j * np.pi * np.random.rand(N // 2 + 1))
    X = amp * ph; X[0] = 0; X[-1] = X[-1].real
    x = np.fft.irfft(X, n=N)
    return x / x.std()                            # unit variance, like normalised pixels

def psd(x):
    return np.mean(np.abs(np.fft.rfft(x, axis=-1)) ** 2, axis=0)

real = np.stack([make_real() for _ in range(B)])
blur = np.stack([np.convolve(x, np.ones(5) / 5, mode="same") for x in real])   # mean-seeking decoder ~ low-pass

def upsample(x):                                   # zero-insertion x2 then 3-tap kernel, like a transposed conv
    z = np.zeros(N); z[::2] = x[::2] * 2
    return np.convolve(z, [0.25, 0.5, 0.25], mode="same")

fake = np.stack([upsample(x) for x in real])

# forward diffusion: x_t = sqrt(ab) x0 + sqrt(1-ab) eps ; per-frequency SNR_k = ab*S_k / ((1-ab)*N)
S = psd(real)                                      # |FFT|^2 of white noise with var 1 is N per bin
ab = np.linspace(0.999, 0.001, 200)
snr = ab[:, None] * S[None, :] / ((1 - ab)[:, None] * N)
cut = np.array([f[1:][np.where(row >= 1)[0].max()] if (row >= 1).any() else 0.0 for row in snr[:, 1:]])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(f[1:], psd(real)[1:], label="real (1/f$^2$)")
ax[0].semilogy(f[1:], psd(blur)[1:], label="blurry decoder (AE/VAE-like)")
ax[0].semilogy(f[1:], psd(fake)[1:], label="zero-insert upsampler (GAN-like)")
ax[0].set_xlabel("frequency (cycles/sample)"); ax[0].set_ylabel("power"); ax[0].legend(fontsize=8)
ax[0].set_title("Spectral fingerprints")
ax[1].plot(1 - ab, cut)
ax[1].set_xlabel("noise level  1 - $\\bar\\alpha_t$"); ax[1].set_ylabel("highest freq with SNR >= 1")
ax[1].set_title("Diffusion: high frequencies drown first")
plt.savefig("graph_DAY_145.png", dpi=120, bbox_inches="tight")

k = np.argmin(np.abs(f - 0.4))
print("real/blur power ratio at f=0.4: %.1fx" % (psd(real)[k] / psd(blur)[k]))
print("upsampler/real power at f=0.4: %.2f / %.2f" % (psd(fake)[k], psd(real)[k]))
for a in (0.9, 0.5, 0.1, 0.01):
    i = np.argmin(np.abs(ab - a)); print("abar=%.2f  SNR>=1 up to f=%.3f" % (a, cut[i]))
