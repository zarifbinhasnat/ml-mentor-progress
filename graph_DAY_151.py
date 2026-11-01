"""
Day 151 - Consolidation: generative-modelling thread, what each family leaves in the spectrum.
Left : radially averaged PSD of a synthetic 1/f "real" image vs VAE-like (blur), GAN-like (zero-insert upsample) outputs.
Right: diffusion forward noising -- signal PSD vs white-noise PSD; the crossover frequency moves inward as t grows.
Synthetic, seed 42, numpy + matplotlib only.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 64
fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")
r = np.sqrt(fx ** 2 + fy ** 2)
r[0, 0] = 1.0
amp = 1.0 / r                                   # amplitude ~ 1/f  (power ~ 1/f^2)
amp[0, 0] = 0.0
img = np.real(np.fft.ifft2(amp * np.fft.fft2(np.random.randn(N, N))))
img /= img.std()

def radial_psd(x):
    P = np.abs(np.fft.fft2(x - x.mean())) ** 2
    rr = np.sqrt(fx ** 2 + fy ** 2)
    bins = np.linspace(0, 0.5, 17)
    idx = np.digitize(rr.ravel(), bins) - 1
    out = np.array([P.ravel()[idx == i].mean() for i in range(16)])
    return 0.5 * (bins[:-1] + bins[1:]), out / N ** 2

# VAE-like: Gaussian low-pass (averaging posterior samples blurs)
vae = np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-(r ** 2) / (2 * 0.12 ** 2))))
# GAN-like: low-res 32x32 -> zero-insertion upsample, then a short smoothing kernel (transposed-conv checkerboard)
small = img[::2, ::2]
up = np.zeros((N, N)); up[::2, ::2] = small
k = np.array([0.25, 0.5, 0.25]); 
gan = sum(w * np.roll(up, s, axis=0) for w, s in zip(k, (-1, 0, 1)))
gan = sum(w * np.roll(gan, s, axis=1) for w, s in zip(k, (-1, 0, 1)))
gan *= 2

f, p_real = radial_psd(img)
_, p_vae = radial_psd(vae)
_, p_gan = radial_psd(gan)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(f, p_real, "k-o", ms=3, label="real (1/f)")
ax[0].semilogy(f, p_vae + 1e-12, "C0-o", ms=3, label="VAE-like (blur)")
ax[0].semilogy(f, p_gan + 1e-12, "C3-o", ms=3, label="GAN-like (zero-insert up)")
ax[0].set_xlabel("radial frequency (cycles/pixel)"); ax[0].set_ylabel("PSD")
ax[0].set_title("What each family leaves in the spectrum"); ax[0].legend(fontsize=8)

for (name, ab), c in zip({"t small": 0.9, "t mid": 0.3, "t large": 0.02}.items(), ("C2", "C1", "C3")):
    ax[1].semilogy(f, ab * p_real, color=c, label=f"signal, {name} (alpha_bar={ab})")
    ax[1].semilogy(f, np.full_like(f, 1 - ab), color=c, ls="--")   # white-noise PSD = 1-alpha_bar
ax[1].set_xlabel("radial frequency (cycles/pixel)"); ax[1].set_ylabel("PSD")
ax[1].set_title("Forward noising: high freqs sink below noise first"); ax[1].legend(fontsize=7)
plt.savefig("graph_DAY_151.png", dpi=120, bbox_inches="tight")
