"""
Transposed convolution, viewed as a DSP upsampler.

Left panel: zero-insertion (stride-2 "fractional stride") stuffs a zero
between every sample of a low-rate signal. In the frequency domain this
does NOT just stretch the spectrum -- it creates spectral IMAGES (mirrored
copies of the original spectrum) above the original Nyquist band. That's
the exact mechanism a transposed-conv's learned kernel has to clean up.

Right panel: a small 2D "constant" feature map run through a stride-2
transposed convolution with an odd kernel (size 3, doesn't evenly divide
the stride) vs. an even kernel (size 4, does evenly divide the stride).
Uniform overlap = smooth output; uneven overlap = the classic checkerboard
artifact, visible directly as a striped pattern in the output heatmap.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Left panel: 1D zero-insertion upsampling and its spectral images ----
N = 32                              # low-rate signal length
n = np.arange(N)
f1, f2 = 3, 7                       # cycles over N samples
x = np.sin(2 * np.pi * f1 * n / N) + 0.5 * np.sin(2 * np.pi * f2 * n / N)

L = 4                                # upsample factor (stride of the transposed conv)
x_up = np.zeros(N * L)
x_up[::L] = x                        # zero-insertion == what stride does before the "conv" part

X = np.abs(np.fft.rfft(x, n=N * L))               # zero-padded FFT of original, same freq axis
X_up = np.abs(np.fft.rfft(x_up))                   # spectrum AFTER zero-insertion
freqs = np.fft.rfftfreq(N * L, d=1.0) * (N * L)    # bin index axis, matched length

# a short lowpass FIR = stand-in for the learned kernel that must suppress the images
kernel = np.array([0.25, 0.5, 0.25])
x_filtered = np.convolve(x_up, kernel, mode="same")
X_filtered = np.abs(np.fft.rfft(x_filtered))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(freqs, X_up, color="#d62728", lw=1.8, label="after zero-insertion (raw)")
ax.plot(freqs, X_filtered, color="#1f77b4", lw=1.8, label="after learned-kernel filtering")
ax.axvline(N / 2, color="gray", ls="--", lw=1, label="original Nyquist")
ax.set_title("Transposed conv = zero-insert + filter\n(spectral images must be suppressed)")
ax.set_xlabel("frequency bin")
ax.set_ylabel("|X(f)|")
ax.legend(fontsize=8, loc="upper right")

# ---- Right panel: checkerboard artifact, odd vs even kernel/stride overlap ----
def transposed_conv2d_uniform_kernel(x, stride, k):
    """Minimal transposed-conv2d with an all-ones (uniform, untrained) kernel,
    just to expose the overlap pattern stride/kernel geometry produces."""
    H, W = x.shape
    out_h, out_w = (H - 1) * stride + k, (W - 1) * stride + k
    out = np.zeros((out_h, out_w))
    kernel2d = np.ones((k, k))
    for i in range(H):
        for j in range(W):
            out[i * stride: i * stride + k, j * stride: j * stride + k] += x[i, j] * kernel2d
    return out

feat = np.ones((6, 6))               # constant input: isolates the overlap geometry itself
odd_out = transposed_conv2d_uniform_kernel(feat, stride=2, k=3)    # 3 doesn't divide 2 evenly
even_out = transposed_conv2d_uniform_kernel(feat, stride=2, k=4)   # 4 divides 2 evenly

combo = np.concatenate([odd_out, np.full((odd_out.shape[0], 2), np.nan), even_out[:odd_out.shape[0], :]], axis=1)

ax2 = axes[1]
im = ax2.imshow(odd_out, cmap="magma")
ax2.set_title("stride=2, kernel=3 (uneven overlap)\n→ checkerboard stripes")
ax2.set_xlabel("uniform input, untrained kernel:\nstripes are pure geometry, not noise")
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("graph_DAY_44.png", dpi=120, bbox_inches="tight")
print("saved graph_DAY_44.png")
