"""
Day 134 - Consolidation: the weight-sharing thread.

Left : a circular 1-D convolution is a dense matrix with shared diagonals (circulant). Shifting the input
       and then applying it equals applying it and then shifting; a generic dense matrix with the same
       number of "free" outputs breaks this. Bars show the equivariance error ||f(shift x) - shift f(x)||.
Right: the eigenvalues of the circulant matrix are the DFT of the kernel, so |FFT(kernel)| is the filter's
       frequency response. A 3-tap smoothing kernel is a low-pass; a 3-tap difference kernel is a high-pass.
All data synthetic, seed 42, N = 32 samples.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 32
x = np.random.randn(N)

# ---- left: equivariance of circulant (shared) vs generic dense weights ----
k = np.zeros(N)
k[[0, 1, N - 1]] = [0.5, 0.25, 0.25]                       # 3-tap smoothing kernel, circular
C = np.stack([np.roll(k, i) for i in range(N)])            # circulant matrix: row i is k shifted by i
D = np.random.randn(N, N) * 0.3                            # generic dense matrix, N*N free parameters
shift = lambda v, s: np.roll(v, s)
errs = {}
for name, W in [("circulant\n(3 params)", C), ("dense\n(1024 params)", D)]:
    errs[name] = np.linalg.norm(W @ shift(x, 5) - shift(W @ x, 5))
    print(f"{name.splitlines()[0]:10s} equivariance error = {errs[name]:.2e}")

# check: circulant action equals FFT multiplication
y_mat = C @ x
y_fft = np.fft.ifft(np.fft.fft(x) * np.fft.fft(k, N)).real
print(f"max |C@x - ifft(fft(x)*fft(k))| = {np.abs(y_mat - y_fft).max():.2e}")
eig = np.sort_complex(np.linalg.eigvals(C))
print(f"eigenvalues of C match fft(k): {np.allclose(np.sort(np.abs(eig)), np.sort(np.abs(np.fft.fft(k, N))))}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.bar(list(errs.keys()), [max(v, 1e-17) for v in errs.values()], color=["tab:blue", "tab:red"])
ax1.set_yscale("log")
ax1.set_ylabel("||W(shift x) - shift(W x)||")
ax1.set_title("Weight sharing buys shift-equivariance")

# ---- right: frequency response = DFT of the kernel ----
freqs = np.fft.rfftfreq(N)
kd = np.zeros(N)
kd[[0, 1, N - 1]] = [0.0, 0.5, -0.5]                        # central difference kernel
ax2.plot(freqs, np.abs(np.fft.rfft(k)), c="tab:blue", lw=2, label="smoothing [.25 .5 .25]")
ax2.plot(freqs, np.abs(np.fft.rfft(kd)), c="tab:orange", lw=2, label="difference [.5 0 -.5]")
ax2.set_xlabel("frequency (cycles/sample)")
ax2.set_ylabel("|H(f)|")
ax2.set_title("Eigenvalues of the shared-weight matrix = DFT of kernel")
ax2.legend()
plt.savefig("graph_DAY_134.png", dpi=120, bbox_inches="tight")
