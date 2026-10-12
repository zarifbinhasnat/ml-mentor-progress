"""
Day 133 - Consolidation: the noise-as-a-resource thread.

Left : SGD on f(w) = 0.5*h*w^2 with gradient noise of variance s^2/B (minibatch size B).
       The stationary std of w is sqrt(lr*s^2 / (B*h*(2 - lr*h))), so bigger B means a quieter iterate.
Right: forward diffusion on a 1/f-spectrum signal. x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps.
       Signal power falls with frequency, noise power is flat, so high frequencies drown first.
All data synthetic, seed 42, N = 64 samples per signal, 40 signals averaged for the PSD.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- left: SGD noise vs batch size ----
h, s, lr, steps, burn = 1.0, 1.0, 0.1, 400, 100
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for B, c in [(1, "tab:red"), (16, "tab:blue")]:
    w, traj = 3.0, []
    for _ in range(steps):
        g = h * w + s / np.sqrt(B) * np.random.randn()   # true gradient + minibatch noise
        w -= lr * g
        traj.append(w)
    traj = np.array(traj)
    emp = traj[burn:].std()
    pred = np.sqrt(lr * s**2 / (B * h * (2 - lr * h)))
    print(f"B={B:2d}: empirical std {emp:.3f}, predicted {pred:.3f}")
    ax1.plot(traj, c=c, lw=1, label=f"B={B} (std {emp:.2f}, theory {pred:.2f})")
ax1.axhline(0, c="k", lw=0.5)
ax1.set_xlabel("step")
ax1.set_ylabel("w")
ax1.set_title("SGD noise floor shrinks as 1/sqrt(B)")
ax1.legend()

# ---- right: forward diffusion and the spectrum ----
N, R = 64, 40
f = np.fft.rfftfreq(N)[1:]                               # skip DC
x0 = np.zeros((R, N))
for r in range(R):
    amp = np.zeros(N // 2 + 1, dtype=complex)
    ph = np.exp(2j * np.pi * np.random.rand(N // 2))
    amp[1:] = (1.0 / f) * ph                             # amplitude ~ 1/f  ->  power ~ 1/f^2
    x0[r] = np.fft.irfft(amp, n=N)
x0 /= x0.std()                                           # unit variance signal
psd = lambda x: (np.abs(np.fft.rfft(x, axis=1))**2).mean(0)[1:] / N
P0 = psd(x0)
ax2.loglog(f, P0, c="k", lw=2, label="clean signal")
for abar, c in [(0.9, "tab:green"), (0.3, "tab:orange"), (0.05, "tab:red")]:
    xt = np.sqrt(abar) * x0 + np.sqrt(1 - abar) * np.random.randn(R, N)
    Pt = psd(xt)
    noise_floor = 1 - abar                               # flat white-noise PSD of the added noise
    sig = abar * P0
    cross = f[np.argmax(sig < noise_floor)] if (sig < noise_floor).any() else np.nan
    print(f"abar={abar}: signal falls below noise at f = {cross:.3f} cycles/sample")
    ax2.loglog(f, Pt, c=c, lw=1, label=f"abar={abar} (cutoff f={cross:.2f})")
    ax2.axhline(noise_floor, c=c, ls=":", lw=0.8)
ax2.set_xlabel("frequency (cycles/sample)")
ax2.set_ylabel("PSD")
ax2.set_title("Forward noising drowns high frequencies first")
ax2.legend(fontsize=8)
plt.savefig("graph_DAY_133.png", dpi=120, bbox_inches="tight")
