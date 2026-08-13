"""
Day 77 — Scaling laws.
Left panel: Chinchilla-style loss-vs-parameters power law + log-log fit.
Right panel: 1/f^alpha noise PSD slope estimation (Signal Lab tie-in) —
same log-log linear-regression machinery, different domain.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- Left panel: L(N) = E + A / N^alpha ----------
N = np.logspace(6, 10, 40)          # 1e6 .. 1e10 parameters
E, A, alpha_true = 1.5, 400.0, 0.35
noise = np.random.normal(0, 0.03, size=N.shape)
L = E + A / N ** alpha_true + noise
L = np.clip(L, 1e-3, None)

log_N = np.log(N)
log_resid = np.log(L - E)                       # assume E (irreducible loss) is known/estimated
alpha_fit, log_A_fit = np.polyfit(log_N, log_resid, 1)
alpha_fit = -alpha_fit

# ---------- Right panel: synthetic 1/f^alpha noise, recover alpha from PSD slope ----------
n_samples = 4096
alpha_psd = 1.8
white = np.random.randn(n_samples)
spectrum = np.fft.rfft(white)
freqs_full = np.fft.rfftfreq(n_samples, d=1.0)
freqs = freqs_full[1:]                           # drop DC bin (f=0 -> division blows up)
shaped = spectrum[1:] / (freqs ** (alpha_psd / 2))   # amplitude ~ f^-(alpha/2) => power ~ f^-alpha
signal_1f = np.fft.irfft(np.concatenate([[0], shaped]), n=n_samples)

psd = np.abs(np.fft.rfft(signal_1f)) ** 2
psd = psd[1:]
log_f = np.log(freqs)
log_psd = np.log(psd + 1e-12)
slope_psd, intercept_psd = np.polyfit(log_f, log_psd, 1)

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].loglog(N, L - E, 'o', ms=4, alpha=0.6, label='observed (L - E)')
axes[0].loglog(N, np.exp(log_A_fit) * N ** (-alpha_fit), 'r-', lw=2,
               label=f'fit: alpha={alpha_fit:.2f} (true {alpha_true})')
axes[0].set_xlabel('Model parameters N')
axes[0].set_ylabel('Excess loss (L - E)')
axes[0].set_title('Scaling law: loss vs N')
axes[0].legend(fontsize=8)

axes[1].loglog(freqs, psd, '.', ms=3, alpha=0.4, label='periodogram')
axes[1].loglog(freqs, np.exp(intercept_psd) * freqs ** slope_psd, 'r-', lw=2,
               label=f'fit slope={-slope_psd:.2f} (true {alpha_psd})')
axes[1].set_xlabel('Frequency f')
axes[1].set_ylabel('PSD')
axes[1].set_title('1/f^alpha noise: PSD slope')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_77.png", dpi=120, bbox_inches="tight")
