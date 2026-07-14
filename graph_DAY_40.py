"""
Day 40 -- VGG
Left:  stacking n 3x3 convs vs one big (2n+1)x(2n+1) conv -- receptive field
       grows identically (linear in n) but parameter count grows quadratically
       for the single big kernel, linearly for the stack.
Right: DSP analogue of the same idea -- cascading two 3-tap box filters gives
       a 5-tap triangular filter with far better stopband attenuation (less
       spectral leakage) than one 5-tap box filter of the same footprint.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # deterministic synthetic data

# ---------------------------------------------------------------------------
# Left panel: receptive field parity, parameter cost divergence
# ---------------------------------------------------------------------------
n_layers = np.array([1, 2, 3, 4])
k = 3
receptive_field = n_layers * (k - 1) + 1          # RF of n stacked 3x3 convs
stacked_params = n_layers * (k * k)                # per (in_ch, out_ch) pair
single_big_params = receptive_field ** 2           # one conv of size RF x RF

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(n_layers, stacked_params, "o-", color="#1f77b4", label="n stacked 3x3 convs")
ax.plot(n_layers, single_big_params, "s-", color="#d62728", label="one (2n+1)x(2n+1) conv")
for x, rf in zip(n_layers, receptive_field):
    ax.annotate(f"RF={rf}", (x, stacked_params[x - 1]), textcoords="offset points",
                xytext=(-4, 10), ha="center", fontsize=7.5, color="#1f77b4")
ax.set_xlabel("number of stacked 3x3 layers (n)")
ax.set_ylabel("weights per (in_ch, out_ch) pair")
ax.set_title("Same receptive field, divergent parameter cost")
ax.set_xticks(n_layers)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# ---------------------------------------------------------------------------
# Right panel: cascading small filters vs one big filter (frequency domain)
# ---------------------------------------------------------------------------
box3 = np.ones(3) / 3.0                 # single 3-tap box filter
cascade5 = np.convolve(box3, box3)      # two box3's cascaded -> 5-tap triangle
box5 = np.ones(5) / 5.0                 # one 5-tap box filter, same footprint

n_fft = 512
freqs = np.fft.rfftfreq(n_fft, d=1.0)   # normalized frequency, 0..0.5

def mag_db(h):
    H = np.fft.rfft(h, n=n_fft)
    return 20 * np.log10(np.maximum(np.abs(H), 1e-6))

ax = axes[1]
ax.plot(freqs, mag_db(box5), color="#d62728", label="one 5-tap box filter")
ax.plot(freqs, mag_db(cascade5), color="#1f77b4", label="two cascaded 3-tap box filters")
ax.set_xlabel("normalized frequency")
ax.set_ylabel("magnitude (dB)")
ax.set_title("Cascading small filters = better sidelobe rejection")
ax.set_ylim(-60, 5)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_40.png", dpi=120, bbox_inches="tight")
