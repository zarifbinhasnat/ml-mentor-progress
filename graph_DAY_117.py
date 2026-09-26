import numpy as np
import matplotlib.pyplot as plt

# Signal propagation through a deep ReLU net under three init scales.
# Same mechanism drives both forward activation blowup/collapse and the
# backward gradient product from Day 116's chain-rule connection -- the
# weights multiply the signal at every layer either way.

N_LAYERS = 30
WIDTH = 256
N_SAMPLES = 64


def forward_track_std(std_fn, seed=42):
    rng = np.random.default_rng(seed)  # np.random.default_rng: modern NumPy RNG, seeded for reproducibility
    x = rng.normal(size=(N_SAMPLES, WIDTH))
    stds = [x.std()]
    for _ in range(N_LAYERS):
        W = rng.normal(0.0, std_fn(WIDTH), size=(WIDTH, WIDTH))
        x = np.maximum(0.0, x @ W)  # ReLU; @ is matrix multiply
        stds.append(x.std())
    return stds


he_scale = lambda n: np.sqrt(2.0 / n)          # He init: variance-preserving for ReLU
too_small_scale = lambda n: 0.1 * he_scale(n)  # 10x below He -> signal collapses
too_large_scale = lambda n: 4.0 * he_scale(n)  # 4x above He -> signal blows up

stds_small = forward_track_std(too_small_scale)
stds_he = forward_track_std(he_scale)
stds_large = forward_track_std(too_large_scale)

layers = np.arange(N_LAYERS + 1)

plt.figure(figsize=(7, 4.5))
plt.semilogy(layers, stds_small, marker="o", label="0.1x He std (vanishing)")
plt.semilogy(layers, stds_he, marker="o", label="He init (stable)")
plt.semilogy(layers, stds_large, marker="o", label="4x He std (exploding)")
plt.xlabel("Layer depth")
plt.ylabel("Activation std, log scale")
plt.title("Signal propagation through a 30-layer ReLU net vs. init scale")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("graph_DAY_117.png", dpi=120, bbox_inches="tight")
