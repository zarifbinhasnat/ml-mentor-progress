import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Synthetic "teacher" logits for a 10-class problem: one dominant class,
# a few plausible runners-up, and a long tail of near-zero logits —
# a realistic shape for a trained classifier's raw output.
logits = np.array([4.8, 2.1, 1.6, 0.9, 0.3, -0.2, -0.6, -1.0, -1.4, -2.0])
classes = np.arange(len(logits))


def softmax_with_temperature(z, T):
    z = z / T
    z = z - z.max()  # numerical stability: subtract max before exponentiating
    e = np.exp(z)
    return e / e.sum()


def spectral_flatness(p):
    # Wiener entropy / spectral flatness measure, borrowed straight from
    # audio spectral analysis: geometric_mean / arithmetic_mean of a
    # nonnegative "spectrum". Here the spectrum is the softmax distribution
    # itself. 0 = all energy in one bin (peaky), 1 = perfectly flat (white).
    eps = 1e-12
    p = p + eps
    gmean = np.exp(np.mean(np.log(p)))
    amean = np.mean(p)
    return gmean / amean


temperatures = [1, 2, 4, 8]
distributions = [softmax_with_temperature(logits, T) for T in temperatures]
flatness = [spectral_flatness(p) for p in distributions]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

width = 0.2
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for i, (T, p, c) in enumerate(zip(temperatures, distributions, colors)):
    offset = (i - 1.5) * width
    axes[0].bar(classes + offset, p, width=width, label=f"T={T}", color=c)

axes[0].set_xlabel("class index")
axes[0].set_ylabel("softmax probability")
axes[0].set_title("Teacher softmax at increasing temperature")
axes[0].set_xticks(classes)
axes[0].legend()

axes[1].plot(temperatures, flatness, "o-", color="#9467bd")
axes[1].set_xlabel("temperature T")
axes[1].set_ylabel("spectral flatness (0=peaky, 1=flat)")
axes[1].set_title("Distribution \"flatness\" vs. temperature")
axes[1].set_ylim(0, 1)
axes[1].grid(alpha=0.3)

fig.suptitle("Knowledge distillation: temperature softens the teacher's distribution")
fig.tight_layout()
plt.savefig("graph_DAY_75.png", dpi=120, bbox_inches="tight")
