import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- Synthetic Zipfian corpus: word rank r has frequency proportional to 1/r ----
n_words = 120
ranks = np.arange(1, n_words + 1)
freqs = 1.0 / ranks          # Zipf's law, s = 1
freqs = freqs / freqs.sum()  # normalize to a probability mass

# ---- Word-level tokenization: cumulative coverage as vocab (top-k words) grows ----
cum_coverage = np.cumsum(freqs)

# ---- Subword tokenization: any word decomposes into a small, fixed inventory of ----
# ---- characters/subword pieces, so coverage is ~100% at a tiny, corpus-size-independent vocab ----
subword_vocab_size = 20
subword_coverage = 0.999

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: Zipf's law itself (log-log) -- the long tail that breaks word-level vocabs
ax[0].loglog(ranks, freqs, color="#2b6cb0", marker="o", markersize=3, linewidth=1.2)
ax[0].set_xlabel("word rank (log scale)")
ax[0].set_ylabel("relative frequency (log scale)")
ax[0].set_title("Token frequency follows Zipf's law")
ax[0].grid(True, which="both", alpha=0.3)

# Right: coverage vs vocab size -- word-level needs huge vocab to climb the tail;
# subword tokenization gets near-100% coverage at a small, fixed vocab size
ax[1].plot(ranks, cum_coverage, color="#2b6cb0", linewidth=2, label="word-level vocab (top-k words)")
ax[1].axhline(subword_coverage, color="#c05621", linestyle="--", linewidth=1.5,
              label=f"subword tokenizer: ~100% coverage")
ax[1].axvline(subword_vocab_size, color="#c05621", linestyle=":", linewidth=1.2,
              label=f"...at only {subword_vocab_size} vocab entries")
ax[1].set_xlabel("vocabulary size")
ax[1].set_ylabel("fraction of corpus covered")
ax[1].set_title("Coverage vs. vocab size")
ax[1].set_ylim(0, 1.05)
ax[1].legend(fontsize=8, loc="lower right")
ax[1].grid(True, alpha=0.3)

fig.suptitle("Why subwords beat whole words: Zipf's long tail vs. fixed small vocab", fontsize=11)
fig.tight_layout()
plt.savefig("graph_DAY_65.png", dpi=120, bbox_inches="tight")
