"""
Day 115 -- Consolidation: Phase 5 (RNN sequence modeling)
Connection 3: the fixed-context bottleneck (Concept 52) degrades with source
length regardless of encoder quality, while an attention-style decoder (Phase 6)
does not -- this is the exact gap that motivates attention.

Synthetic data only (seed 42, < 100 points). No downloads, no internet.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

source_lengths = np.arange(5, 61, 5)  # 12 points: 5, 10, ..., 60 tokens

# Fixed-context vector (vanilla seq2seq, Concept 51-52): a single fixed-size
# vector must encode the WHOLE sentence, so information loss grows with length.
# Modeled as a mild superlinear growth in reconstruction error + noise.
fixed_context_error = 0.015 * source_lengths + 0.0006 * source_lengths**2
fixed_context_error += np.random.normal(0, 0.15, size=source_lengths.shape)
fixed_context_error = np.clip(fixed_context_error, 0, None)

# Attention-based decoder (Phase 6 preview): decoder re-reads every encoder
# state at every step, so no single vector ever has to hold the whole sentence.
# Error stays flat (small, length-independent) up to noise.
attention_error = np.full_like(source_lengths, 0.9, dtype=float)
attention_error += np.random.normal(0, 0.15, size=source_lengths.shape)
attention_error = np.clip(attention_error, 0, None)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(source_lengths, fixed_context_error, "o-", color="#d62728",
        label="Fixed context vector (Concept 51-52)")
ax.plot(source_lengths, attention_error, "s-", color="#1f77b4",
        label="Attention over encoder states (Phase 6)")

ax.set_xlabel("Source sentence length (tokens)")
ax.set_ylabel("Decode error (synthetic, arb. units)")
ax.set_title("Fixed-context bottleneck vs. attention as source length grows")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

fig.savefig("graph_DAY_115.png", dpi=120, bbox_inches="tight")
print("saved graph_DAY_115.png")
