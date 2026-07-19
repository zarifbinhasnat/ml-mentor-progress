"""
Day 52 — Fixed-Context Bottleneck
Two illustrative (synthetic, not measured) panels: (1) the classic qualitative
shape reported for encoder-decoder seq2seq without attention -- quality holds
up on short sequences and degrades on long ones, because everything must fit
through one fixed-size vector -- contrasted with attention's flatter profile;
(2) a quantitative reason why: under the encoder's own recurrence, EARLY
source tokens contribute exponentially less to the final context vector than
LATE ones, by the same a^(T-t) decay mechanism from Day 46/48.
"""
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(8, 7))

# --- Top: illustrative quality-vs-length curves (synthetic, for intuition only) ---
ax = axes[0]
lengths = np.linspace(5, 60, 200)

# fixed-context: holds up while length < encoder's effective "capacity", then decays
capacity = 18
fixed_context_quality = 1.0 / (1 + np.exp((lengths - capacity) / 6))

# attention (previewed for contrast): degrades far more gently
attention_quality = 1.0 / (1 + np.exp((lengths - 45) / 12))

ax.plot(lengths, fixed_context_quality, color="#d62728", label="fixed-context seq2seq (today)")
ax.plot(lengths, attention_quality, color="#2ca02c", linestyle="--",
        label="with attention (previewed, Phase 6)")
ax.set_xlabel("source sequence length (tokens)")
ax.set_ylabel("translation quality (illustrative, arbitrary units)")
ax.set_title("Illustrative shape only -- not measured data -- of the length-vs-quality effect")
ax.legend(fontsize=8)

# --- Bottom: how much of position t's information survives into the final context c ---
ax2 = axes[1]
T = 50
positions = np.arange(1, T + 1)

for a, c, label in [(0.7, "#1f77b4", "a=0.7 (fast recency bias)"),
                     (0.9, "#ff7f0e", "a=0.9 (slower recency bias)")]:
    surviving_influence = a ** (T - positions)  # same a^(T-t) motif as Day 46/48
    ax2.plot(positions, surviving_influence, color=c, label=label)

ax2.axvline(1, color="gray", linestyle=":", linewidth=0.8)
ax2.set_xlabel("source token position t (1 = first word, 50 = last word)")
ax2.set_ylabel("influence surviving into final context c")
ax2.set_title("The encoder's OWN recurrence already discounts early tokens before attention ever helps")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_52.png", dpi=120, bbox_inches="tight")
