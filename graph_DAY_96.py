"""
Day 96 -- Mixed precision (fp16/bf16)

Simulates a wide dynamic range of gradient magnitudes (as seen deep in a
real network) and shows what happens when each value is rounded down into
fp16 vs bf16. fp16 has a narrow exponent range -> tiny gradients underflow
to exactly zero. bf16 keeps fp32's exponent range -> nothing flushes to
zero, it just loses mantissa precision (visible as banding/noise on the
log-log line).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Synthetic gradient magnitudes spanning a realistic dynamic range:
# from tiny (deep layers, late training) up to O(1) (early layers).
true_grad = np.logspace(-8, 1, 80).astype(np.float32)  # fp32 "ground truth"


def to_fp16(x):
    # Real fp16 downcast: narrow 5-bit exponent -> values below ~6e-8
    # (smallest subnormal) hard-flush to zero.
    return x.astype(np.float16).astype(np.float32)


def to_bf16(x):
    # bf16 = fp32's 8-bit exponent + only 7 mantissa bits.
    # Simulate by truncating the low 16 bits of the fp32 bit pattern.
    bits = x.view(np.uint32)
    bits = bits & np.uint32(0xFFFF0000)  # in-place mantissa truncation, no C++ equivalent bit-hack needed here
    return bits.view(np.float32)


fp16_grad = to_fp16(true_grad)
bf16_grad = to_bf16(true_grad)

fp16_flushed = fp16_grad == 0.0
n_flushed = int(fp16_flushed.sum())

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(true_grad, true_grad, "k--", lw=1.5, label="fp32 (exact)")
ax.loglog(true_grad, bf16_grad, "o-", color="#2b8cbe", ms=4, label="bf16 (truncated mantissa, full range)")

# fp16 hits exact zero for small magnitudes -> can't plot on log scale,
# so mark those points explicitly at the bottom of the axis instead.
ax.loglog(true_grad[~fp16_flushed], fp16_grad[~fp16_flushed], "s-", color="#de2d26", ms=4, label="fp16 (representable)")
if n_flushed:
    ymin = ax.get_ylim()[0]
    ax.scatter(true_grad[fp16_flushed], np.full(n_flushed, ymin * 1.5),
               marker="x", color="#de2d26", s=40, label=f"fp16 flushed to 0 (n={n_flushed})")

ax.axvline(6.1e-5, color="#de2d26", ls=":", lw=1, alpha=0.7)
ax.text(6.1e-5, ax.get_ylim()[1] * 0.4, " fp16 min normal\n ~6.1e-5", fontsize=8, color="#de2d26")

ax.set_xlabel("true gradient magnitude (fp32)")
ax.set_ylabel("value after downcast")
ax.set_title("Day 96: fp16 underflows tiny gradients, bf16 does not")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", alpha=0.3)

plt.savefig("graph_DAY_96.png", dpi=120, bbox_inches="tight")
print(f"fp16 flushed {n_flushed}/{len(true_grad)} gradients to exact zero; bf16 flushed 0.")
