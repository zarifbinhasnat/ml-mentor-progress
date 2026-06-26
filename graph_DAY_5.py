import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("Day 5 — Softmax + Cross-Entropy", fontsize=13, fontweight="bold")

# Panel 1: CE loss vs predicted probability for correct class
p_hat = np.linspace(0.01, 0.999, 400)
ce_loss = -np.log(p_hat)

ax = axes[0]
ax.plot(p_hat, ce_loss, color="#e06c75", linewidth=2)
ax.set_xlabel("Predicted probability for correct class")
ax.set_ylabel("Cross-entropy loss  −log(p̂)")
ax.set_title("CE loss vs. model confidence")
ax.fill_between(p_hat, ce_loss, alpha=0.15, color="#e06c75")
ax.annotate("Confidently\nwrong → huge loss", xy=(0.05, 3.0), xytext=(0.3, 4),
            arrowprops=dict(arrowstyle="->", color="darkred"), fontsize=8, color="darkred")
ax.annotate("Confidently\nright → near 0", xy=(0.97, 0.03), xytext=(0.65, 1.0),
            arrowprops=dict(arrowstyle="->", color="darkgreen"), fontsize=8, color="darkgreen")
ax.grid(True, alpha=0.3)

# Panel 2: gradient of CE+softmax w.r.t. logit is just p̂ - p
C = 5
logits = np.array([1.5, 3.0, 0.5, -1.0, 2.0])
true_class = 1
exp_z = np.exp(logits - logits.max())
p_hat_vec = exp_z / exp_z.sum()
p_true = np.zeros(C); p_true[true_class] = 1.0
gradient = p_hat_vec - p_true

ax = axes[1]
colors = ["#98c379" if i == true_class else "#61afef" for i in range(C)]
bars = ax.bar(range(C), p_hat_vec, color=colors, edgecolor="k", linewidth=0.7, label="p̂ (softmax)")
for i, (pb, g) in enumerate(zip(p_hat_vec, gradient)):
    ax.text(i, pb + 0.01, f"g={g:+.2f}", ha="center", fontsize=7.5, color="navy")
ax.set_xticks(range(C))
ax.set_xticklabels([f"class {i}" + (" ✓" if i == true_class else "") for i in range(C)], fontsize=8)
ax.set_ylabel("Predicted probability")
ax.set_title("Gradient = p̂ − p  (predicted − true)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

# Panel 3: log-sum-exp numerical stability
ax = axes[2]
z_vals = np.array([100.0, 101.0, 99.0])
# naive (overflows, shown as inf)
shifts = np.arange(0, 200, 5, dtype=float)
naive_ok  = []
stable_ok = []
for shift in shifts:
    zs = z_vals + shift
    try:
        naive = np.exp(zs) / np.exp(zs).sum()
        naive_ok.append(not np.isnan(naive).any())
    except:
        naive_ok.append(False)
    m = zs.max()
    s = np.exp(zs - m); s /= s.sum()
    stable_ok.append(not np.isnan(s).any())

ax.step(shifts, np.array(naive_ok, dtype=float), color="#e06c75", linewidth=2, where="post", label="Naive exp(z) — overflows")
ax.step(shifts, np.array(stable_ok, dtype=float) * 0.95, color="#98c379", linewidth=2, where="post", label="Log-sum-exp stable")
ax.set_xlabel("Logit shift (added to all logits)")
ax.set_ylabel("Numerically valid? (1=yes, 0=nan/inf)")
ax.set_title("Numerical stability of softmax")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.2)

plt.tight_layout()
plt.savefig("graph_DAY_5.png", dpi=120, bbox_inches="tight")
print("graph_DAY_5.png saved")
