import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# Simulate per-parameter effective LR for AdaGrad vs RMSprop
# Parameter A: frequent large gradients; Parameter B: rare small gradients
steps = 200
lr = 0.1; rho = 0.9; eps = 1e-8

# Parameter A — large gradients (g ~ N(0, 4))
ga = np.random.randn(steps) * 2.0
# Parameter B — sparse gradients (mostly 0, occasional spike)
gb = np.zeros(steps)
gb[np.random.choice(steps, 20, replace=False)] = np.random.randn(20) * 3

def adagrad_lr(grads, lr, eps):
    G = 0.0; eff_lrs = []
    for g in grads:
        G += g**2
        eff_lrs.append(lr / (np.sqrt(G) + eps))
    return np.array(eff_lrs)

def rmsprop_lr(grads, lr, rho, eps):
    v = 0.0; eff_lrs = []
    for g in grads:
        v = rho * v + (1 - rho) * g**2
        eff_lrs.append(lr / (np.sqrt(v) + eps))
    return np.array(eff_lrs)

eff_ada_a = adagrad_lr(ga, lr, eps)
eff_rms_a = rmsprop_lr(ga, lr, rho, eps)
eff_ada_b = adagrad_lr(gb, lr, eps)
eff_rms_b = rmsprop_lr(gb, lr, rho, eps)

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle("Day 15 (AdaGrad & RMSprop) — Effective Learning Rate per Parameter", fontsize=13, fontweight="bold")

for row, (grads, ada, rms, tag) in enumerate([
    (ga, eff_ada_a, eff_rms_a, "Param A — frequent, large gradients"),
    (gb, eff_ada_b, eff_rms_b, "Param B — sparse, occasional gradients"),
]):
    ax_g = axes[row][0]
    ax_g.plot(grads, color="#c678dd", linewidth=0.9, alpha=0.8)
    ax_g.axhline(0, color="k", linewidth=0.5)
    ax_g.set_xlabel("Step"); ax_g.set_ylabel("Gradient g_t")
    ax_g.set_title(f"{tag}: gradient signal")
    ax_g.grid(True, alpha=0.3)

    ax_lr = axes[row][1]
    ax_lr.plot(ada, color="#e06c75", linewidth=1.8, label="AdaGrad eff. LR (monotone ↓)")
    ax_lr.plot(rms, color="#61afef", linewidth=1.8, label=f"RMSprop eff. LR (ρ={rho})")
    ax_lr.set_xlabel("Step"); ax_lr.set_ylabel("Effective LR η / √(G+ε)")
    ax_lr.set_title(f"Effective LR — AdaGrad dies, RMSprop recovers")
    ax_lr.legend(fontsize=8.5); ax_lr.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_15b.png", dpi=120, bbox_inches="tight")
print("graph_DAY_15b.png saved")
