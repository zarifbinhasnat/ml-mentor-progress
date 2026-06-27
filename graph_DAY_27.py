import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Dropout: Regularization via Stochastic Masking", fontsize=14, fontweight='bold')

# ── Panel 1: Train vs Test loss with vs without Dropout ──
ax = axes[0]
epochs = np.arange(1, 51)

# Without dropout — overfitting
train_loss_no_drop = 0.9 * np.exp(-0.08 * epochs) + 0.05
val_loss_no_drop   = 0.9 * np.exp(-0.04 * epochs) + 0.18 + 0.003 * epochs

# With dropout — generalization
train_loss_drop    = 0.9 * np.exp(-0.06 * epochs) + 0.10
val_loss_drop      = 0.9 * np.exp(-0.055 * epochs) + 0.12

ax.plot(epochs, train_loss_no_drop, 'b--',  label='Train (no dropout)', linewidth=1.5)
ax.plot(epochs, val_loss_no_drop,   'r--',  label='Val   (no dropout)', linewidth=1.5)
ax.plot(epochs, train_loss_drop,    'b-',   label='Train (dropout)',    linewidth=2)
ax.plot(epochs, val_loss_drop,      'g-',   label='Val   (dropout)',    linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Overfitting vs. Dropout Regularization")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)

# ── Panel 2: Neuron activation distributions ──
ax = axes[1]
x = np.linspace(-3, 3, 300)

# Full network activations (e.g. output of a dense layer before dropout)
full_activations = np.random.randn(1000)

# After dropout (p=0.5): ~half zeroed, other half scaled by 1/(1-p)=2
keep_mask = np.random.rand(1000) > 0.5
dropped_activations = full_activations * keep_mask * 2.0  # inverted dropout scaling

ax.hist(full_activations,  bins=40, alpha=0.6, color='steelblue', density=True, label='Before dropout')
ax.hist(dropped_activations, bins=40, alpha=0.6, color='coral', density=True, label='After dropout (p=0.5)')
ax.axvline(0, color='black', linewidth=0.8, linestyle=':')
ax.set_xlabel("Activation value")
ax.set_ylabel("Density")
ax.set_title("Activation Distribution\n(Inverted Dropout, p=0.5)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 3: Ensemble intuition — p(correct) vs num_subnetworks ──
ax = axes[2]
n = np.arange(1, 21)
# If each sub-network has accuracy 0.7, majority vote accuracy grows with n
p_single = 0.70
# Binomial majority vote: P(>n/2 correct)
from scipy.special import comb as sc_comb

def majority_vote_acc(n_models, p):
    majority = int(np.ceil(n_models / 2))
    return sum(sc_comb(n_models, k, exact=True) * (p**k) * ((1-p)**(n_models-k))
               for k in range(majority, n_models+1))

ensemble_acc = [majority_vote_acc(int(ni), p_single) for ni in n]

ax.plot(n, [p_single]*len(n), 'r--', label=f'Single model ({p_single:.0%})', linewidth=1.5)
ax.plot(n, ensemble_acc, 'g-o', markersize=4, label='Majority-vote ensemble', linewidth=2)
ax.set_xlabel("Number of sub-networks")
ax.set_ylabel("Accuracy (majority vote)")
ax.set_title("Dropout as Implicit Ensembling\n(each sub-net accuracy=70%)")
ax.set_ylim(0.5, 1.0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("graph_DAY_27.png", dpi=120, bbox_inches="tight")
print("Saved graph_DAY_27.png")
