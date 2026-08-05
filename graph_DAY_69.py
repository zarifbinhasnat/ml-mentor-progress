"""
Day 69 -- Transfer learning & fine-tuning.

Simulates logistic regression on a small TARGET task, starting from two
different initializations:
  - "scratch"     : random init, trained only on the small target set
  - "fine-tuned"  : init at weights learned on a large, related SOURCE task,
                    then fine-tuned on the same small target set

Shows the classic transfer-learning effect: a good pretrained starting point
converges faster and generalizes better when target data is scarce.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_task(w_true, b_true, n, noise=0.3):
    X = np.random.randn(n, 2)
    logits = X @ w_true + b_true + noise * np.random.randn(n)
    y = (sigmoid(logits) > 0.5).astype(float)
    return X, y


def bce_loss(w, b, X, y):
    p = np.clip(sigmoid(X @ w + b), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def grad_step(w, b, X, y, lr):
    p = sigmoid(X @ w + b)
    err = p - y
    grad_w = X.T @ err / len(y)
    grad_b = np.mean(err)
    return w - lr * grad_w, b - lr * grad_b


# ---- SOURCE task: large dataset, defines the "pretraining" boundary ----
w_source_true = np.array([2.0, -1.5])
b_source_true = 0.2
X_source, y_source = make_task(w_source_true, b_source_true, n=500)

w_pre, b_pre = np.zeros(2), 0.0
for _ in range(300):  # pretrain to convergence on the big source set
    w_pre, b_pre = grad_step(w_pre, b_pre, X_source, y_source, lr=0.5)

# ---- TARGET task: small, related-but-shifted dataset ----
w_target_true = w_source_true + np.array([0.6, 0.4])  # related, not identical
b_target_true = -0.3
X_train, y_train = make_task(w_target_true, b_target_true, n=15)  # scarce data
X_val, y_val = make_task(w_target_true, b_target_true, n=300)

EPOCHS = 60
LR = 0.3

# scratch: random init, trained only on the tiny target set
w_scratch = 0.5 * np.random.randn(2)
b_scratch = 0.0
scratch_val_losses = []

# fine-tuned: starts from the pretrained source weights
w_ft, b_ft = w_pre.copy(), b_pre
ft_val_losses = []

for epoch in range(EPOCHS):
    scratch_val_losses.append(bce_loss(w_scratch, b_scratch, X_val, y_val))
    ft_val_losses.append(bce_loss(w_ft, b_ft, X_val, y_val))
    w_scratch, b_scratch = grad_step(w_scratch, b_scratch, X_train, y_train, LR)
    w_ft, b_ft = grad_step(w_ft, b_ft, X_train, y_train, LR)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(scratch_val_losses, color="tab:red", linewidth=2, label="from scratch (random init)")
ax.plot(ft_val_losses, color="tab:blue", linewidth=2, label="fine-tuned (pretrained init)")
ax.set_xlabel("epoch")
ax.set_ylabel("validation BCE loss")
ax.set_title("Transfer learning on a scarce target task (n=15 train examples)")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig("graph_DAY_69.png", dpi=120, bbox_inches="tight")
