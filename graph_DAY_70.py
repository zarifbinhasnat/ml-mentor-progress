"""
Day 70 -- In-context learning & prompting.

Simulates ICL as a frozen-representation nearest-neighbor rule: a "prompt"
supplies k labeled demonstrations, and the frozen model classifies a query
by similarity to those demonstrations (no weight updates, ever). Compares
accuracy as a function of shot count k under two conditions:
  - correct demonstration labels
  - demonstration labels randomly shuffled (same inputs, same format,
    same k -- only the label mapping is destroyed)

This mirrors the well-known empirical finding (Min et al. 2022) that a large
chunk of ICL's benefit comes from exposure to the input distribution and
prompt *format*, not strictly from correct label mapping -- but correct
labels still matter and the gap should be visible.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---- synthetic two-class task in a toy 2D "frozen embedding" space ----
n_per_class = 40
class0 = np.random.randn(n_per_class, 2) * 0.9 + np.array([-1.5, 0.0])
class1 = np.random.randn(n_per_class, 2) * 0.9 + np.array([1.5, 0.0])
X = np.vstack([class0, class1])
y = np.array([0] * n_per_class + [1] * n_per_class)

# held-out query set, drawn from the same distribution
n_query = 60
q0 = np.random.randn(n_query // 2, 2) * 0.9 + np.array([-1.5, 0.0])
q1 = np.random.randn(n_query // 2, 2) * 0.9 + np.array([1.5, 0.0])
X_query = np.vstack([q0, q1])
y_query = np.array([0] * (n_query // 2) + [1] * (n_query // 2))

shots = [1, 2, 4, 8, 16, 32, 64]
n_trials = 30  # average over random demonstration subsets for a smooth curve


def knn_predict(x, demo_X, demo_y, k=3):
    # frozen "similarity lookup" -- exactly what ICL does over demonstrations,
    # just with real transformer attention instead of raw Euclidean distance
    k = min(k, len(demo_X))
    dists = np.linalg.norm(demo_X - x, axis=1)
    nearest = np.argsort(dists)[:k]
    votes = demo_y[nearest]
    return 1 if votes.mean() >= 0.5 else 0


def accuracy_for_shots(n_shots, shuffle_labels, rng):
    accs = []
    for _ in range(n_trials):
        idx = rng.choice(len(X), size=n_shots, replace=False)
        demo_X, demo_y = X[idx], y[idx].copy()
        if shuffle_labels:
            rng.shuffle(demo_y)  # same inputs/format, label mapping destroyed
        preds = np.array([knn_predict(q, demo_X, demo_y) for q in X_query])
        accs.append((preds == y_query).mean())
    return np.mean(accs)


rng = np.random.RandomState(42)
acc_correct = [accuracy_for_shots(k, shuffle_labels=False, rng=rng) for k in shots]
acc_shuffled = [accuracy_for_shots(k, shuffle_labels=True, rng=rng) for k in shots]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(shots, acc_correct, marker="o", color="#1f77b4", label="correct demo labels")
ax.plot(shots, acc_shuffled, marker="o", color="#d62728", label="shuffled demo labels")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
ax.set_xscale("log", base=2)
ax.set_xticks(shots)
ax.set_xticklabels(shots)
ax.set_xlabel("number of in-context demonstrations (shots)")
ax.set_ylabel("query accuracy (frozen model, zero weight updates)")
ax.set_title("In-context learning: accuracy vs. shot count and label quality")
ax.legend(loc="lower right")
ax.set_ylim(0.35, 1.02)
fig.tight_layout()
fig.savefig("graph_DAY_70.png", dpi=120, bbox_inches="tight")
