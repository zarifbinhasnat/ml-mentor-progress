"""
Day 71 -- Instruction tuning.

Simulates the core empirical finding behind instruction tuning (Wei et al.,
FLAN, 2021): zero-shot generalization to a HELD-OUT task improves with the
*number of distinct tasks* seen during fine-tuning, not with the total
number of training examples.

Toy model: every task's true decision boundary is w_task = w_shared +
noise_task, where w_shared is a fixed direction common to ALL tasks (a stand
-in for "the general instruction-following skill") and noise_task is a
random, task-specific component ("quirks of this one task"). Fitting a task
from examples gives a noisy estimate of w_task.

  - "single-task fine-tuning" (Day 69's approach, more examples of ONE task):
    the estimate converges to w_task = w_shared + noise_task -- more data
    only shrinks *estimation* noise, but the task's own private noise_task
    term never goes away. Accuracy on an unrelated held-out task plateaus,
    capped by how correlated that one task's private direction happens to
    be with the shared one.
  - "instruction tuning" (more DISTINCT tasks, few examples each): averaging
    estimates across many independent tasks cancels the noise_task terms by
    the same law-of-large-numbers logic as any Monte Carlo average, leaving
    something close to pure w_shared -- which transfers to any new task.

Same total example budget spent either way.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

D = 6  # shared "frozen representation" dimensionality
n_total_tasks = 500

w_shared = np.random.randn(D)
w_shared /= np.linalg.norm(w_shared)  # the common "follow instructions" direction

task_noise_sigma = 0.35
task_noise = np.random.randn(n_total_tasks, D) * task_noise_sigma
task_weights = w_shared[None, :] + task_noise  # w_task = w_shared + noise_task

held_out_idx = n_total_tasks - 1
train_task_pool = list(range(n_total_tasks - 1))
single_task_id = 0  # the one task all of Day-69-style fine-tuning is done on


def sample_task_points(task_id, n, rng):
    X = rng.randn(n, D)
    y = (X @ task_weights[task_id] > 0).astype(int)
    return X, y


def fit_task_direction(task_id, n_examples, rng):
    X, y = sample_task_points(task_id, n_examples, rng)
    sign = np.where(y == 1, 1.0, -1.0)
    return (X * sign[:, None]).mean(axis=0)  # correlation estimate of w_task


def eval_held_out(w_hat, rng, n_query=400):
    Xq, yq = sample_task_points(held_out_idx, n_query, rng)
    preds = (Xq @ w_hat > 0).astype(int)
    return (preds == yq).mean()


budget_points = [10, 20, 40, 80, 160, 320, 640, 1280]
examples_per_task_div = 5  # fixed, shallow per-task exposure
n_trials = 30
rng = np.random.RandomState(42)

diversity_acc, single_task_acc = [], []
for budget in budget_points:
    n_tasks_div = min(len(train_task_pool), max(1, budget // examples_per_task_div))

    accs_div, accs_single = [], []
    for _ in range(n_trials):
        tasks_div = rng.choice(train_task_pool, size=n_tasks_div, replace=False)
        w_div = np.mean([fit_task_direction(t, examples_per_task_div, rng) for t in tasks_div], axis=0)
        accs_div.append(eval_held_out(w_div, rng))

        w_single = fit_task_direction(single_task_id, budget, rng)
        accs_single.append(eval_held_out(w_single, rng))

    diversity_acc.append(np.mean(accs_div))
    single_task_acc.append(np.mean(accs_single))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(budget_points, diversity_acc, marker="o", color="#1f77b4",
        label="instruction tuning (more tasks, 5 ex/task)")
ax.plot(budget_points, single_task_acc, marker="o", color="#d62728",
        label="single-task fine-tuning (Day 69 style, 1 task)")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
ax.set_xscale("log", base=2)
ax.set_xticks(budget_points)
ax.set_xticklabels(budget_points)
ax.set_xlabel("total fine-tuning examples (fixed budget)")
ax.set_ylabel("zero-shot accuracy on held-out task")
ax.set_title("Instruction tuning: task diversity vs. single-task scaling, same budget")
ax.legend(loc="lower right", fontsize=8)
ax.set_ylim(0.45, 1.02)
fig.tight_layout()
fig.savefig("graph_DAY_71.png", dpi=120, bbox_inches="tight")
