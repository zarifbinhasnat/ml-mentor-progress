import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

D = 64          # embedding dimension
K = 11          # number of negative candidates
n_trials = 300  # averaged trials for the smooth noise-sweep curve


def rand_unit(n, d):
    v = np.random.randn(n, d)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ---------- Panel 1: one illustrative trial -- cosine similarity of a text query to
# 1 true image match (noisy embedding) + K random negatives, exactly the per-row
# similarity vector CLIP's S_ij matrix computes for a single anchor ----------
query = rand_unit(1, D)[0]
negatives = rand_unit(K, D)
noise_demo = 0.4
positive_demo = query + noise_demo * np.random.randn(D)
positive_demo /= np.linalg.norm(positive_demo)
sims_demo = np.concatenate([[positive_demo @ query], negatives @ query])
order = np.argsort(-sims_demo)
colors = ["firebrick" if i == 0 else "steelblue" for i in order]

# ---------- Panel 2: temperature (logit_scale = 1/tau) sharpens the softmax over
# that SAME similarity vector -- this is exactly the logit_scale in the CLIP loss ----------
scales = [1, 10, 100]
probs_by_scale = {s: softmax(s * sims_demo[order]) for s in scales}

# ---------- Panel 3: sweep embedding noise (domain gap / imperfect encoder) and
# measure mean P(true match) at each temperature -- the matched-filter
# "selectivity vs. robustness" tradeoff: a sharper filter has more gain when clean,
# but collapses faster once the signal (embedding) is noisy ----------
noise_levels = np.linspace(0.0, 1.5, 25)
p_correct = {s: [] for s in scales}

for sigma in noise_levels:
    qs = rand_unit(n_trials, D)
    negs = rand_unit(n_trials * K, D).reshape(n_trials, K, D)
    pos = qs + sigma * np.random.randn(n_trials, D)
    pos /= np.linalg.norm(pos, axis=-1, keepdims=True)

    trial_sims = np.zeros((n_trials, K + 1))
    trial_sims[:, 0] = np.sum(pos * qs, axis=-1)
    trial_sims[:, 1:] = np.einsum("nkd,nd->nk", negs, qs)

    for s in scales:
        p = softmax(s * trial_sims)
        p_correct[s].append(p[:, 0].mean())

# ---------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

ax = axes[0]
ax.bar(range(len(sims_demo)), sims_demo[order], color=colors)
ax.set_xticks(range(len(sims_demo)))
ax.set_title("cosine similarity: query vs. candidates")
ax.set_xlabel("candidate (sorted by similarity, red = true match)")
ax.set_ylabel("cos similarity")

ax = axes[1]
x = np.arange(len(sims_demo))
for s in scales:
    ax.plot(x, probs_by_scale[s], marker="o", label=f"logit_scale={s}")
ax.set_title("softmax over the SAME similarities, 3 temperatures")
ax.set_xlabel("candidate (sorted by similarity)")
ax.set_ylabel("softmax probability")
ax.legend(fontsize=8)

ax = axes[2]
for s in scales:
    ax.plot(noise_levels, p_correct[s], label=f"logit_scale={s}")
ax.set_title("P(true match) vs. embedding noise, by temperature")
ax.set_xlabel("noise sigma on the positive embedding")
ax.set_ylabel("mean P(correct)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("graph_DAY_90.png", dpi=120, bbox_inches="tight")
