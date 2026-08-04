"""
Day 68 - GPT & Causal LM
Exposure bias, made mechanical: a causal LM is trained with teacher forcing --
every prediction is conditioned on the TRUE previous token, so its per-step
loss is flat, bounded by how well it fits P(x_t | x_{t-1}) on real data. At
inference it must condition on its OWN previous output instead. The instant
it makes one mistake, every later prediction is conditioned on a token
sequence the model never saw during training (an out-of-distribution
context) -- so its error rate from that point on jumps, and once derailed it
tends to stay derailed. Simulated directly below via many independent greedy
rollouts, matching the classic Bengio et al. scheduled-sampling argument.
"""
import numpy as np
import matplotlib.pyplot as plt

N_STEPS = 40
N_TRIALS = 400
EPS_ON_TRACK = 0.06   # per-step error prob when conditioned on the TRUE previous token (in-distribution)
NLL_ON_TRACK = 0.35   # loss when correctly conditioned -- what teacher-forced training actually optimizes
NLL_DERAILED = 2.6    # loss once conditioned on the model's own earlier mistake (out-of-distribution context)

# ---- teacher-forced: ALWAYS conditioned on the true previous token, every step, forever ----
teacher_forced_loss = np.full(N_STEPS, NLL_ON_TRACK)

# ---- free-running: simulate many independent greedy rollouts. Each stays "on
#      track" until its first mistake (prob EPS_ON_TRACK per step); after that,
#      every later step is conditioned on a self-generated, wrong token ----
rng = np.random.default_rng(42)
derail_step = rng.geometric(EPS_ON_TRACK, size=N_TRIALS)  # step index of first mistake, per rollout
free_running_loss = np.zeros(N_STEPS)
for t in range(N_STEPS):
    on_track = derail_step > t  # still on track at step t -> hasn't derailed yet
    free_running_loss[t] = on_track.mean() * NLL_ON_TRACK + (~on_track).mean() * NLL_DERAILED

# closed-form expectation, for reference: P(on track at t) = P(derail_step > t)
p_on_track = (1 - EPS_ON_TRACK) ** np.arange(N_STEPS)
free_running_expected = p_on_track * NLL_ON_TRACK + (1 - p_on_track) * NLL_DERAILED

fig, ax = plt.subplots(figsize=(9, 5))
steps = np.arange(N_STEPS)
ax.plot(steps, teacher_forced_loss, color="#2b6cb0", linewidth=2.5,
        label="teacher-forced (training): always conditioned on TRUE history")
ax.plot(steps, free_running_loss, "o", color="#c53030", alpha=0.5, markersize=4,
        label=f"free-running (inference): {N_TRIALS} simulated greedy rollouts")
ax.plot(steps, free_running_expected, color="#c53030", linewidth=2.5, linestyle="--",
        label="free-running: closed-form expectation")
ax.set_xlabel("generation position t")
ax.set_ylabel("loss / NLL of the true continuation")
ax.set_title("Exposure bias: one early mistake compounds under free-running generation")
ax.legend(fontsize=9, loc="center right")
plt.tight_layout()
plt.savefig("graph_DAY_68.png", dpi=120, bbox_inches="tight")

print("teacher-forced loss (flat):", teacher_forced_loss[-1])
print("free-running loss at t=39:", free_running_loss[-1])
