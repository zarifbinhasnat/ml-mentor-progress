"""
Day 105 -- Q-learning & DQN: tabular Q-learning converging to the Bellman-optimal Q-value.

Environment: a 5-state 1D chain (states 0..4). The agent starts at state 0
each episode; actions = {left, right}; reward is -0.01 per step and +1.0 on
reaching the goal at state 4 (episode ends there). This toy chain has a
closed-form optimal Q*(s, right) computable by value iteration, so we can
watch the model-free TD update from CONCEPT OF THE DAY converge to the
known-correct answer, plus watch the greedy policy's step count shrink to
the optimal 4 steps as Q improves.

Synthetic environment only, no downloads. np.random.seed(42) for reproducibility.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N_STATES = 5
GOAL = N_STATES - 1
GAMMA = 0.9
ALPHA = 0.5
EPSILON = 0.2
N_EPISODES = 70
STEP_REWARD = -0.01
GOAL_REWARD = 1.0
ACTIONS = [-1, 1]  # left, right


def step(s, a):
    s_next = int(np.clip(s + ACTIONS[a], 0, GOAL))
    if s_next == GOAL:
        return s_next, GOAL_REWARD, True
    return s_next, STEP_REWARD, False


# ---- ground truth via value iteration (environment-only, never seen by the agent) ----
Q_star = np.zeros((N_STATES, 2))
for _ in range(200):
    Q_new = np.zeros_like(Q_star)
    for s in range(N_STATES):
        for a in range(2):
            s_next, r, done = step(s, a)
            bootstrap = 0.0 if done else Q_star[s_next].max()
            Q_new[s, a] = r + GAMMA * bootstrap
    Q_star = Q_new
true_q0_right = Q_star[0, 1]

# ---- tabular Q-learning: model-free, epsilon-greedy, learns from sampled transitions only ----
Q = np.zeros((N_STATES, 2))
q0_right_history = []
steps_per_episode = []

for ep in range(N_EPISODES):
    s = 0
    n_steps = 0
    while True:
        if np.random.rand() < EPSILON:  # epsilon-greedy exploration
            a = np.random.randint(2)
        else:
            a = int(np.argmax(Q[s]))
        s_next, r, done = step(s, a)
        td_target = r + (0.0 if done else GAMMA * Q[s_next].max())
        Q[s, a] += ALPHA * (td_target - Q[s, a])  # the Q-learning update from CONCEPT OF THE DAY
        s = s_next
        n_steps += 1
        if done or n_steps > 50:
            break
    q0_right_history.append(Q[0, 1])
    steps_per_episode.append(n_steps)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(range(N_EPISODES), q0_right_history, color="#3b82f6", linewidth=2,
         label=r"learned $Q(s{=}0,\ \mathrm{right})$")
ax1.axhline(true_q0_right, color="#ef4444", linestyle="--", linewidth=1.5,
            label=r"true $Q^*(0,\mathrm{right})=%.3f$" % true_q0_right)
ax1.set_xlabel("episode")
ax1.set_ylabel("Q-value")
ax1.set_title("TD bootstrapping converges to $Q^*$")
ax1.legend(loc="lower right", fontsize=9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2.plot(range(N_EPISODES), steps_per_episode, color="#10b981", linewidth=1.5)
ax2.axhline(GOAL, color="#ef4444", linestyle="--", linewidth=1.5, label=f"optimal = {GOAL} steps")
ax2.set_xlabel("episode")
ax2.set_ylabel("steps to reach goal")
ax2.set_title("Policy improves as Q converges")
ax2.legend(loc="upper right", fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("graph_DAY_105.png", dpi=120, bbox_inches="tight")
print(f"true Q*(0,right) = {true_q0_right:.4f}, learned = {q0_right_history[-1]:.4f}")
print(f"final steps/episode = {steps_per_episode[-1]} (optimal = {GOAL})")
