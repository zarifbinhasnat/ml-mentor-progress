import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Wald's Sequential Probability Ratio Test (SPRT): H0: theta=0 vs H1: theta=1, unit-variance Gaussian.
# This is the "stop gathering evidence vs act" decision an agent makes every loop iteration:
# keep calling tools (keep sampling) or stop and answer (cross a decision boundary).
alpha, beta = 0.05, 0.05                    # false-accept / false-reject rates for the two hypotheses
upper = np.log((1 - beta) / alpha)          # cross this -> stop, accept H1 ("enough evidence, act now")
lower = np.log(beta / (1 - alpha))          # cross this -> stop, accept H0 ("enough evidence, act now")

true_theta = 1.0                            # ground truth: H1 is actually correct
max_steps = 40
samples = true_theta + np.random.randn(max_steps)   # sequential noisy observations ("tool call" results)

llr = 0.0
trace = [0.0]
stop_step = None
for i, x in enumerate(samples, start=1):
    llr += x - 0.5                          # log-likelihood-ratio increment for N(1,1) vs N(0,1)
    trace.append(llr)
    if stop_step is None and (llr >= upper or llr <= lower):
        stop_step = i

trace = np.array(trace)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(trace, color="#1f77b4", lw=2, label="cumulative log-likelihood ratio")
ax.axhline(upper, color="#d62728", ls="--", lw=1.5, label=f"upper bound (accept H1) = {upper:.2f}")
ax.axhline(lower, color="#2ca02c", ls="--", lw=1.5, label=f"lower bound (accept H0) = {lower:.2f}")
if stop_step is not None:
    ax.axvline(stop_step, color="gray", ls=":", lw=1)
    ax.scatter([stop_step], [trace[stop_step]], color="black", zorder=5,
               label=f"stop at step {stop_step}")
ax.set_xlabel("observation index (tool call / evidence step)")
ax.set_ylabel("log-likelihood ratio")
ax.set_title("SPRT: stop-and-act vs keep-gathering-evidence (agent loop analogy)")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
plt.savefig("graph_DAY_109.png", dpi=120, bbox_inches="tight")
