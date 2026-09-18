"""
Day 109 — Agents & tool use: aliasing when an agent's tool-call rate
undersamples a fast-changing environment (Nyquist applied to agentic loops).
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

f_true = 8.0     # Hz, true environment oscillation (e.g. intraday price microstructure)
fs_agent = 10.0  # Hz, agent's tool-call budget (samples/sec)

# Dense curve = ground truth the agent never directly sees (rendering resolution only).
t_dense = np.linspace(0, 1, 500)
true_signal = np.sin(2 * np.pi * f_true * t_dense)

# The agent's actual observations: one tool call every 1/fs_agent seconds.
t_sampled = np.arange(0, 1, 1 / fs_agent)
observed = np.sin(2 * np.pi * f_true * t_sampled)

# What frequency those samples actually look like once connected (the aliased signal).
f_alias = abs(f_true - round(f_true / fs_agent) * fs_agent)
t_fine = np.linspace(0, 1, 500)
alias_curve = np.sin(2 * np.pi * f_alias * t_fine + 0.0)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_dense, true_signal, color="#888888", lw=1.5, alpha=0.6,
        label=f"true env. state ({f_true:.0f} Hz)")
ax.plot(t_fine, alias_curve, color="#d62728", lw=2, ls="--",
        label=f"what the agent perceives ({f_alias:.0f} Hz alias)")
ax.scatter(t_sampled, observed, color="#1f77b4", zorder=5, s=60,
           label=f"agent tool calls ({fs_agent:.0f}/s)")

ax.set_xlabel("time (s)")
ax.set_ylabel("state value")
ax.set_title("Undersampling a fast environment: tool-call rate < Nyquist rate")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(-1.4, 1.4)

plt.savefig("graph_DAY_109.png", dpi=120, bbox_inches="tight")
