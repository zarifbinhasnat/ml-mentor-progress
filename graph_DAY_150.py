"""
Day 150 - Consolidation: sequence modelling thread.
Left : gradient reach across lag k -- vanilla tanh RNN vs gated (LSTM-style cell path) vs attention (direct path).
Right: sinusoidal positional-encoding similarity PE(p).PE(p+k)/d versus offset k (a sum of cosines = filter bank).
Synthetic, deterministic (seed 42), numpy only, nothing downloaded.
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
H, T = 16, 80

# --- left: ||d h_T / d h_{T-k}|| for a vanilla tanh RNN with spectral radius 0.9
W = np.random.randn(H, H)
W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))
h = np.zeros(H)
Js = []
for t in range(T):
    h = np.tanh(W @ h + 0.5 * np.random.randn(H))
    Js.append(np.diag(1 - h ** 2) @ W)             # one-step Jacobian
rnn, J = [], np.eye(H)
for k in range(1, T + 1):
    J = J @ Js[T - k]
    rnn.append(np.linalg.norm(J, 2))
lags = np.arange(1, T + 1)
f = np.clip(0.95 + 0.03 * np.random.randn(T), 0, 1)  # LSTM forget gates near 1
lstm = np.cumprod(f)
attn = np.full(T, 0.3)                              # direct path: softmax weight, no product over lag

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(lags, rnn, label="vanilla tanh RNN (rho=0.9)")
ax[0].semilogy(lags, lstm, label="LSTM cell path, prod of forget gates")
ax[0].semilogy(lags, attn, "--", label="attention (path length 1)")
ax[0].set_xlabel("lag k (tokens back)"); ax[0].set_ylabel("gradient reach")
ax[0].set_title("Gradient reach vs lag"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
print(f"lag 50: RNN={rnn[49]:.2e}  LSTM={lstm[49]:.3f}  attention={attn[49]:.2f}")

# --- right: PE similarity vs offset
d = 64
i = np.arange(d // 2)
omega = 10000.0 ** (-2 * i / d)
k = np.arange(0, 301)
sim = np.cos(np.outer(k, omega)).sum(1) / (d / 2)   # PE(p).PE(p+k) / (d/2), independent of p
ax[1].plot(k, sim)
ax[1].set_xlabel("offset k"); ax[1].set_ylabel("normalized PE dot product")
ax[1].set_title("PE similarity: sum of 32 cosines"); ax[1].grid(alpha=0.3)
periods = 2 * np.pi / omega
n_long = int((periods > 512).sum())
print(f"pairs with period > 512: {n_long} of {d//2};  fastest period={periods[0]:.2f} tokens (Nyquist = 2)")
print(f"sim at k=1: {sim[1]:.3f}, k=10: {sim[10]:.3f}, k=100: {sim[100]:.3f}")

plt.savefig("graph_DAY_150.png", dpi=120, bbox_inches="tight")
