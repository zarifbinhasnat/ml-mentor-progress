import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Day 6 — Computational Graph", fontsize=13, fontweight="bold")

# Panel 1: draw the forward pass DAG for  z = w*x + b, a = sigma(z), L = CE(a,y)
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(-1, 5)
ax.axis("off")
ax.set_title("Forward pass DAG")

nodes = {
    "x":  (1, 4), "w":  (1, 2), "b":  (1, 0),
    "u":  (3.5, 3),
    "z":  (5.5, 3),
    "a":  (7.5, 3),
    "L":  (9.5, 3),
}
colors_n = {"x":"#61afef","w":"#98c379","b":"#e5c07b","u":"#c678dd","z":"#c678dd","a":"#c678dd","L":"#e06c75"}

for name, (xn, yn) in nodes.items():
    circle = plt.Circle((xn, yn), 0.4, color=colors_n[name], zorder=3)
    ax.add_patch(circle)
    ax.text(xn, yn, name, ha="center", va="center", fontsize=11, fontweight="bold", color="white", zorder=4)

# arrows
edges = [("x","u"), ("w","u"), ("u","z"), ("b","z"), ("z","a"), ("a","L")]
for (src, dst) in edges:
    xs, ys = nodes[src]
    xd, yd = nodes[dst]
    ax.annotate("", xy=(xd-0.42, yd), xytext=(xs+0.42, ys),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.5))

# labels on edges
ax.text(2.3, 3.8, "×w", fontsize=9)
ax.text(2.3, 2.2, "×x", fontsize=9)
ax.text(4.5, 3.2, "+b", fontsize=9)
ax.text(6.5, 3.2, "σ(z)", fontsize=9)
ax.text(8.3, 3.2, "CE", fontsize=9)

# Panel 2: compare forward values and backward deltas
ax2 = axes[1]
ax2.set_title("Backward pass: error signal δ per node")

np.random.seed(42)
x_val = 1.0; w_val = 0.5; b_val = 0.1
u_val = w_val * x_val
z_val = u_val + b_val
a_val = 1 / (1 + np.exp(-z_val))
y_val = 1.0
L_val = -y_val * np.log(a_val) - (1 - y_val) * np.log(1 - a_val)

# deltas (backward)
dL_da = -y_val / a_val + (1 - y_val) / (1 - a_val)
dL_dz = dL_da * a_val * (1 - a_val)
dL_du = dL_dz
dL_db = dL_dz
dL_dw = dL_du * x_val

forward_vals = [x_val, w_val, b_val, u_val, z_val, a_val, L_val]
backward_deltas = [None, dL_dw, dL_db, dL_du, dL_dz, dL_da, 1.0]
node_names_ord = ["x", "w", "b", "u", "z", "a", "L"]

xpos = np.arange(len(node_names_ord))
ax2.bar(xpos - 0.2, forward_vals, width=0.35, color="#61afef", alpha=0.8, label="Forward value")
bd = [abs(d) if d is not None else 0 for d in backward_deltas]
ax2.bar(xpos + 0.2, bd, width=0.35, color="#e06c75", alpha=0.8, label="|δ| (backward)")
ax2.set_xticks(xpos)
ax2.set_xticklabels(node_names_ord)
ax2.set_ylabel("Magnitude")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("graph_DAY_6.png", dpi=120, bbox_inches="tight")
print("graph_DAY_6.png saved")
