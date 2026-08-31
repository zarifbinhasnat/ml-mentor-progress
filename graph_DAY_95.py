"""
Day 95 — Object detection: IoU + greedy Non-Max Suppression, visualized.

Synthetic detector output: one true car (cluster of overlapping boxes,
all high-confidence) plus one true pedestrian (single clean box) plus
one low-confidence false positive far away. Runs a from-scratch greedy
NMS pass and plots before/after side by side.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(42)

# each row: [x1, y1, x2, y2, score]  (hardcoded synthetic detector output, <100 points)
boxes = np.array([
    [50, 40, 150, 140, 0.95],   # car - best box
    [55, 45, 148, 138, 0.90],   # car - near-duplicate detection
    [60, 42, 155, 145, 0.82],   # car - another near-duplicate
    [48, 55, 145, 150, 0.71],   # car - looser duplicate, still overlapping
    [300, 60, 360, 200, 0.88],  # pedestrian - clean, isolated box
    [500, 300, 520, 320, 0.35], # far-away low-confidence false positive
], dtype=float)


def iou(box, others):
    # vectorized IoU of one box against an array of boxes (this is the PYTHONIC EDGE trick)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])

    inter_w = np.clip(x2 - x1, 0, None)   # 0 if boxes don't overlap on this axis
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_others = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area_box + area_others - inter

    return inter / union


def nms(boxes, iou_threshold=0.5):
    order = boxes[:, 4].argsort()[::-1]     # sort indices by score, descending
    kept = []
    while order.size > 0:
        current = order[0]
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = iou(boxes[current], boxes[rest])
        order = rest[overlaps <= iou_threshold]   # drop everything that overlaps too much
    return kept


kept_idx = nms(boxes, iou_threshold=0.5)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
titles = ["Before NMS: 6 raw detections", f"After NMS (IoU thr=0.5): {len(kept_idx)} kept"]
box_sets = [range(len(boxes)), kept_idx]

for ax, title, idx_set in zip(axes, titles, box_sets):
    ax.set_xlim(0, 560)
    ax.set_ylim(0, 340)
    ax.invert_yaxis()
    ax.set_title(title)
    for i in idx_set:
        x1, y1, x2, y2, score = boxes[i]
        color = "tab:red" if i in kept_idx else "tab:gray"
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{score:.2f}", color=color, fontsize=9)

plt.tight_layout()
plt.savefig("graph_DAY_95.png", dpi=120, bbox_inches="tight")
