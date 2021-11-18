import sys
import os
import numpy as np

model = sys.argv[1]

MIN_IOU = 0.5
n_steps = 200
tick_step = 0.1

# Label is in format [x_center, y_center, width, height]
def label2bbox(label):
    x_center, y_center, width, height = label
    bbox = {
        "x1": x_center - width,
        "x2": x_center + width,
        "y1": y_center - height,
        "y2": y_center + height,
    }
    return bbox


# Credit to https://stackoverflow.com/a/42874377
def compute_iou(bbox1, bbox2):
    x_left = max(bbox1["x1"], bbox2["x1"])
    y_top = max(bbox1["y1"], bbox2["y1"])
    x_right = min(bbox1["x2"], bbox2["x2"])
    y_bottom = min(bbox1["y2"], bbox2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
    bbox2_area = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])

    union_area = float(bbox1_area + bbox2_area - intersection_area)

    iou = intersection_area / union_area

    return iou


predDir = f"./tests/RGB-{model}/labels"
labelDir = "./datasets/rgb/labels"

preds = os.listdir(predDir)

label_count = np.zeros(len(preds))
conf_thres = np.linspace(1/n_steps, 1, n_steps)
tp = np.zeros((len(preds), len(conf_thres)))
fp = np.zeros((len(preds), len(conf_thres)))


for i, pred in enumerate(preds):
    with open(f"{labelDir}/{pred}") as labelFile:
        lines = labelFile.readlines()
    labels = np.array(
        [list(map(lambda x: float(x), line.split()[1:])) for line in lines]
    )
    label_count[i] = labels.shape[0]
    with open(f"{predDir}/{pred}") as predFile:
        while (line := predFile.readline()) != "":
            l = line.split()
            conf = float(l[-1])
            predLabel = np.array(list(map(lambda x: float(x), l[1:-1])))
            correct = False
            for label in labels:
                bboxPred = label2bbox(predLabel)
                bboxLabel = label2bbox(label)
                iou = compute_iou(bboxPred, bboxLabel)

                if iou > MIN_IOU:
                    correct = True
                    break

            if correct:
                for j, c in enumerate(conf_thres):
                    if conf > c:
                        tp[i][j] += 1
                    else:
                        break
            else:
                for j, c in enumerate(conf_thres):
                    if conf > c:
                        fp[i][j] += 1
                    else:
                        break

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

images_containing_sheep = np.where(label_count > 0, 1, 0).sum()

least_one_tp = np.array(
    [np.where(tp[:, i] > 0, 1, 0) for i in range(tp.shape[1])]
).sum(axis=1) / images_containing_sheep

avg_fp = fp.mean(axis=0)

fig, ax = plt.subplots(figsize=(8,5))
ax2 = ax.twinx()

(line,) = ax.plot(
    conf_thres,
    least_one_tp,
    linewidth=3,
)
bar = ax2.bar(conf_thres, avg_fp, 1/n_steps, color="orange")

ticks = np.arange(0.0, 1.0, tick_step) + tick_step

ax.set_zorder(ax2.get_zorder()+1)
ax.set_facecolor("#00000000")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
#ax.grid()

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 2*avg_fp[len(avg_fp)//2])

ax.set_xlabel("Confidence threshold")
ax.set_ylabel("True positives ≥ 1 (% of images)")
ax2.set_ylabel("False positive count")

ax.legend(
    [line, bar],
    [
        "Predictions with ≥ 1 true positives",
        "Average number of false positives",
    ],
    bbox_to_anchor=(0.745,1.155)
)

plt.savefig(f"./tests/RGB-{model}/1_TP_curve.png")
