import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def normalize(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# =========================
# Path setting
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

file_list = ["gulfport"]  # ['los-angeles-1', 'los-angeles-2', 'gulfport']

# 原始 main.py 結果
result_root = os.path.join(project_root, "results")
method_name = "GT-HAD"

# 如果要算 main2 改良版，改成下面兩行：
# result_root = os.path.join(project_root, "results_main2")
# method_name = "GT-HAD-MSRD"

save_root = os.path.join(project_root, "confusion_matrix")
os.makedirs(save_root, exist_ok=True)


for file in file_list:
    print("=" * 60)
    print("Dataset:", file)

    data_file = os.path.join(project_root, "data", file + ".mat")
    result_file = os.path.join(result_root, file, method_name + "_map.mat")

    if not os.path.exists(data_file):
        print("Data file not found:", data_file)
        continue

    if not os.path.exists(result_file):
        print("Result file not found:", result_file)
        continue

    gt = sio.loadmat(data_file)["map"]
    score_map = sio.loadmat(result_file)["show"]

    score = normalize(score_map)
    gt_bin = (gt > 0).astype(np.uint8)

    # threshold：異常通常很少，先用 99 percentile
    threshold = np.percentile(score, 99)
    pred_bin = (score >= threshold).astype(np.uint8)

    y_true = gt_bin.flatten()
    y_pred = pred_bin.flatten()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm = [[TN, FP],
    #       [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Threshold: {threshold:.6f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # =========================
    # Plot confusion matrix
    # =========================
    save_dir = os.path.join(save_root, file)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")

    plt.title(f"{file} - {method_name}\nThreshold={threshold:.4f}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks([0, 1], ["Background", "Anomaly"])
    plt.yticks([0, 1], ["Background", "Anomaly"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)

    plt.colorbar()
    plt.tight_layout()

    save_path = os.path.join(save_dir, method_name + "_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)

print("Done.")