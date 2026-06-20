import os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np

# =========================
# Path setting
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

save_dir = os.path.join(project_root, "roc_compare")
os.makedirs(save_dir, exist_ok=True)

# =========================
# 要比較的版本
# =========================
experiments = {
    "Main_GT-HAD": {
        "result_dir": os.path.join(project_root, "results"),
        "roc_file": "GT-HAD_roc.mat"
    },
    "Main2_GT-HAD-MSRD": {
        "result_dir": os.path.join(project_root, "results_main2"),
        "roc_file": "GT-HAD2_roc.mat"
    }
}

file_list = ["los-angeles-1", "los-angeles-2", "gulfport"]
color_list = ["red", "blue", "green", "orange"]

# =========================
# Draw ROC
# =========================
for file in file_list:
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.set_xscale("log", base=10)
    ax.set_xlim(1e-3, 1.0)
    ax.set_ylim(0.0, 1.05)

    ax.set_xlabel("False alarm rate", fontsize=14, fontweight="bold")
    ax.set_ylabel("Probability of detection", fontsize=14, fontweight="bold")
    ax.set_title(f"ROC Curve Comparison - {file}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    has_curve = False

    for idx, (method_name, info) in enumerate(experiments.items()):
        roc_path = os.path.join(
            info["result_dir"],
            file,
            info["roc_file"]
        )

        print("Loading:", roc_path)

        if not os.path.exists(roc_path):
            print("File not found:", roc_path)
            continue

        mat = sio.loadmat(roc_path)

        fpr = np.squeeze(mat["PF"])
        tpr = np.squeeze(mat["PD"])

        # log scale 不能畫 fpr=0，所以拿掉
        valid = fpr > 0
        fpr_plot = fpr[valid]
        tpr_plot = tpr[valid]

        if len(fpr_plot) == 0:
            print(f"No valid ROC points for {method_name} on {file}")
            continue

        roc_auc = auc(fpr, tpr)
        print(f"{file} - {method_name}: AUC = {roc_auc:.4f}")

        ax.semilogx(
            fpr_plot,
            tpr_plot,
            linewidth=2.5,
            color=color_list[idx % len(color_list)],
            label=f"{method_name} AUC={roc_auc:.4f}"
        )

        has_curve = True

    if has_curve:
        ax.legend(loc="lower right", fontsize=11)
        save_path = os.path.join(save_dir, file + "_roc_compare.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved:", save_path)
    else:
        print(f"No ROC curve saved for {file}")

    plt.close()

print("Done.")