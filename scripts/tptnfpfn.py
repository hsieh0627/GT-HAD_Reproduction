import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def normalize(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def get_confusion_masks(score_map, gt, threshold_percentile=99):
    """
    score_map: anomaly score map, larger means more anomalous
    gt: ground truth mask, anomaly > 0
    """
    score = normalize(score_map)
    gt_bin = gt > 0

    threshold = np.percentile(score, threshold_percentile)
    pred_bin = score >= threshold

    tp = pred_bin & gt_bin          # 預測異常，GT 也是異常
    tn = (~pred_bin) & (~gt_bin)    # 預測背景，GT 也是背景
    fp = pred_bin & (~gt_bin)       # 預測異常，但 GT 是背景
    fn = (~pred_bin) & gt_bin       # 預測背景，但 GT 是異常

    return score, tp, tn, fp, fn, threshold


def make_color_map(mask, color, shape):
    """
    只把指定 mask 區域畫上顏色，其他地方保持黑色
    """
    rgb = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    rgb[mask] = color
    return rgb


def save_single_map(rgb_img, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_img)
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_combined_map(tp_img, tn_img, fp_img, fn_img, counts, threshold, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    images = [
        (tp_img, f"TP: True Positive\nDetected anomaly correctly: {counts['TP']}"),
        (tn_img, f"TN: True Negative\nBackground correctly ignored: {counts['TN']}"),
        (fp_img, f"FP: False Positive\nBackground misdetected as anomaly: {counts['FP']}"),
        (fn_img, f"FN: False Negative\nMissed anomaly: {counts['FN']}")
    ]

    for ax, (img, title) in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.suptitle(f"TP / TN / FP / FN Maps, threshold={threshold:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Path setting
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 你要跑的資料集
file_list = ["gulfport"]  # ['los-angeles-1', 'los-angeles-2', 'gulfport']

# 如果你要畫原始 main.py 的結果，用 results + GT-HAD_map.mat
result_root = os.path.join(project_root, "results_main2")
method_name = "GT-HAD2"

# 如果你要畫 main2.py 改良版，改成下面兩行：
# result_root = os.path.join(project_root, "results_main2")
# method_name = "GT-HAD-MSRD"

output_root = os.path.join(project_root, "tp_tn_fp_fn_maps")
os.makedirs(output_root, exist_ok=True)


for file in file_list:
    print(f"Processing {file}...")

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

    score, tp, tn, fp, fn, threshold = get_confusion_masks(
        score_map,
        gt,
        threshold_percentile=99
    )

    h, w = gt.shape

    # 顏色設定
    # TP = green, TN = gray, FP = red, FN = blue
    tp_img = make_color_map(tp, color=[0, 1, 0], shape=(h, w))
    tn_img = make_color_map(tn, color=[0.5, 0.5, 0.5], shape=(h, w))
    fp_img = make_color_map(fp, color=[1, 0, 0], shape=(h, w))
    fn_img = make_color_map(fn, color=[0, 0, 1], shape=(h, w))

    counts = {
        "TP": int(tp.sum()),
        "TN": int(tn.sum()),
        "FP": int(fp.sum()),
        "FN": int(fn.sum())
    }

    print(counts)

    save_dir = os.path.join(output_root, file)
    os.makedirs(save_dir, exist_ok=True)

    # 分別輸出四張圖
    save_single_map(
        tp_img,
        f"{file} - TP Map",
        os.path.join(save_dir, "TP_map.png")
    )

    save_single_map(
        tn_img,
        f"{file} - TN Map",
        os.path.join(save_dir, "TN_map.png")
    )

    save_single_map(
        fp_img,
        f"{file} - FP Map",
        os.path.join(save_dir, "FP_map.png")
    )

    save_single_map(
        fn_img,
        f"{file} - FN Map",
        os.path.join(save_dir, "FN_map.png")
    )

    # 另外輸出一張 2x2 合併圖
    save_combined_map(
        tp_img,
        tn_img,
        fp_img,
        fn_img,
        counts,
        threshold,
        os.path.join(save_dir, "TP_TN_FP_FN_combined.png")
    )

    print("Saved to:", save_dir)

print("Done.")