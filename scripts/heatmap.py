import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# folders
results_dir = os.path.join(script_dir, '../results_main2/')
data_dir = os.path.join(script_dir, '../data/')
save_dir = os.path.join(script_dir, '../heat_map/')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def make_rgb_image(hsi):
    """
    hsi shape: (H, W, B)
    使用 first / middle / last band 組成 pseudo-RGB
    """
    h, w, b = hsi.shape
    r = hsi[:, :, 0]
    g = hsi[:, :, b // 2]
    b_ = hsi[:, :, -1]

    rgb = np.stack([r, g, b_], axis=-1)
    rgb = rgb - rgb.min()
    rgb = rgb / (rgb.max() + 1e-8)
    return rgb

def normalize_img(img):
    img = img.astype(np.float32)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img

def plot_original_gt_heatmap(original_img, gt_img, heatmap_img, save_name,
                             title_left="Original", title_mid="Ground Truth", title_right="GT-HAD Output"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左邊：原圖
    axes[0].imshow(original_img)
    axes[0].set_title(title_left, fontsize=12)
    axes[0].axis("off")

    # 中間：Ground Truth
    axes[1].imshow(gt_img, cmap='hot')
    axes[1].set_title(title_mid, fontsize=12)
    axes[1].axis("off")

    # 右邊：Heatmap
    sns.heatmap(
        heatmap_img,
        cmap='turbo',
        vmax=1.0,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        linewidths=0.0,
        ax=axes[2]
    )
    axes[2].set_title(title_right, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()

method_list = ['GT-HAD2']
file_list = ['los-angeles-1', 'los-angeles-2', 'gulfport']

for file in file_list:
    # 讀原始資料與 GT
    data_path = os.path.join(data_dir, file + '.mat')
    data_mat = sio.loadmat(data_path)

    hsi = data_mat['data']     # (H, W, B)
    gt = data_mat['map']       # (H, W)

    # 原圖：pseudo-RGB
    original_img = make_rgb_image(hsi)

    # GT 正規化成顯示用
    gt_img = normalize_img(gt)

    # 結果資料夾
    mat_dir = os.path.join(results_dir, file)

    for method in method_list:
        mat_name = os.path.join(mat_dir, method + '_map.mat')
        print("Loading:", mat_name)

        if not os.path.exists(mat_name):
            print("File not found:", mat_name)
            continue

        mat = sio.loadmat(mat_name)
        heatmap_img = mat['show']
        heatmap_img = normalize_img(heatmap_img)

        # 輸出資料夾
        save_subdir = os.path.join(save_dir, file)
        if not os.path.exists(save_subdir):
            os.makedirs(save_subdir)

        # 輸出 png
        save_name = os.path.join(save_subdir, method + '.png')

        plot_original_gt_heatmap(
            original_img,
            gt_img,
            heatmap_img,
            save_name,
            title_left=f"{file} Original",
            title_mid="Ground Truth",
            title_right=f"{method} Heatmap"
        )

print("Done.")