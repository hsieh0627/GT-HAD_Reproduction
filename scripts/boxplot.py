import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

gt_dir = os.path.join(project_root, 'data')
results_dir = os.path.join(project_root, 'results')
save_dir = os.path.join(project_root, 'box_plot')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

method_list = ['GT-HAD']
file_list = ['los-angeles-1', 'los-angeles-2', 'gulfport']

for file in file_list:
    print(f"Processing {file}...")

    gt_path = os.path.join(gt_dir, file + '.mat')
    if not os.path.exists(gt_path):
        print("GT file not found:", gt_path)
        continue

    mat = sio.loadmat(gt_path)
    gt = mat['map']

    mat_dir = os.path.join(results_dir, file)

    data = []
    labels = []
    colors = []

    for method in method_list:
        mat_name = os.path.join(mat_dir, method + '_map.mat')

        if not os.path.exists(mat_name):
            print("Result file not found:", mat_name)
            continue

        mat = sio.loadmat(mat_name)
        img = mat['show'].astype(np.float32)

        # normalize
        img = img - img.min()
        img = img / (img.max() + 1e-8)

        # anomaly pixels and background pixels
        ab = img[gt != 0].flatten()
        bk = img[gt == 0].flatten()

        print(f"{method} anomaly pixels:", len(ab))
        print(f"{method} background pixels:", len(bk))

        if len(ab) == 0 or len(bk) == 0:
            print(f"Warning: empty anomaly/background data for {file}")
            continue

        data.append(ab)
        data.append(bk)

        labels.append(method + "\nAnomaly")
        labels.append(method + "\nBackground")

        colors.append((1, 0, 0))  # red
        colors.append((0, 0, 1))  # blue

    if len(data) == 0:
        print(f"No valid data for {file}")
        continue

    # draw boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(False)
    ax.set_ylim(0.0, 1.19)

    ax.set_ylabel(
        'Normalized detection statistic range',
        fontsize=13,
        fontweight='bold'
    )

    positions = np.arange(1, len(data) + 1)

    bp = ax.boxplot(
        data,
        widths=0.35,
        patch_artist=True,
        showfliers=False,
        positions=positions,
        medianprops={'color': 'black'},
        whiskerprops={'linestyle': '--'}
    )

    # set box colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set(linewidth=0.75)

    # set whisker and cap colors
    for i, color in enumerate(colors):
        bp['whiskers'][2*i].set(color=color, linewidth=2.0)
        bp['whiskers'][2*i+1].set(color=color, linewidth=2.0)
        bp['caps'][2*i].set(color=color, linewidth=2.0)
        bp['caps'][2*i+1].set(color=color, linewidth=2.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, fontsize=10, fontweight='bold')

    # legend
    anomaly_patch = plt.Line2D([0], [0], color='red', lw=8)
    background_patch = plt.Line2D([0], [0], color='blue', lw=8)
    ax.legend(
        [anomaly_patch, background_patch],
        ['Anomaly', 'Background'],
        loc='upper right',
        fontsize=11
    )

    save_path = os.path.join(save_dir, file + '_boxplot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved:", save_path)

print("Done.")