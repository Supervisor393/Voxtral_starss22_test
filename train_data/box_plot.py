import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# 路径与文件模式（按你的组织方式）
# =========================
ROOT = "/data/user/jzt/crd/audioLLM/train_data/4_limit_VOXTRAL"

# segment 映射：长度（秒） -> 文件名中的“窗口”标记
SEGMENTS = [
    (5,  "05"),
    (10, "10"),
    (20, "20"),
    (30, "30"),
    (40, "40"),
    (50, "50"),
    (60, "60"),
    (90, "90"),
    (120,"120"),
]

# 两个子源（SONY / TAU）的命名模式
# 例如： sony.win05_earliest.csv, tau.win10_earliest.csv
def expected_csv_paths(root, win_code):
    return [
        os.path.join(root, f"sony.win{win_code}_earliest.csv"),
        os.path.join(root, f"tau.win{win_code}_earliest.csv"),
    ]

# =========================
# 读取并构造误差数据
# =========================
def load_errors_for_length(root, win_code):
    """
    返回某个长度（由 win_code 决定）下所有样本的
    signed_error 与 abs_error 的 ndarray（合并 sony 与 tau）。
    忽略 pred_start 缺失的条目。
    """
    dfs = []
    for p in expected_csv_paths(root, win_code):
        if os.path.exists(p):
            df = pd.read_csv(
                p,
                na_values=["", "nan", "NaN", "None"],
                keep_default_na=True
            )
            dfs.append(df)
    if not dfs:
        return np.array([]), np.array([])

    df_all = pd.concat(dfs, ignore_index=True)

    # 仅保留有 gt_start 与 pred_start 的行
    # （按你说明：pred_start 为空则不参与计算）
    df_all = df_all.dropna(subset=["gt_start", "pred_start"])

    # 转为浮点
    df_all["gt_start"] = pd.to_numeric(df_all["gt_start"], errors="coerce")
    df_all["pred_start"] = pd.to_numeric(df_all["pred_start"], errors="coerce")

    # 再次去除无法转数值的行
    df_all = df_all.dropna(subset=["gt_start", "pred_start"])

    signed = df_all["pred_start"].values - df_all["gt_start"].values
    absv   = np.abs(signed)

    return signed, absv

# 为每个长度收集误差分布
signed_errors_by_len = []
abs_errors_by_len    = []
segment_lengths      = []

for length, code in SEGMENTS:
    signed, absv = load_errors_for_length(ROOT, code)
    if signed.size > 0:
        segment_lengths.append(length)
        signed_errors_by_len.append(signed)
        abs_errors_by_len.append(absv)

# 若没有有效数据，提前退出
if len(segment_lengths) == 0:
    raise RuntimeError("未在指定路径读取到任何有效的 Voxtral 结果 CSV（pred_start 均为空或文件不存在）。")

# =========================
# 画双指标箱线图（每个长度两个箱体：|error| & signed error）
# 风格尽量与折线图保持一致（LaTeX, Times New Roman, 网格等）
# =========================
matplotlib.rc('text', usetex=True)
plt.rcParams.update({'lines.linewidth': 2})
plt.rcParams.update({'lines.markersize': 8})
plt.rcParams.update({'lines.markeredgewidth': 1})
plt.rcParams["font.family"] = "Times New Roman"

fig = plt.figure(figsize=(13, 7.5))
ax = plt.axes()

# 网格与刻度样式（尽量贴近你给的折线图）
ax.grid(which='major', axis='y', linestyle='-.')
ax.grid(which='minor', axis='both', linestyle='-.', alpha=0.2)
ax.tick_params(axis='x', which='minor', labelsize=30, length=14, direction='in', top=True)
ax.tick_params(axis='y', which='minor', labelsize=30, length=0, direction='in')

# 为了将两个箱体并排放置，计算位置
n = len(segment_lengths)
x_base = np.arange(n)  # 基础 x 位置：0,1,2,...
offset = 0.15          # 左右偏移
pos_abs   = x_base - offset  # 左：绝对误差
pos_signed= x_base + offset  # 右：有符号误差

# 画箱线图：绝对误差
bp_abs = ax.boxplot(
    abs_errors_by_len,
    positions=pos_abs,
    widths=0.25,
    showfliers=True,   # 可根据需要是否显示离群点
    patch_artist=True, # 允许填充
    medianprops=dict(linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    boxprops=dict(linewidth=1.5)
)

# 画箱线图：有符号误差
bp_signed = ax.boxplot(
    signed_errors_by_len,
    positions=pos_signed,
    widths=0.25,
    showfliers=True,
    patch_artist=True,
    medianprops=dict(linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    boxprops=dict(linewidth=1.5)
)

# 不指定具体颜色以保持中性（若需要可按期刊模板统一设置）
for patch in bp_abs['boxes']:
    patch.set_alpha(0.6)
for patch in bp_signed['boxes']:
    patch.set_alpha(0.6)

# X 轴刻度：显示为 “5s, 10s, …”
ax.set_xticks(x_base)
ax.set_xticklabels([f"{s}s" for s in segment_lengths], fontsize=30)

# Y 轴设置
ax.set_ylabel(r'\textbf{Error (s)}', fontsize=30)
plt.yticks(rotation=0, size=25)

# 画一条 y=0 的零基线，帮助解读 signed error 的提前/滞后
ax.axhline(0.0, color='k', linewidth=1.2, linestyle='--', alpha=0.6)

# 图例（自定义图例句柄）
legend_handles = [
    Patch(alpha=0.6, label=r'Absolute error ($|t_{\mathrm{pred}} - t_{\mathrm{gt}}|$)'),
    Patch(alpha=0.6, label=r'Signed error ($t_{\mathrm{pred}} - t_{\mathrm{gt}}$)'),
]
ax.legend(
    handles=legend_handles,
    fontsize=22,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.12),
    fancybox=True,
    shadow=True,
    ncol=2
)

# 边距与保存
plt.tight_layout(pad=5)
plt.tight_layout()

plt.savefig('fig_voxtral_box_dual.pdf', format='pdf')
plt.savefig('fig_voxtral_box_dual.png', format='png', dpi=300)

print("Saved: fig_voxtral_box_dual.pdf / fig_voxtral_box_dual.png")
