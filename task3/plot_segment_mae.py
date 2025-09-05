#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制不同窗口大小 & segment_index 的“缩放误差”曲线：
先对每个样本计算绝对误差 |pred - gt|，再除以窗口长度 window_sec，
然后按 (window_sec, segment_index) 分组求平均并作图。

输入 CSV 格式（需包含至少这些列）：
audio_path,class,class_name,gt_onset_sec,gt_offset_sec,window_sec,segment_index,placement,source,pred_onset_sec,raw_model_output
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="绘制不同窗口大小下的分段缩放误差曲线（|pred-gt|/window 再平均）")
    parser.add_argument("csv_file", help="预测结果 CSV")
    parser.add_argument("--out", default="segment_mae_scaled.png", help="输出图像文件")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    required = {"gt_onset_sec", "pred_onset_sec", "window_sec", "segment_index"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列: {required}")

    # 去掉没有预测的样本
    df = df.dropna(subset=["pred_onset_sec"]).copy()

    # 每个样本的绝对误差 / 窗口长度
    df["scaled_err"] = (df["pred_onset_sec"] - df["gt_onset_sec"]).abs() / df["window_sec"]

    # 按 window_sec 和 segment_index 聚合（对 scaled_err 求平均）
    stats = (
        df.groupby(["window_sec", "segment_index"])["scaled_err"]
        .mean()
        .reset_index()
        .sort_values(["window_sec", "segment_index"])
    )

    # 画图
    plt.figure(figsize=(8, 5))
    for window_sec, g in stats.groupby("window_sec"):
        plt.plot(
            g["segment_index"], g["scaled_err"],
            marker="o", label=f"{int(window_sec)}s"
        )

    plt.xlabel("Segment Index (1~5)")
    plt.ylabel("Mean(|pred - gt| / window)")
    plt.title("Scaled Prediction Error by Segment Position and Window Size")
    plt.legend(title="Window Size")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(0, 1)   # 固定纵轴范围 0~5
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"[DONE] 图像已保存到 {args.out}")

if __name__ == "__main__":
    main()
