#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析不同窗口 & 不同 segment_index 的预测偏差（平均绝对误差，秒）。
输入 CSV 格式：
audio_path,class,class_name,gt_onset_sec,gt_offset_sec,window_sec,segment_index,placement,source,pred_onset_sec,raw_model_output
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="统计不同窗口和分段的预测绝对误差均值")
    parser.add_argument("csv_file", help="预测结果 CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    required = {"gt_onset_sec", "pred_onset_sec", "window_sec", "segment_index"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 必须包含列: {required}")

    # 去掉缺失预测的样本
    df = df.dropna(subset=["pred_onset_sec"]).copy()
    df["abs_err"] = (df["pred_onset_sec"] - df["gt_onset_sec"]).abs()

    # 按 window_sec & segment_index 分组
    stats = (
        df.groupby(["window_sec", "segment_index"])["abs_err"]
        .mean()
        .reset_index()
        .sort_values(["window_sec", "segment_index"])
    )

    print("=== 不同窗口 & 分段的平均绝对误差（秒） ===")
    for _, row in stats.iterrows():
        print(f"窗口 {int(row['window_sec']):2d}s | 段 {int(row['segment_index'])}: MAE={row['abs_err']:.3f} s")

if __name__ == "__main__":
    main()
