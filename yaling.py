# -*- coding: utf-8 -*-
"""
绘制 Params vs mAP 的对比散点连线图
ResNet50 用方形，FI-MambaNet 用圆形
每个检测框架的两点用虚线连接，并标注 ΔmAP
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_param_map_plot(file_path):
    # --- 1. 读取数据 ---
    df = pd.read_excel(file_path)

    # 提取数据
    frameworks = df['Frameworks'].tolist()
    res_map = df['map'].iloc[:, 0].to_numpy()        # ResNet50 map
    res_params = df['p'].iloc[:, 0].to_numpy()       # ResNet50 Params
    new_map = df['map'].iloc[:, 1].to_numpy()        # FI-MambaNet map
    new_params = df['p'].iloc[:, 1].to_numpy()       # FI-MambaNet Params

    # --- 2. 定义颜色 ---
    colors = ['#22c55e', '#facc15', '#3b82f6', '#a855f7', '#eab308', '#ef4444']
    if len(frameworks) > len(colors):
        colors = colors * (len(frameworks) // len(colors) + 1)

    # --- 3. 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    for i, fw in enumerate(frameworks):
        # 画连线（虚线）
        ax.plot([res_params[i], new_params[i]], [res_map[i], new_map[i]],
                linestyle='--', color=colors[i], alpha=0.7)

        # 画 ResNet50 方形
        ax.scatter(res_params[i], res_map[i], marker='s', s=120, color=colors[i], edgecolor='black', zorder=3)

        # 画 FI-MambaNet 圆形
        ax.scatter(new_params[i], new_map[i], marker='o', s=120, color=colors[i], edgecolor='black', zorder=3)

        # 标注 ResNet50 值
        ax.text(res_params[i], res_map[i] - 0.5,
                f"{res_map[i]:.2f}", ha='center', va='top', fontsize=9, color=colors[i])

        # 标注 FI-MambaNet 值
        ax.text(new_params[i], new_map[i] + 0.5,
                f"{new_map[i]:.2f}", ha='center', va='bottom', fontsize=9, color=colors[i])

        # 标注 ΔmAP
        delta = new_map[i] - res_map[i]
        ax.text((res_params[i] + new_params[i]) / 2,
                (res_map[i] + new_map[i]) / 2,
                f"(+{delta:.2f})", ha='center', va='center', fontsize=9, color=colors[i])

    # --- 4. 设置坐标轴 ---
    ax.set_xlabel("Params (M)", fontsize=12)
    ax.set_ylabel("mAP (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- 5. 图例 ---
    ax.scatter([], [], marker='s', s=100, color='gray', edgecolor='black', label='ResNet50')
    ax.scatter([], [], marker='o', s=100, color='gray', edgecolor='black', label='FI-MambaNet')
    ax.legend(fontsize=11)

    plt.tight_layout()
    output_filename = 'params_vs_map.png'
    plt.savefig(output_filename, dpi=300)
    print(f"图表已成功保存为 '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    excel_file_name = 'compare.xlsx'
    create_param_map_plot(excel_file_name)
