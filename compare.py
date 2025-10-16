import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# 1. 数据准备（保持不变）
data = {
    'Model': ['Rotate FCOS', 'R3Det', 'S²ANet', 'RoI Trans', 'Faster R-CNN', 'ORCNN'],
    'ResNet50_params': [31.92, 41.9, 38.6, 55.13, 41.14, 41.14],
    'ResNet50_map': [72.35, 73.36, 75.01, 74.61, 73.61, 75.87],
    'FI_MambaNet_params': [23.4, 29.25, 30.42, 47.13, 33.14, 33.14],
    'FI_MambaNet_map': [75.67, 77.13, 77.64, 77.95, 78.47, 80.2]
}
df = pd.DataFrame(data)

# 2. 创建画布（预留足够空间）
plt.figure(figsize=(12, 7))

# 3. 绘制散点、连线 + 在连线上标注1个模型名称
for i, row in df.iterrows():
    model = row['Model']  # 每个模型仅标注1次
    # ResNet50（圆形）和 FI-MambaNet（三角形）的坐标
    r50_p, r50_m = row['ResNet50_params'], row['ResNet50_map']
    fim_p, fim_m = row['FI_MambaNet_params'], row['FI_MambaNet_map']

    # 3.1 绘制散点（圆形=ResNet50，三角形=FI-MambaNet，大小s=120）
    plt.scatter(r50_p, r50_m, color=f'C{i}', marker='o', s=300)
    plt.scatter(fim_p, fim_m, color=f'C{i}', marker='^', s=300)

    # 3.2 绘制两点连线（虚线）
    plt.plot([r50_p, fim_p], [r50_m, fim_m], color=f'C{i}', linestyle='--', alpha=0.7)

    # 3.3 关键：在连线上标注模型名称（取连线中点，微调位置避免遮挡）
    # 计算连线中点坐标（横轴中点、纵轴中点）
    mid_p = (r50_p + fim_p) / 2  # 横轴中点
    mid_m = (r50_m + fim_m) / 2  # 纵轴中点
    # 在中点位置标注文字（向上偏移0.3，避免压在连线上）
    plt.text(mid_p, mid_m + 0.3, model, fontsize=15, color=f'C{i}',
             ha='center', va='bottom')  # ha/va 控制文字水平/垂直居中

# 4. 右上角框架说明图例（保持不变）
custom_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=15, label='ResNet50'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=15, label='FI-MambaNet')
]
plt.legend(handles=custom_legend, loc='upper right', fontsize=15)

# 5. 横轴设置（数值间隔更大，保持不变）
plt.xlim(20, 60)
plt.xticks(range(20, 61, 5))  # 刻度：20、30、40、50、60

# 6. 坐标轴与标题
plt.xlabel('Params (M)', fontsize=11)
plt.ylabel('mAP (%)', fontsize=11)


# 7. 网格（辅助阅读）
plt.grid(True, alpha=0.3)

# 8. 自动调整布局，避免文字截断
plt.tight_layout()
plt.show()