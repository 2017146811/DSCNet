import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 设置全局英文字体为Times New Roman（不加粗）
plt.rcParams["font.family"] = ["Times New Roman", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 读取Excel数据
df = pd.read_excel('hrsc.xlsx')

# 2. 提取数据
methods = df['Method'].tolist()
mAP07 = df['mAP (07)'].tolist()
mAP12 = df['mAP (12)'].tolist()

# 3. 计算合适的纵轴起点
all_values = mAP07 + mAP12
min_val = min(all_values)
y_min = np.floor(min_val * 0.95)
y_max = np.ceil(max(all_values) * 1.05)

# 4. 设置柱状图参数
x = np.arange(len(methods))
width = 0.35

# 5. 创建画布
plt.figure(figsize=(12, 7))

# 6. 绘制柱状图
bars1 = plt.bar(x - width/2, mAP07, width, label='mAP (07)', color='skyblue', edgecolor='black')
bars2 = plt.bar(x + width/2, mAP12, width, label='mAP (12)', color='coral', edgecolor='black')

# 7. 设置纵轴范围
plt.ylim(y_min, y_max)

# 8. 添加标签和标题（无需重复设置fontfamily，全局已生效）
# plt.xlabel('Methods', fontsize=12)
plt.ylabel('mAP (%)', fontsize=12)
plt.title('Comparison of mAP (07) and mAP (12) for Different Methods', fontsize=10, pad=2)
plt.xticks(x, methods, rotation=45, ha='right', fontsize=10)

# 9. 设置图例字体（使用FontProperties兼容方式）
legend_font = FontProperties(family='Times New Roman', size=10)
plt.legend(prop=legend_font)

# 10. 在柱状图上标注数值
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9  # 全局字体已设置，无需重复指定
        )

add_labels(bars1)
add_labels(bars2)

# 11. 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 12. 调整布局并显示
plt.tight_layout()
plt.show()
