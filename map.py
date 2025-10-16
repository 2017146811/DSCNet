import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel("map.xlsx")  # 替换为你的实际路径

# 提取模型名称和最后一列 mAP（或 mMP）
methods = df["Method"]
mAP_col = df.columns[-1]
mAPs = df[mAP_col]

# 排序（从小到大）
sorted_indices = mAPs.argsort()
methods_sorted = methods.iloc[sorted_indices].reset_index(drop=True)
mAPs_sorted = mAPs.iloc[sorted_indices].reset_index(drop=True)

# 颜色：默认蓝色，'Ours' 改为红色
colors = ['skyblue' if name != 'Ours' else 'orange' for name in methods_sorted]

# 绘图
plt.figure(figsize=(12, 6))
bars = plt.bar(methods_sorted, mAPs_sorted, color=colors, edgecolor='black')

# 添加柱顶数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.2, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

# 设置图形属性
plt.title(f"mAP of all class")
plt.ylabel(mAP_col)
plt.xlabel("Methods")
plt.ylabel("mAP50 (%)")
plt.ylim(70, max(mAPs_sorted) + 2)  # 从70开始
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("map_comparison_sorted.png", dpi=300)
plt.show()
