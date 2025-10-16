import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.cm as cm

# 读取 Excel 表格
df = pd.read_excel("ap_of_ABC.xlsx")  # 替换为你的路径
df = df.dropna(axis=1, how='all')  # 删除空列
methods = df["Method"]
categories = df.columns[1:]

# 创建保存图像的目录
temp_dir = "temp_category_plots"
os.makedirs(temp_dir, exist_ok=True)

# 使用颜色映射（tab20）给15个类别分配不同颜色
cmap = cm.get_cmap('tab20', len(categories))
color_list = [cmap(i) for i in range(len(categories))]

# 逐个类别画图并保存
for idx, category in enumerate(categories):
    plt.figure(figsize=(6, 4))
    y = df[category]
    ymin, ymax = y.min(), y.max()
    padding = (ymax - ymin) * 0.1 if ymax > ymin else 1  # 至少加1防止压扁

    plt.plot(methods, y, color=color_list[idx], marker='o', linewidth=2)
    plt.title(f"mAP50 of {category}")
    plt.xlabel("Methods")
    plt.ylabel("mAP50 (%)")
    plt.ylim(ymin - padding, ymax + padding)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(f"{temp_dir}/{category}.png", dpi=200)
    plt.close()

# 合并为5行3列的子图
fig, axs = plt.subplots(5, 3, figsize=(18, 20))
axs = axs.flatten()

for i, category in enumerate(categories):
    img_path = f"{temp_dir}/{category}.png"
    img = Image.open(img_path)
    axs[i].imshow(img)
    axs[i].axis('off')

# 隐藏多余子图（如果少于15张图）
for j in range(len(categories), len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig("combined_category_plots_5x3.png", dpi=300)
plt.show()
