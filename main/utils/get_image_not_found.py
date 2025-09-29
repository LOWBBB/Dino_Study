import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 原始文件名列表
ranked_list = \
    {'ashmolean_000036', 'ashmolean_000005', 'radcliffe_camera_000478', 'ashmolean_000052'}

# 添加.jpg后缀
image_files = [f"{name}.jpg" for name in ranked_list]

# 图片所在文件夹路径（请根据实际情况修改）
image_folder = "../data/oxford5k_raw"

# 收集所有图片
images = []
for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    if not os.path.exists(img_path):
        print(f"警告：图片文件不存在 - {img_path}")
        continue
    try:
        with Image.open(img_path) as img:
            # 转换为RGB模式以统一格式
            img = img.convert('RGB')
            images.append((img_file, img))
    except Exception as e:
        print(f"读取图片 {img_file} 时出错: {e}")

if not images:
    print("没有找到任何有效图片，程序退出")
    exit()

# 计算排列方式（尽量接近正方形）
num_images = len(images)
rows = int(np.ceil(np.sqrt(num_images)))
cols = int(np.ceil(num_images / rows))

# 创建画布
plt.figure(figsize=(cols * 4, rows * 4))  # 每个子图大约4x4英寸

# 显示图片
for i, (img_name, img) in enumerate(images, 1):
    plt.subplot(rows, cols, i)
    plt.imshow(np.array(img))
    plt.title(os.path.splitext(img_name)[0], fontsize=8)  # 显示不带后缀的文件名
    plt.axis('off')  # 关闭坐标轴

plt.tight_layout()  # 调整布局
plt.show()