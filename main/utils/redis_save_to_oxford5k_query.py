import os
import shutil
import pickle

'''
将gnd文件中的查询数据集所指向的原始图片存入一个query文件夹当中
'''

# 配置路径（请根据实际情况修改）
pkl_file_path = '../data/oxford5k_answer/gnd_roxford5k.pkl'
raw_image_dir = '../data/oxford5k_raw'    # 原始图像目录
query_image_dir = '../data/oxford5k_query'  # 目标查询图像目录

# 确保目标目录存在
os.makedirs(query_image_dir, exist_ok=True)

try:
    # 加载pkl文件获取查询图像列表
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        qim_list = data.get("qimlist", [])  # 获取查询图像列表

    if not qim_list:
        print("警告：未从pkl文件中获取到查询图像列表")
        exit(1)

    # 处理每个查询图像
    success_count = 0
    for qim_name in qim_list:
        # 添加.jpg后缀
        qim_filename = f"{qim_name}.jpg"
        # 构建源文件和目标文件路径
        src_path = os.path.join(raw_image_dir, qim_filename)
        dest_path = os.path.join(query_image_dir, qim_filename)

        # 检查源文件是否存在
        if not os.path.exists(src_path):
            print(f"警告：原始图像 '{src_path}' 不存在，跳过处理")
            continue

        # 复制文件到目标目录
        shutil.copy2(src_path, dest_path)  # copy2会保留文件元数据
        success_count += 1
        print(f"已复制: {qim_filename}")

    print(f"\n处理完成 - 成功复制 {success_count}/{len(qim_list)} 个查询图像")

except FileNotFoundError as e:
    print(f"错误：文件未找到 - {e.filename}")
except PermissionError:
    print("错误：没有文件操作权限，请检查目录权限设置")
except Exception as e:
    print(f"发生错误：{e}")