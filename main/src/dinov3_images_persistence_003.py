import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
import torch
import os
import yaml
import importlib
import torch.nn.functional as F
from main.utils import get_gnd_param

'''
2025年10月4日15:20:27
在002的基础上，用gnd文件给的imlist和qimlist顺序，将提取出的特征存入数据库。
'''

# 加载配置文件
def load_config(config_path="../config/config.yml"):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 解析milvus主机函数（修正版）
    try:
        # 导入完整模块路径
        host_module = importlib.import_module(config['milvus']['host_module'])
        # 获取模块中的函数
        host_func = getattr(host_module, config['milvus']['host_func'])
        config['milvus']['host_func'] = host_func
    except ImportError as e:
        raise ValueError(f"无法导入主机函数模块: {config['milvus']['host_module']}, 错误: {str(e)}")
    except AttributeError as e:
        raise ValueError(
            f"模块 {config['milvus']['host_module']} 中没有找到函数 {config['milvus']['host_func']}, 错误: {str(e)}")

    return config


# 加载配置
CONFIG = load_config()


# 批量生成特征向量（dinov3模型特征提取）
def gen_batch_image_features(processor, model, device, images):
    with torch.no_grad():
        # 处理批量图像输入
        inputs = processor(images=images, return_tensors="pt").to(device)
        # DINOv3通过特征提取获取图像特征
        outputs = model(**inputs)
        # 使用[CLS] token的输出作为图像特征
        cls_feat = outputs.last_hidden_state[:, 0, :]
        # 增加L2归一化（在特征转换为numpy前执行）
        normalized_feat = torch.nn.functional.normalize(cls_feat, p=2, dim=1)
        # 转换为numpy数组并确保类型为float32（Milvus要求）
        features = normalized_feat.cpu().numpy().astype('float32')
        return features


def process_image_list(image_name_list):
    """
    按照给定的图像名称列表顺序进行特征提取并持久化到Milvus
    :param image_name_list: 图像名称列表（不含扩展名），如['all_souls_000013', 'all_souls_000026']
    """
    host_ip = CONFIG["milvus"]["host_func"]()
    # 创建Milvus客户端
    client = MilvusClient(f"http://{host_ip}:{CONFIG['milvus']['port']}")
    collection_name = CONFIG["milvus"]["collection"]

    # 检查并创建Milvus集合
    if not client.has_collection(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema={
                "fields": [
                    {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": True},
                    {"name": "vector", "type": DataType.FLOAT_VECTOR, "dim": CONFIG["model"]["feature_dim"]},
                    {"name": "image_name", "type": DataType.VARCHAR, "max_length": 256}
                ]
            }
        )
        print(f"已创建Milvus集合: {collection_name}")
    else:
        print(f"Milvus集合已存在: {collection_name}")

    # 加载dinov3模型
    processor = AutoImageProcessor.from_pretrained(CONFIG["model"]["dir"])
    model = AutoModel.from_pretrained(CONFIG["model"]["dir"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"使用设备: {device}")

    # 读取数据集路径
    dataset_path = CONFIG["data"]["dataset_path"]
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return

    # 为列表中的每个图像名称查找实际文件（带扩展名）
    valid_image_files = []
    for base_name in image_name_list:
        found = False
        # 尝试所有可能的图像扩展名
        for ext in CONFIG["data"]["image_extensions"]:
            full_name = f"{base_name}{ext}"
            full_path = os.path.join(dataset_path, full_name)
            if os.path.exists(full_path):
                valid_image_files.append(full_name)
                found = True
                break
        if not found:
            print(f"警告: 未找到图像文件 {base_name}（尝试了所有扩展名）")

    # 批量处理参数
    batch_size = CONFIG["processing"]["batch_size"]
    total_batches = (len(valid_image_files) + batch_size - 1) // batch_size

    # 批量处理图像并插入Milvus（严格按照列表顺序）
    for batch_idx in tqdm(range(total_batches), desc="处理图像批次"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(valid_image_files))
        batch_files = valid_image_files[start_idx:end_idx]

        batch_images = []
        valid_names = []  # 存储成功加载的图像名称

        # 加载当前批次的图像
        for image_name in batch_files:
            image_path = os.path.join(dataset_path, image_name)
            try:
                image = Image.open(image_path).convert('RGB')  # 统一转为RGB格式
                batch_images.append(image)
                valid_names.append(image_name)
            except Exception as e:
                print(f"加载图像 {image_name} 失败: {str(e)}")
                continue

        if not batch_images:  # 跳过空批次
            continue

        # 批量提取特征
        try:
            features = gen_batch_image_features(processor, model, device, batch_images)
        except Exception as e:
            print(f"批次 {batch_idx} 特征提取失败: {str(e)}")
            continue

        # 批量准备插入数据
        insert_data = [
            {"vector": feat.tolist(), "image_name": name}
            for feat, name in zip(features, valid_names)
        ]

        # 批量插入Milvus
        try:
            client.insert(
                collection_name=collection_name,
                data=insert_data
            )
        except Exception as e:
            print(f"批次 {batch_idx} 插入Milvus失败: {str(e)}")
            continue

    print("特征提取与存储完成")


if __name__ == '__main__':
    # 保证图片文件的读入顺序同gnd文件中一致
    data = get_gnd_param.inspect_pkl('../data/datasets/roxford5k/gnd_roxford5k.pkl')
    imlist = data.get('imlist')
    qimlist = data.get('qimlist')
    # 处理指定的图像列表
    process_image_list(qimlist)





