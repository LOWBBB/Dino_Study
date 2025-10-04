from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
import torch
import os
from main.utils import get_ipadress

# 配置参数集中管理
CONFIG = {
    # Milvus相关配置
    "milvus_collection": "oxford5k_query_dinov3",
    "milvus_host_func": get_ipadress.get_host_ip,
    "milvus_port": "19530",

    # 模型相关配置
    "model_dir": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "feature_dim": 768,  # dinov3-vitb16的特征维度

    # 数据相关配置
    "dataset_path": "../data/oxford5k_query",
    "image_extensions": ('.png', '.jpg', '.jpeg', '.bmp', '.gif'),

    # 处理参数配置
    "batch_size": 32
}


# 批量生成特征向量（dinov3模型特征提取）
def gen_batch_image_features(processor, model, device, images):
    with torch.no_grad():
        # 处理批量图像输入
        inputs = processor(images=images, return_tensors="pt").to(device)
        # DINOv3通过特征提取获取图像特征
        outputs = model(**inputs)
        # 使用[CLS] token的输出作为图像特征，并归一化
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
        # 转换为numpy数组并确保类型为float32（Milvus要求）
        return features


def main():
    host_ip = CONFIG["milvus_host_func"]()
    # 创建Milvus客户端
    client = MilvusClient(f"http://{host_ip}:{CONFIG['milvus_port']}")
    collection_name = CONFIG["milvus_collection"]

    # 检查并创建Milvus集合
    if not client.has_collection(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema={
                "fields": [
                    {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": True},
                    {"name": "vector", "type": DataType.FLOAT_VECTOR, "dim": CONFIG["feature_dim"]},
                    {"name": "image_name", "type": DataType.VARCHAR, "max_length": 256}
                ]
            }
        )
        print(f"已创建Milvus集合: {collection_name}")
    else:
        print(f"Milvus集合已存在: {collection_name}")

    # 加载dinov3模型
    processor = AutoImageProcessor.from_pretrained(CONFIG["model_dir"])
    model = AutoModel.from_pretrained(CONFIG["model_dir"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"使用设备: {device}")

    # 读取数据集
    dataset_path = CONFIG["dataset_path"]
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return

    # 获取所有图像文件路径
    image_files = []
    for image_name in os.listdir(dataset_path):
        if image_name.lower().endswith(CONFIG["image_extensions"]):
            image_files.append(image_name)

    # 批量处理参数
    batch_size = CONFIG["batch_size"]
    total_batches = (len(image_files) + batch_size - 1) // batch_size

    # 批量处理图像并插入Milvus
    for batch_idx in tqdm(range(total_batches), desc="处理图像批次"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]

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
    main()