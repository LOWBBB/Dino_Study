from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pymilvus import MilvusClient, DataType, IndexType
from tqdm import tqdm
import torch
import os
import numpy as np
from main.utils import get_ipadress
from torchvision import transforms


# 增强的图像预处理（提升特征质量）
def get_image_transforms():
    """创建更细致的图像预处理管道，提升特征一致性"""
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 固定尺寸，确保输入一致性
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP默认均值
            std=[0.26862954, 0.26130258, 0.27577711]  # CLIP默认标准差
        )
    ])


# 生成特征向量（适配CLIP原始维度）
def gen_image_features(processor, model, device, image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        # 获取CLIP原始图像特征（不经过投影层）
        outputs = model.get_image_features(**inputs)
        # L2归一化（CLIP特征标准处理）
        image_features = outputs / outputs.norm(dim=-1, keepdim=True)
        # 微小扰动增强鲁棒性（不改变维度）
        noise = torch.randn_like(image_features) * 1e-5
        image_features = (image_features + noise).clamp(-1.0, 1.0)
        # 转换为numpy数组（保持原始维度）
        return image_features.squeeze().cpu().numpy().astype('float32')


# 批量特征提取（适配CLIP原始维度）
def batch_gen_image_features(processor, model, device, images):
    """批量处理图像，使用CLIP原始特征维度"""
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model.get_image_features(** inputs)
        # L2归一化
        image_features = outputs / outputs.norm(dim=-1, keepdim=True)
        # 批量归一化确保分布一致性（不改变维度）
        mean = torch.mean(image_features, dim=0, keepdim=True)
        std = torch.std(image_features, dim=0, keepdim=True) + 1e-8
        image_features = (image_features - mean) / std
        # 打印CLIP原始特征维度
        print(f"CLIP原始特征维度: {image_features.shape[1]}")
        return image_features.cpu().numpy().astype('float32')


def main():
    host_ip = get_ipadress.get_host_ip()
    client = MilvusClient(f"http://{host_ip}:19530")
    collection_name = "oxford5k_raw_clip"
    batch_size = 32  # 批量处理大小

    # 加载CLIP模型并获取原始输出维度
    model_dir = "openai/clip-vit-base-patch32"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 获取CLIP模型的原始输出维度（动态适配）
    with torch.no_grad():
        dummy_input = processor(images=Image.new('RGB', (224, 224)), return_tensors="pt").to(device)
        dummy_output = model.get_image_features(**dummy_input)
        clip_output_dim = dummy_output.shape[1]  # 对于vit-base-patch32，此处应为768
    print(f"CLIP模型原始输出维度: {clip_output_dim}")

    # 检查并创建适配CLIP维度的Milvus集合
    if not client.has_collection(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema={
                "fields": [
                    {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": True},
                    {"name": "vector", "type": DataType.FLOAT_VECTOR, "dim": clip_output_dim},  # 使用CLIP原始维度
                    {"name": "image_name", "type": DataType.VARCHAR, "max_length": 256},
                ]
            }
        )
        print(f"已创建适配CLIP维度的Milvus集合: {collection_name}")

        # 创建索引（距离度量保持COSINE，适配归一化特征）
        client.create_index(
            collection_name=collection_name,
            field_name="vector",
            index_params={
                "index_type": IndexType.HNSW,
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200}
            }
        )
        print("已创建优化的HNSW索引")
    else:
        print(f"Milvus集合已存在: {collection_name}")

    # 启用混合精度推理
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    print(f"使用设备: {device}")

    # 读取数据集
    dataset_path = "../data/oxford5k_raw"
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return

    # 收集所有图像路径
    image_paths = []
    for image_name in os.listdir(dataset_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(dataset_path, image_name))

    # 批量处理图像
    print(f"开始批量处理 {len(image_paths)} 张图像...")
    for i in tqdm(range(0, len(image_paths), batch_size), desc="批量处理图像"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_names = []

        # 加载批量图像
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
                batch_names.append(os.path.basename(img_path))
            except Exception as e:
                print(f"加载图像 {os.path.basename(img_path)} 失败: {str(e)}")
                continue

        if not batch_images:
            continue

        # 批量提取特征（使用CLIP原始维度，无投影层）
        try:
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                features = batch_gen_image_features(processor, model, device, batch_images)  # 移除projection参数

            # 准备插入数据（特征维度为CLIP原始维度）
            insert_data = []
            for feat, name in zip(features, batch_names):
                insert_data.append({
                    "vector": feat.tolist(),
                    "image_name": name
                })

            # 批量插入Milvus
            client.insert(
                collection_name=collection_name,
                data=insert_data
            )
        except Exception as e:
            print(f"处理批量图像失败: {str(e)}")
            continue

    print("特征提取与存储完成")

    # 优化集合
    client.compact(collection_name=collection_name)
    print("集合优化完成")


if __name__ == '__main__':
    main()