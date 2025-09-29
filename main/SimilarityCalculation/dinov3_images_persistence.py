from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
import torch
import os
from main.utils import get_ipadress


# 生成特征向量（dinov3模型特征提取）
def gen_image_features(processor, model, device, image):
    with torch.no_grad():
        # 处理图像输入
        inputs = processor(images=image, return_tensors="pt").to(device)
        # DINOv3通过特征提取获取图像特征
        outputs = model(**inputs)
        # 获取最后一层特征并进行平均池化
        last_hidden_state = outputs.last_hidden_state
        image_features = torch.mean(last_hidden_state, dim=1)
        # L2归一化提升检索效果
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # 转换为numpy数组并确保类型为float32（Milvus要求）
        return image_features.squeeze().cpu().numpy().astype('float32')


def main():
    host_ip = get_ipadress.get_host_ip()
    # 创建Milvus客户端
    client = MilvusClient(f"http://{host_ip}:19530")
    # Milvus collection名
    collection_name = "oxford5k_raw"

    # 检查并创建Milvus集合（dinov3-vitb16的特征维度为768）
    if not client.has_collection(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema={
                "fields": [
                    {"name": "id", "type": DataType.INT64, "is_primary": True, "auto_id": True},
                    {"name": "vector", "type": DataType.FLOAT_VECTOR, "dim": 768},
                    {"name": "image_name", "type": DataType.VARCHAR, "max_length": 256}
                ]
            }
        )
        print(f"已创建Milvus集合: {collection_name}")
    else:
        print(f"Milvus集合已存在: {collection_name}")

    # 加载dinov3模型
    model_dir = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # 明确使用dinov3模型
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"使用设备: {device}")

    # 读取数据集
    dataset_path = "../data/oxford5k_raw"
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        return

    # 批量处理图像并插入Milvus
    for image_name in tqdm(os.listdir(dataset_path), desc="处理图像"):
        image_path = os.path.join(dataset_path, image_name)
        # 过滤非图像文件
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        try:
            image = Image.open(image_path).convert('RGB')  # 统一转为RGB格式
            # 提取特征向量
            features = gen_image_features(processor, model, device, image)
            # 存入Milvus
            client.insert(
                collection_name=collection_name,
                data={
                    "vector": features.tolist(),  # 转换为列表格式
                    "image_name": image_name
                }
            )
        except Exception as e:
            print(f"处理图像 {image_name} 失败: {str(e)}")
            continue

    print("特征提取与存储完成")


if __name__ == '__main__':
    main()