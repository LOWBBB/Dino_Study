from transformers import AutoImageProcessor, AutoModel  # DINOv3使用的处理器和模型
from PIL import Image
from pymilvus import MilvusClient
import matplotlib.pyplot as plt
import torch
import os
from main.utils import get_ipadress

# 生成特征向量（适配DINOv3模型）
def gen_image_features(processor, model, device, images):
    with torch.no_grad():
        # 处理批量图像输入
        inputs = processor(images=images, return_tensors="pt").to(device)
        # DINOv3通过特征提取获取图像特征
        outputs = model(**inputs)
        # 使用[CLS] token的输出作为图像特征，并归一化
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # 转换为numpy数组并确保类型为float32（Milvus要求）
        return features.astype('float32')

def main():
    # 得到当前IP
    host_ip = get_ipadress.get_host_ip()
    # 创建Milvus客户端
    client = MilvusClient("http://"+host_ip+":19530")
    # 加载DINOv3模型（使用facebook的dinov3-vitb16-pretrain-lvd1689m）
    model_dir = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # DINOv3模型路径
    processor = AutoImageProcessor.from_pretrained(model_dir)  # DINOv3处理器
    model = AutoModel.from_pretrained(model_dir)  # DINOv3模型

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 检索图像, 采用不在Milvus数据集中的图像
    image = Image.open("../data/oxford5k_query/ashmolean_000000.jpg")
    # 提取特征向量
    features = gen_image_features(processor, model, device, [image])
    print("特征类型:", features.dtype)  # 应输出 float32
    # 特征召回（保持不变）
    limit_num = 25
    results = client.search(
        collection_name="oxford5k_raw_dinov3",  # 注意：需要确保该集合使用相同模型提取的特征
        data=features,
        limit=limit_num,
        output_fields=["image_name"],
        search_params={
            "metric_type": "L2",  # DINOv3特征适合用L2距离
            "params": {}
        }
    )
    result_image_names = []
    plt.figure(figsize=(20, 4))  # 调整图像显示尺寸
    plt.axis('off')
    for i, res in enumerate(results[0]):
        image_name = res["entity"]["image_name"]
        image_name_without_suffix = image_name.replace(".jpg", "")
        image_path = os.path.join("../data/oxford5k_raw", image_name)
        print(f"{image_name} (距离: {res['distance']:.4f})")
        result_image_names.append(image_name_without_suffix)
        image = Image.open(image_path)
        plt.subplot(1, limit_num, (i+1))
        plt.imshow(image)
        plt.axis('off')  # 关闭子图坐标轴

    print("排序结果文件名列表:")
    print(result_image_names)
    plt.tight_layout()  # 调整布局
    plt.show()

if __name__ == '__main__':
    main()