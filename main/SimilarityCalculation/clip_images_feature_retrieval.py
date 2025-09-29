from transformers import CLIPProcessor, CLIPModel  # 替换为CLIP的处理器和模型
from PIL import Image
from pymilvus import MilvusClient
import matplotlib.pyplot as plt
import torch
import os
from main.utils import get_ipadress

# 生成特征向量（适配CLIP原始维度）
def gen_image_features(processor, model, device, images):
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
    # 得到当前IP
    host_ip = get_ipadress.get_host_ip()
    # 创建Milvus客户端
    client = MilvusClient("http://"+host_ip+":19530")
    # 加载CLIP模型（使用openai的clip-vit-base-patch32）
    model_dir = "openai/clip-vit-base-patch32"  # CLIP模型路径
    processor = CLIPProcessor.from_pretrained(model_dir)  # CLIP处理器
    model = CLIPModel.from_pretrained(model_dir)  # CLIP模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 检索图像, 采用不在Milvus数据集中的图像
    image = Image.open("../data/oxford5k_query/ashmolean_000000.jpg")

    # 创建特征维度投影层（将CLIP输出维度映射到768维）
    # 先获取CLIP模型的实际输出维度
    with torch.no_grad():
        dummy_input = processor(images=Image.new('RGB', (224, 224)), return_tensors="pt").to(device)
        dummy_output = model.get_image_features(**dummy_input)
        clip_output_dim = dummy_output.shape[1]
    print(f"CLIP模型原始输出维度: {clip_output_dim}")


    # 提取特征向量
    features = gen_image_features(processor, model, device, image)

    # 特征召回（CLIP特征同样适合COSINE距离度量）
    limit_num = 10
    results = client.search(
        collection_name="oxford5k_raw_clip",  # 注意：需要确保该集合使用CLIP模型提取的特征
        data=[features],
        limit=limit_num,
        output_fields=["image_name"],
        search_params={
            "metric_type": "COSINE",  # CLIP特征推荐使用COSINE距离
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