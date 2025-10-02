from transformers import AutoImageProcessor, AutoModel
from pymilvus import MilvusClient
import torch
import os
import json  # 导入json模块
from main.utils import get_ipadress
'''
从milvus查询集中 到raw集里面去进行召回特征 最后生成一个json文件 
'''

# 生成特征向量函数（保留但批量查询时不使用）
def gen_image_features(processor, model, device, images):
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(** inputs)
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
        return features


def main():
    # 获取当前IP并创建Milvus客户端
    host_ip = get_ipadress.get_host_ip()
    client = MilvusClient("http://" + host_ip + ":19530")

    # 加载DINOv3模型（纯查询可注释）
    '''model_dir = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.to(device)'''

    # 从Milvus查询集合批量读取所有特征
    query_collection = "oxford5k_query_dinov3"
    query_entities = client.query(
        collection_name=query_collection,
        filter="",  # 查询所有实体
        output_fields=["image_name", "vector"],  # 假设包含图像名和特征向量字段
        limit=70
    )

    # 对每个查询特征进行召回并在控制台输出结果
    limit_num = 10
    target_collection = "oxford5k_raw_dinov3"
    all_results = {}  # 存储所有查询结果

    for entity in query_entities:
        query_image_name = entity["image_name"]
        query_vector = entity["vector"]

        # 执行特征召回
        results = client.search(
            collection_name=target_collection,
            data=[query_vector],
            limit=limit_num,
            output_fields=["image_name"],
            search_params={"metric_type": "L2", "params": {}}
        )

        # 处理并输出当前查询结果
        result_image_names = []
        print(f"\n===== 查询图像: {query_image_name} 的召回结果 =====")
        for i, res in enumerate(results[0], 1):
            image_name = res["entity"]["image_name"]
            distance = res["distance"]
            image_name_without_suffix = image_name.replace(".jpg", "")
            result_image_names.append(image_name_without_suffix)
            print(f"{i}. {image_name} (距离: {distance:.4f})")

        print(f"\n排序结果文件名列表: {result_image_names}")
        # 关键修改：用去除.jpg的查询图像名作为键
        query_key = query_image_name.replace(".jpg", "")
        all_results[query_key] = result_image_names

    # 保存到JSON文件
    with open("retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n结果已保存到 retrieval_results.json 文件")


if __name__ == '__main__':
    main()