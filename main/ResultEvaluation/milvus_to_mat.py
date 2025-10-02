import scipy.io as sio
import numpy as np
from mpmath import limit
from pymilvus import connections, Collection
from main.utils import get_ipadress
from pymilvus import MilvusClient

'''
将milvus中的特征数据写入.mat文件当中
'''

host_ip = get_ipadress.get_host_ip()
print("ip = " + host_ip)
client = MilvusClient("http://" + host_ip + ":19530")


def replace_q_x_with_milvus_data(mat_file_path, output_file_path,
                                 milvus_host=host_ip, milvus_port='19530',
                                 collection_name_q='oxford5k_query_dinov3',
                                 collection_name_x='oxford5k_raw_dinov3'):
    """
    从Milvus读取数据替换.mat文件中的Q和X

    参数:
    - mat_file_path: 原始.mat文件路径
    - output_file_path: 输出文件路径
    - milvus_host: Milvus服务器地址
    - milvus_port: Milvus端口
    - collection_name_q: 存储Q向量的集合名
    - collection_name_x: 存储X向量的集合名
    """

    # 1. 连接Milvus
    print("连接Milvus...")
    connections.connect(host=milvus_host, port=milvus_port)

    # 2. 从Milvus读取Q数据
    print(f"从集合 {collection_name_q} 读取Q数据...")
    collection_q = Collection(collection_name_q)
    collection_q.load()

    # 查询所有Q数据（根据您的实际查询条件调整）
    results_q = collection_q.query(
        expr="",  # 查询所有，或根据您的条件
        output_fields=["vector"],  # 根据您的schema调整字段名
        limit=6000
    )
    Q_new = np.array([item["vector"] for item in results_q])

    # 3. 从Milvus读取X数据
    print(f"从集合 {collection_name_x} 读取X数据...")
    collection_x = Collection(collection_name_x)
    collection_x.load()

    # 查询所有X数据
    results_x = collection_x.query(
        expr="",  # 查询所有，或根据您的条件
        output_fields=["vector"],  # 根据您的schema调整字段名
        limit=6000
    )
    X_new = np.array([item["vector"] for item in results_x])

    # 4. 加载原始.mat文件
    print("加载原始.mat文件...")
    data = sio.loadmat(mat_file_path)

    # 5. 替换Q和X数据
    print("替换Q和X数据...")
    data['Q'] = Q_new
    data['X'] = X_new

    # 6. 保存修改后的文件
    print(f"保存到 {output_file_path}...")
    sio.savemat(output_file_path, data)

    # 7. 释放Milvus集合
    collection_q.release()
    collection_x.release()

    print("完成！")

    return Q_new.shape, X_new.shape


# 使用示例
if __name__ == "__main__":
    # 替换为您的实际参数
    input_file = "../data/features/roxford5k_resnet_rsfm120k_gem.mat"
    output_file = "../data/features/roxford5k_resnet_rsfm120k_gem_modified.mat"

    q_shape, x_shape = replace_q_x_with_milvus_data(
        mat_file_path=input_file,
        output_file_path=output_file,
        milvus_host=host_ip,  # 您的Milvus地址
        milvus_port='19530',  # 您的Milvus端口
        collection_name_q='oxford5k_query_dinov3',  # 存储Q的集合名
        collection_name_x='oxford5k_raw_dinov3'  # 存储X的集合名
    )

    print(f"新的Q形状: {q_shape}")
    print(f"新的X形状: {x_shape}")