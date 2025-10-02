from pymilvus import connections, Collection, utility
import os
from main.utils import get_ipadress
from pymilvus import MilvusClient
# 1. 连接到Milvus服务（根据实际情况修改参数）

host_ip = get_ipadress.get_host_ip()
print("ip = " + host_ip)
client = MilvusClient("http://" + host_ip + ":19530")

# 1. 连接到Milvus服务（根据实际情况修改参数）
connections.connect(
    alias="default",
    host=host_ip,  # Milvus服务地址
    port="19530"       # Milvus服务端口
)

# 2. 定义参数
collection_name = "oxford5k_raw_dinov3"
query_dir = "../data/oxford5k_query"
primary_key_field = "id"  # 请替换为你的collection实际的主键字段名（通常是"id"）

# 3. 检查collection是否存在
if not utility.has_collection(collection_name):
    raise ValueError(f"Collection {collection_name} 不存在")

# 4. 加载collection并获取对象
collection = Collection(collection_name)
collection.load()

# 5. 获取目标目录下的所有文件名（不含路径）
try:
    filenames = [f for f in os.listdir(query_dir) if os.path.isfile(os.path.join(query_dir, f))]
    if not filenames:
        print(f"目录 {query_dir} 中没有文件，无需删除")
        exit()
    print(f"共获取到 {len(filenames)} 个文件名，准备查询对应主键")
except FileNotFoundError:
    raise FileNotFoundError(f"目录 {query_dir} 不存在")

# 6. 查询文件名对应的主键（仅返回主键字段）
# 构建查询条件：image_name在文件名列表中
query_expr = f"image_name in {filenames}"
try:
    # 查询时只返回主键字段，减少数据传输
    result = collection.query(
        expr=query_expr,
        output_fields=[primary_key_field]  # 仅获取主键
    )
    if not result:
        print("未查询到匹配的主键，无需删除")
        exit()
    # 提取主键列表（结果是字典列表，需转换为纯主键值列表）
    primary_keys = [item[primary_key_field] for item in result]
    print(f"查询到 {len(primary_keys)} 个匹配的主键，准备删除")
except Exception as e:
    print(f"查询主键失败：{str(e)}")
    exit()

# 7. 根据主键批量删除数据
try:
    # 通过主键删除（主键需是列表格式）
    delete_result = collection.delete(expr=f"{primary_key_field} in {primary_keys}")
    print(f"删除成功，实际删除行数：{delete_result.delete_count}")
except Exception as e:
    print(f"删除数据失败：{str(e)}")

# 8. 释放资源并断开连接
collection.release()
connections.disconnect("default")