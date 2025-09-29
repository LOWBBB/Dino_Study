import pymilvus
from pymilvus import MilvusClient
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType


print("pymilvus 版本:", pymilvus.__version__)

client = MilvusClient("http://localhost:19530",
                        token="root:Milvus",
                        db_name="default")

# 确保已连接
connections.connect("default", host="localhost", port="19530")

# 定义集合结构
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2)
]
schema = CollectionSchema(fields, "测试连接的集合")

# 创建集合（若连接失败，会抛出异常）
collection = Collection(name="test_connection", schema=schema)
print("集合创建成功，连接正常！")

# 清理测试数据（可选）
# collection.drop()