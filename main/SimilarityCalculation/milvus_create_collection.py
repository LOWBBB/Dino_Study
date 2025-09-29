from pymilvus import MilvusClient, DataType
from main.utils import get_ipadress

host_ip = get_ipadress.get_host_ip()
print("ip = " + host_ip)
client = MilvusClient("http://" + host_ip + ":19530")

schema = MilvusClient.create_schema(
    auto_id=True,
    enable_dynamic_field=False,
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="image_name", datatype=DataType.VARCHAR, max_length=256)
schema.verify()
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="id",
    index_type="STL_SORT"
)
index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 1024}
)
# 创建 collection
client.create_collection(
    collection_name="oxford5k_raw_dinov3",
    schema=schema,
    index_params=index_params
)
