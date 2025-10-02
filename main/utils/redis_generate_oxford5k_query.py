# 批量处理redis中大主键+.jpg
# 将这些图片通过for循环进行批量的特征提取并持久化
# 从milvus中分别提取出这些特征与测试集做计算得到每个查询图片对应的结果集合
# 用oxford_official_compute对每一个图片做运算 分别得到ap
# 算数平均得到map

# 批量处理redis中大主键+.jpg
# 将这些图片通过for循环进行批量的特征提取并持久化
# 从milvus中分别提取出这些特征与测试集做计算得到每个查询图片对应的结果集合
# 用oxford_official_compute对每一个图片做运算 分别得到ap
# 算数平均得到map

import redis
import os
import shutil


def add_jpg_to_all_keys(redis_host='localhost', redis_port=6379, redis_db=0, match_pattern='*'):
    """
    从Redis指定连接中读取所有符合条件的键，在每个键名末尾添加.jpg并返回

    参数:
        redis_host: Redis服务器主机名
        redis_port: Redis服务器端口
        redis_db: 使用的数据库编号
        match_pattern: 键的匹配模式（默认匹配所有键）

    返回:
        处理后的键名列表，每个名称末尾带有.jpg
    """
    # 连接Redis
    r = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        decode_responses=True  # 自动解码为字符串，避免bytes类型
    )

    processed_keys = []
    cursor = 0  # 迭代游标，初始为0

    try:
        # 使用scan分批获取键，避免大键集一次性加载
        while True:
            # scan命令：cursor为游标，match指定匹配模式，count指定每次迭代返回的近似数量
            cursor, keys = r.scan(cursor=cursor, match=match_pattern, count=1000)
            # 为当前批次的键添加.jpg后缀
            processed_keys.extend([key + '.jpg' for key in keys])
            # 游标为0时表示迭代结束
            if cursor == 0:
                break

        return processed_keys

    except Exception as e:
        print(f"操作Redis时发生错误: {str(e)}")
        return []


def copy_query_images(file_list, source_dir="../data/oxford5k_raw", dest_dir="../data/oxford5k_query"):
    """
    将文件列表中的图片从源目录复制到目标目录

    参数:
        file_list: 包含图片文件名的列表（带.jpg后缀）
        source_dir: 源图片目录路径
        dest_dir: 目标查询目录路径
    """
    # 创建目标目录（如果不存在）
    os.makedirs(dest_dir, exist_ok=True)

    copied_count = 0
    missing_count = 0
    missing_files = []

    for filename in file_list:
        source_path = os.path.join(source_dir, filename)

        if os.path.exists(source_path):
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(source_path, dest_path)  # 保留文件元数据
            copied_count += 1
        else:
            missing_count += 1
            missing_files.append(filename)

    # 输出操作结果统计
    print(f"成功复制 {copied_count} 个文件到 {dest_dir}")
    if missing_count > 0:
        print(f"警告：未找到 {missing_count} 个文件:")
        for file in missing_files[:10]:  # 显示前10个缺失文件
            print(f"  {file}")
        if len(missing_files) > 10:
            print(f"  ... 还有 {len(missing_files) - 10} 个文件未显示")


# 使用示例
if __name__ == "__main__":
    # 替换为实际的Redis连接参数
    result = add_jpg_to_all_keys(
        redis_host='localhost',
        redis_port=6379,
        redis_db=0,
        # 如需筛选特定键（例如以"img_"开头的键），可修改匹配模式
        # match_pattern='img_*'
    )

    print(f"共处理{len(result)}个键")
    # 如需查看部分结果，可打印前N个
    if result:
        print("前60个处理后的键:", result[:60])
        # 复制图片到查询目录
        copy_query_images(result)
    else:
        print("未获取到文件列表，无法执行复制操作")

