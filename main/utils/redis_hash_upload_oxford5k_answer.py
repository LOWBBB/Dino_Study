import redis
from dotenv import load_dotenv
import os
import glob
from collections import defaultdict


def get_file_group_prefix(file_name):
    """提取文件名的分组前缀（如"report_2024_ok.txt"→"report_2024"）"""
    base_name = os.path.splitext(file_name)[0]  # 去除.txt后缀，得到"report_2024_ok"
    if "_" in base_name:
        # 按最后一个"_"分割，提取前缀（确保后缀是ok/good/junk/query）
        prefix, suffix = base_name.rsplit("_", 1)
        if suffix in ["ok", "good", "junk", "query"]:
            return prefix
    return base_name  # 不符合命名规则时，用完整基名作为前缀


def extract_query_hash_key(query_file_path):
    """从query文件提取Redis Hash总键（首行空格前字符串去除"oxc1_"）"""
    try:
        with open(query_file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()  # 读取首行并清理空白字符
            if not first_line:
                print(f"⚠️ query文件首行为空：{os.path.basename(query_file_path)}")
                return None

            # 截取第一个空格前的字符串（如"oxc1_report_key 备注"→"oxc1_report_key"）
            key_with_prefix = first_line.split(" ")[0]
            # 去除"oxc1_"前缀
            if key_with_prefix.startswith("oxc1_"):
                final_key = key_with_prefix[len("oxc1_"):]  # 得到"report_key"
                return final_key if final_key else None
            else:
                print(f"⚠️ query文件首行无'oxc1_'前缀：{os.path.basename(query_file_path)}（首行：{first_line}）")
                return None
    except Exception as e:
        print(f"❌ 读取query文件出错：{os.path.basename(query_file_path)}，错误：{e}")
        return None


def batch_query_first_process(target_dir):
    """优先处理query文件生成Hash总键，再处理ok/good/junk文件"""
    # 1. 加载Redis配置并建立连接
    load_dotenv()
    redis_config = {
        "host": os.getenv("REDIS_HOST", "127.0.0.1"),
        "port": int(os.getenv("REDIS_PORT", 6379)),
        "password": os.getenv("REDIS_PASSWORD", ""),
        "db": int(os.getenv("REDIS_DB", 0)),
        "decode_responses": True  # 返回字符串格式，避免字节流解码
    }

    # 连接Redis并测试
    try:
        r = redis.Redis(**redis_config)
        r.ping()
        print(f"✅ Redis 连接成功！目标处理目录：{os.path.abspath(target_dir)}")
    except redis.ConnectionError as e:
        print(f"❌ Redis 连接失败：{e}（请检查服务状态和配置）")
        return
    except Exception as e:
        print(f"❌ Redis 未知连接错误：{e}")
        return

    # 2. 第一步：扫描所有文件，优先处理query文件，生成组→Hash总键的映射
    print("\n" + "=" * 50)
    print("📌 第一步：优先处理所有query文件，生成Hash总键")
    print("=" * 50)

    # 存储所有文件的路径和分组信息（key=分组前缀，value=文件列表）
    all_file_groups = defaultdict(list)
    # 存储分组前缀→Hash总键的映射（核心缓存，后续处理依赖此映射）
    group_hash_map = {}

    # 先遍历所有.txt文件，分类到对应分组
    txt_files = glob.glob(os.path.join(target_dir, "*.txt"), recursive=False)
    if not txt_files:
        print(f"⚠️ 目标目录 {target_dir} 中无.txt文件，程序退出")
        r.close()
        return

    # 第一步：按分组前缀归类所有文件
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        group_prefix = get_file_group_prefix(file_name)
        all_file_groups[group_prefix].append({"path": file_path, "name": file_name})

    # 第二步：遍历每个分组，处理其中的query文件，生成Hash总键
    for group_prefix, files in all_file_groups.items():
        # 查找当前分组的query文件
        query_files = [f for f in files if f["name"].endswith("query.txt")]
        if not query_files:
            print(f"ℹ️ 分组 {group_prefix}：无query文件，后续将跳过该组的ok/good/junk文件")
            continue

        # 一个分组只处理一个query文件（若有多个，取第一个并提示）
        target_query_file = query_files[0]
        if len(query_files) > 1:
            print(f"⚠️ 分组 {group_prefix}：存在多个query文件，仅处理第一个：{target_query_file['name']}")

        # 提取当前分组的Hash总键
        hash_key = extract_query_hash_key(target_query_file["path"])
        if hash_key:
            group_hash_map[group_prefix] = hash_key
            print(f"✅ 分组 {group_prefix}：Hash总键生成成功 → {hash_key}（来自query文件：{target_query_file['name']}）")
        else:
            print(f"❌ 分组 {group_prefix}：query文件无法生成Hash总键，后续跳过该组")

    # 若没有任何分组生成Hash总键，直接退出
    if not group_hash_map:
        print(f"\n⚠️ 所有分组均未生成有效Hash总键，无需处理后续文件，程序退出")
        r.close()
        return

    # 3. 第二步：处理ok/good/junk文件，基于已生成的Hash总键写入Redis
    print("\n" + "=" * 50)
    print("📌 第二步：处理ok/good/junk文件，写入Redis")
    print("=" * 50)

    # 遍历所有分组，处理非query文件
    for group_prefix, files in all_file_groups.items():
        # 跳过无Hash总键的分组（即query文件处理失败或无query文件的组）
        if group_prefix not in group_hash_map:
            continue

        current_hash_key = group_hash_map[group_prefix]
        print(f"\n🔖 开始处理分组：{group_prefix}（Redis Hash总键：{current_hash_key}）")

        group_total_lines = 0  # 统计当前分组的总有效行数

        # 遍历组内文件，只处理ok/good/junk后缀
        for file in files:
            file_name = file["name"]
            file_path = file["path"]

            # 跳过query文件（已在第一步处理）
            if file_name.endswith("query.txt"):
                continue

            # 确定当前文件的Value值
            if file_name.endswith("ok.txt"):
                value = 1
            elif file_name.endswith("good.txt"):
                value = 2
            elif file_name.endswith("junk.txt"):
                value = 3
            else:
                print(f"⚠️  跳过非目标文件：{file_name}（后缀不匹配ok/good/junk）")
                continue

            # 读取文件内容并写入Redis
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    line_count = 0
                    for line in f:
                        field = line.strip()
                        if not field:  # 跳过空行
                            continue

                        # 写入Redis Hash：总键=current_hash_key，Field=行内容，Value=对应值
                        r.hset(current_hash_key, field, value)
                        line_count += 1
                        group_total_lines += 1

                print(f"  ✅ {file_name}：处理完成，有效行数：{line_count}")
            except UnicodeDecodeError:
                print(f"  ❌ {file_name}：编码错误（非UTF-8），请检查文件编码")
            except Exception as e:
                print(f"  ❌ {file_name}：处理出错，错误：{e}")

        # 验证当前分组的写入结果
        redis_field_count = r.hlen(current_hash_key)
        print(
            f"✅ 分组 {group_prefix} 处理完毕！总有效行数：{group_total_lines}，Redis Hash（{current_hash_key}）实际Field数：{redis_field_count}")

    # 4. 最终统计与连接关闭
    print("\n" + "=" * 50)
    print(f"🎉 所有文件处理完成！")
    print(f"📊 有效处理分组数：{len(group_hash_map)}")
    print(f"📊 每个分组的Redis Hash总键：{list(group_hash_map.values())}")
    print("=" * 50)
    r.close()
    print("🔌 Redis 连接已关闭")


# 脚本入口
if __name__ == "__main__":
    # ！！修改为你的.txt文件存放目录！！
    TARGET_DIRECTORY = "../data/oxford5k_answer"  # 示例：当前目录下的txt_groups文件夹

    # 确保目标目录存在（不存在则创建）
    if not os.path.exists(TARGET_DIRECTORY):
        os.makedirs(TARGET_DIRECTORY)
        print(f"ℹ️  目标目录 {TARGET_DIRECTORY} 不存在，已自动创建")
        print(f"ℹ️  请将待处理的.txt文件（按分组命名）放入该目录后重新运行脚本")
    else:
        # 执行核心处理函数
        batch_query_first_process(TARGET_DIRECTORY)