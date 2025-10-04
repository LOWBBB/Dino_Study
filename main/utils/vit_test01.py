import os


def delete_matching_files(raw_dir, query_dir):
    """
    删除raw_dir中与query_dir中文件名相同的文件

    Args:
        raw_dir: 原始文件目录
        query_dir: 查询文件目录
    """
    # 获取两个目录中的文件名（不含路径）
    try:
        raw_files = set(os.listdir(raw_dir))
        query_files = set(os.listdir(query_dir))
    except FileNotFoundError as e:
        print(f"目录不存在: {e}")
        return

    # 找出相同的文件名
    common_files = raw_files.intersection(query_files)

    if not common_files:
        print("没有找到文件名相同的文件")
        return

    print(f"找到 {len(common_files)} 个相同的文件名:")
    for file in common_files:
        print(f"  - {file}")

    # 确认是否删除
    confirm = input(f"\n确定要删除 {raw_dir} 中的 {len(common_files)} 个文件吗？(y/n): ")

    if confirm.lower() == 'y':
        deleted_count = 0
        for file in common_files:
            file_path = os.path.join(raw_dir, file)
            try:
                os.remove(file_path)
                print(f"已删除: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"删除失败 {file_path}: {e}")

        print(f"\n成功删除 {deleted_count} 个文件")
    else:
        print("操作已取消")


# 使用示例
if __name__ == "__main__":
    raw_directory = "../data/oxford5k_raw"
    query_directory = "../data/oxford5k_query"

    delete_matching_files(raw_directory, query_directory)