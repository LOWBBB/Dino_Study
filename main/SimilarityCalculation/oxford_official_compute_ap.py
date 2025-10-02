import sys
import redis
import logging
from redis.exceptions import RedisError
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RedisHashProcessor:
    def __init__(self, redis_client):
        """初始化处理器，接收外部Redis客户端（依赖注入）"""
        self.redis_client = redis_client

    @staticmethod
    def compute_ap(pos_set, amb_set, ranked_list):
        """
        原子性计算AP：仅接收三个集合参数，返回AP值及精度召回率
        不依赖类属性，可独立调用
        """
        old_recall = 0.0
        old_precision = 1.0
        ap = 0.0
        intersect_size = 0  # 命中的正样本数量
        j = 0  # 有效样本计数(排除模糊样本)

        for item in ranked_list:
            if item in amb_set:
                continue
            if item in pos_set:
                intersect_size += 1

            # 计算召回率和精确率
            recall = intersect_size / len(pos_set) if pos_set else 0.0
            precision = intersect_size / (j + 1) if (j + 1) != 0 else 0.0

            # 累加AP
            ap += (recall - old_recall) * (old_precision + precision) / 2.0

            # 更新状态变量
            old_recall = recall
            old_precision = precision
            j += 1

        return {
            'ap': ap,
            'precision': old_precision,
            'recall': old_recall
        }

    def extract_hash_data(self, main_key):
        """
        原子性提取哈希数据：仅负责从Redis获取哈希数据
        独立验证键存在性和类型
        """
        if not self.redis_client.exists(main_key):
            logger.warning(f"主键 {main_key} 不存在")
            return None, {"success": False, "message": f"主键 {main_key} 不存在"}

        if self.redis_client.type(main_key) != 'hash':
            logger.warning(f"键 {main_key} 不是哈希类型")
            return None, {"success": False, "message": f"键 {main_key} 不是哈希类型"}

        logger.info(f"提取哈希键 {main_key} 的所有字段值对")
        hash_data = self.redis_client.hgetall(main_key)
        logger.info(f"成功提取 {len(hash_data)} 个字段值对")
        return hash_data, {"success": True, "message": "哈希数据提取成功"}

    @staticmethod
    def group_members(hash_data):
        """
        原子性分组成员：仅接收哈希数据，返回分组结果
        不依赖Redis，纯内存操作
        """
        counts = {
            'total': len(hash_data),
            'ok': 0,
            'good': 0,
            'junk': 0,
            'invalid': 0
        }
        ok_members = set()
        good_members = set()
        junk_members = set()

        for field, value in hash_data.items():
            if value == '1':
                ok_members.add(field)
                counts['ok'] += 1
            elif value == '2':
                good_members.add(field)
                counts['good'] += 1
            elif value == '3':
                junk_members.add(field)
                counts['junk'] += 1
            else:
                logger.warning(f"字段 {field} 的值 {value} 无效，跳过处理")
                counts['invalid'] += 1

        return {
            'ok': ok_members,
            'good': good_members,
            'junk': junk_members,
            'counts': counts
        }

    def save_sets_to_redis(self, groups, ok_set_key, good_set_key, junk_set_key):
        """
        原子性保存集合：仅负责将分组结果保存到Redis
        仅依赖Redis客户端和输入参数
        """
        if groups['ok']:
            self.redis_client.sadd(ok_set_key, *groups['ok'])
            logger.info(f"已添加 {len(groups['ok'])} 个成员到集合 {ok_set_key}")

        if groups['good']:
            self.redis_client.sadd(good_set_key, *groups['good'])
            logger.info(f"已添加 {len(groups['good'])} 个成员到集合 {good_set_key}")

        if groups['junk']:
            self.redis_client.sadd(junk_set_key, *groups['junk'])
            logger.info(f"已添加 {len(groups['junk'])} 个成员到集合 {junk_set_key}")

        return {"success": True, "message": "集合保存成功"}

    @staticmethod
    def calculate_detection_results(pos_set, ranked_list):
        """
        原子性计算检测结果：仅处理集合运算
        """
        detected_set = pos_set.intersection(ranked_list)
        not_detected_set = pos_set.difference(detected_set)
        return detected_set, not_detected_set

# 读取JSON文件
def read_json_file(file_path):
    try:
        # 打开文件并加载JSON数据
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的JSON格式")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None


def create_redis_client(host='localhost', port=6379, db=0, password=None):
    """独立的Redis客户端创建函数，原子性负责连接创建和测试"""
    try:
        client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        client.ping()
        logger.info("成功连接到Redis服务器")
        return client, None
    except RedisError as e:
        error_msg = f"连接Redis失败: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def main():
    # 1. 配置参数（所有参数集中管理）
    redis_config = {
        'host': 'localhost',
        'port': 6379,
        'db': 1,
        'password': None
    }
    file_path = "retrieval_results.json"
    json_data = read_json_file(file_path)

    set_keys = {
        'ok': 'ok_set',
        'good': 'good_set',
        'junk': 'junk_set'
    }

    try:
        # 2. 创建Redis客户端（原子操作）
        redis_client, error = create_redis_client(**redis_config)
        if error:
            raise Exception(error)

        # 3. 初始化处理器
        processor = RedisHashProcessor(redis_client)

    except Exception as e:
        print(f"程序执行失败: {str(e)}")

    for key, value in json_data.items():
        print(f"键：{key}，值：{value}")
        main_hash_key = key
        ranked_list = value
        # 4. 提取哈希数据（原子操作）
        hash_data, extract_result = processor.extract_hash_data(main_hash_key)
        if not extract_result['success']:
            raise Exception(extract_result['message'])

        # 5. 分组处理成员（原子操作）
        groups = processor.group_members(hash_data)

        # 6. 保存集合到Redis（原子操作）
        save_result = processor.save_sets_to_redis(
            groups,
            set_keys['ok'],
            set_keys['good'],
            set_keys['junk']
        )
        if not save_result['success']:
            raise Exception(save_result['message'])

        # 7. 计算AP（原子操作）
        pos_set = groups['good'].union(groups['ok'])
        amb_set = groups['junk']
        ap_result = RedisHashProcessor.compute_ap(pos_set, amb_set, ranked_list)

        # 8. 计算检测结果（原子操作）
        detected_set, not_detected_set = processor.calculate_detection_results(
            pos_set, ranked_list
        )


        # 9. 输出结果
        print("精确率：" + str(ap_result['precision']))
        print("召回率：" + str(ap_result['recall']))
        print("---------------------------------------------------------")
        print("ap = " + str(ap_result['ap']))
        print("---------------------------------------------------------")
        print("---------------------------未检测出------------------------")
        print(str(not_detected_set))
        print("---------------------------检测出--------------------------")
        print(str(detected_set))
        print("---------------------------------------------------------")

        # 10. 打印处理统计
        print("处理结果:")
        print(f"成功: True")
        print(f"总处理数: {groups['counts']['total']}")
        print(f"OK集合成员数: {groups['counts']['ok']}")
        print(f"Good集合成员数: {groups['counts']['good']}")
        print(f"Junk集合成员数: {groups['counts']['junk']}")
        print(f"无效值数量: {groups['counts']['invalid']}")
        print(f"消息: 处理完成")


if __name__ == "__main__":
    main()