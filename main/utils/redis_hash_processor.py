import redis
import logging
from redis.exceptions import RedisError

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RedisHashProcessor:
    def __init__(self, host='localhost', port=6379, db=0, password=None):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True  # 自动将返回值解码为字符串
            )
            # 测试连接
            self.redis_client.ping()
            logger.info("成功连接到Redis服务器")
        except RedisError as e:
            logger.error(f"连接Redis失败: {str(e)}")
            raise

    def process_hash_to_sets(self, main_key, ok_set_key='ok_set', good_set_key='good_set', junk_set_key='junk_set'):
        """
        从指定的哈希键中提取所有字段值对，并根据值分组到不同集合

        :param main_key: 主哈希键名
        :param ok_set_key: value=1时的目标集合键名
        :param good_set_key: value=2时的目标集合键名
        :param junk_set_key: value=3时的目标集合键名
        :return: 处理结果的统计信息
        """
        try:
            # 检查主键是否存在且是哈希类型
            if not self.redis_client.exists(main_key):
                logger.warning(f"主键 {main_key} 不存在")
                return {"success": False, "message": f"主键 {main_key} 不存在"}

            if self.redis_client.type(main_key) != 'hash':
                logger.warning(f"键 {main_key} 不是哈希类型")
                return {"success": False, "message": f"键 {main_key} 不是哈希类型"}

            # 获取所有哈希字段和值
            logger.info(f"开始提取哈希键 {main_key} 的所有字段值对")
            hash_data = self.redis_client.hgetall(main_key)
            logger.info(f"成功提取 {len(hash_data)} 个字段值对")

            # 初始化计数器
            counts = {
                'total': len(hash_data),
                'ok': 0,
                'good': 0,
                'junk': 0,
                'invalid': 0
            }

            # 分组处理
            ok_members = []
            good_members = []
            junk_members = []

            for field, value in hash_data.items():
                if value == '1':
                    ok_members.append(field)
                    counts['ok'] += 1
                elif value == '2':
                    good_members.append(field)
                    counts['good'] += 1
                elif value == '3':
                    junk_members.append(field)
                    counts['junk'] += 1
                else:
                    logger.warning(f"字段 {field} 的值 {value} 无效，跳过处理")
                    counts['invalid'] += 1

            # 批量添加到集合（比逐个添加更高效）
            if ok_members:
                self.redis_client.sadd(ok_set_key, *ok_members)
                logger.info(f"已添加 {counts['ok']} 个成员到集合 {ok_set_key}")

            if good_members:
                self.redis_client.sadd(good_set_key, *ok_members)
                logger.info(f"已添加 {counts['good']} 个成员到集合 {good_set_key}")

            if junk_members:
                self.redis_client.sadd(junk_set_key, *junk_members)
                logger.info(f"已添加 {counts['junk']} 个成员到集合 {junk_set_key}")

            logger.info("处理完成")





            return {
                "success": True,
                "counts": counts,
                "message": "哈希数据已成功分组到相应集合"
            }

        except RedisError as e:
            logger.error(f"处理过程中发生Redis错误: {str(e)}")
            return {"success": False, "message": f"Redis错误: {str(e)}"}
        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}")
            return {"success": False, "message": f"错误: {str(e)}"}


if __name__ == "__main__":
    # 示例用法
    try:
        # 初始化处理器（根据实际情况修改Redis连接参数）
        processor = RedisHashProcessor(
            host='localhost',
            port=6379,
            db=0,
            # password='your_redis_password'  # 如果需要密码
        )

        # 处理指定的主哈希键
        main_hash_key = "all_souls_000013"  # 替换为实际的大主键名
        result = processor.process_hash_to_sets(main_hash_key)

        # 打印处理结果
        print("处理结果:")
        print(f"成功: {result['success']}")
        if 'counts' in result:
            print(f"总处理数: {result['counts']['total']}")
            print(f"OK集合成员数: {result['counts']['ok']}")
            print(f"Good集合成员数: {result['counts']['good']}")
            print(f"Junk集合成员数: {result['counts']['junk']}")
            print(f"无效值数量: {result['counts']['invalid']}")
        print(f"消息: {result['message']}")

    except Exception as e:
        print(f"程序执行失败: {str(e)}")
