import sys

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

    def compute_ap(self, pos_set, amb_set, ranked_list):
        """计算平均精度(AP)"""
        old_recall = 0.0
        old_precision = 1.0
        ap = 0.0
        intersect_size = 0  # 命中的正样本数量
        j = 0  # 有效样本计数(排除模糊样本)

        for item in ranked_list:
            # 跳过模糊样本
            if item in amb_set:
                continue

            # 如果是正样本，增加命中计数
            if item in pos_set:
                intersect_size += 1


            # 计算召回率和精确率
            # 避免除以零：如果正样本集为空，召回率为0
            recall = intersect_size / len(pos_set) if pos_set else 0.0
            # 避免除以零：有效样本数不为零时计算精确率
            precision = intersect_size / (j + 1) if (j + 1) != 0 else 0.0

            # 累加AP
            ap += (recall - old_recall) * (old_precision + precision) / 2.0

            # 更新状态变量
            old_recall = recall
            old_precision = precision
            j += 1

        print("精确率：" + str(old_precision))
        print("召回率：" + str(old_recall))
        return ap

    def process_hash_to_sets(self, main_key,ranked_list, ok_set_key='ok_set', good_set_key='good_set', junk_set_key='junk_set'):
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

            # 分组处理（使用set集合替代list）
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

            # 批量添加到集合（比逐个添加更高效）
            if ok_members:
                self.redis_client.sadd(ok_set_key, *ok_members)
                logger.info(f"已添加 {counts['ok']} 个成员到集合 {ok_set_key}")

            if good_members:
                self.redis_client.sadd(good_set_key, *good_members)  # 修复原代码此处的笔误（原代码误写为*ok_members）
                logger.info(f"已添加 {counts['good']} 个成员到集合 {good_set_key}")

            if junk_members:
                self.redis_client.sadd(junk_set_key, *junk_members)
                logger.info(f"已添加 {counts['junk']} 个成员到集合 {junk_set_key}")

            logger.info("处理完成")

            pos_set = good_members.union(ok_members)

            ap = self.compute_ap(pos_set, junk_members, ranked_list)
            print("---------------------------------------------------------")
            print("ap = " + str(ap))
            print("---------------------------------------------------------")

            print("---------------------------未检测出------------------------")
            # 未检测出
            detected_set = pos_set.intersection(ranked_list)
            not_detected_set = pos_set.difference(detected_set)
            print(str(not_detected_set))
            print("---------------------------检测出--------------------------")
            print(str(detected_set))
            print("---------------------------------------------------------")

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
        main_hash_key = "ashmolean_000000"  # 替换为实际的大主键名
        ranked_list = \
            ['ashmolean_000000', 'oxford_001964', 'ashmolean_000303', 'ashmolean_000079', 'ashmolean_000002', 'ashmolean_000106', 'ashmolean_000305', 'ashmolean_000058', 'ashmolean_000007', 'oxford_002326', 'oxford_001485', 'oxford_001109', 'oxford_003077', 'oxford_002711', 'ashmolean_000316', 'ashmolean_000024', 'ashmolean_000301', 'ashmolean_000036', 'ashmolean_000016', 'oxford_000681', 'oxford_002172', 'oxford_000960', 'oxford_001935', 'ashmolean_000302', 'oxford_000638']

        result = processor.process_hash_to_sets(main_hash_key,ranked_list)

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

# 显示未找到图像