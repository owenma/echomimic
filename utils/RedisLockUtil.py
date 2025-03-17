import redis

class RedisLock:
    def __init__(self, redis_client, lock_key):
        self.redis_client = redis_client
        self.lock_key = lock_key
        self.timeout = 30  # 锁的超时时间

    def acquire_lock(self):
        """尝试获取锁"""
        # 使用 SET 命令，带上 NX（只在键不存在时设置键）和 EX（设置键的过期时间）
        # 这里我们返回一个布尔值表示是否成功获取了锁
        return self.redis_client.set(self.lock_key, '1', nx=True, ex=self.timeout)

    def release_lock(self):
        """释放锁"""
        # 删除键来释放锁
        self.redis_client.delete(self.lock_key)

# 使用示例
# client = redis.StrictRedis(host='localhost', port=6379, db=0)
# lock = RedisLock(client, 'my_distributed_lock')
#
# if lock.acquire_lock():
#     try:
#         # 执行受保护的代码块
#         print("Lock acquired, performing operations...")
#         # 假设这里是你的业务逻辑代码
#
#     finally:
#         # 确保最终释放锁
#         lock.release_lock()
#         print("Lock released.")
# else:
#     print("Could not acquire lock.")
