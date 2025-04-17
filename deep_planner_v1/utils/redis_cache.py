from datetime import date
import redis
import os
from deep_planner_v1.utils.structs import Schemes


# Redis 缓存池类
class RedisSchemeCache:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 1):
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True,
                                        password=os.getenv("REDIS_PASSWORD"))

    def save_schemes(self, schemes: Schemes, target_date: date = None):
        """保存 Schemes 到 Redis，按日期存储"""
        target_date = target_date or date.today()
        key = f"schemes:{target_date.isoformat()}"
        schemes_json = schemes.json()  # 转换为 JSON 字符串
        self.redis_client.set(key, schemes_json)
        # 可选：设置过期时间，比如 7 天
        self.redis_client.expire(key, 7 * 24 * 60 * 60)

    async def get_schemes(self, target_date: date = None) -> Schemes | None:
        """根据日期从 Redis 获取 Schemes"""
        target_date = target_date or date.today()
        key = f"schemes:{target_date.isoformat()}"
        schemes_json = await self.redis_client.get(key)
        if schemes_json:
            return Schemes.parse_raw(schemes_json)  # 从 JSON 反序列化为 Schemes 对象
        return None
