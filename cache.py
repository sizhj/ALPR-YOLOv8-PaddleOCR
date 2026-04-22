"""
cache.py
Redis 缓存模块

用途：缓存实时车位数量，避免每次查询都打数据库。
使用原子操作 DECR/INCR 解决并发场景下的车位超卖问题。

依赖：
    pip install redis
"""

import os
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB   = int(os.getenv("REDIS_DB", 0))

# 连接池，复用连接
pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,
    max_connections=20
)

def get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=pool)


# ── 车位操作 ─────────────────────────────────────────────

def lot_key(lot_id: int) -> str:
    """Redis key 格式：parking:lot:{id}:available"""
    return f"parking:lot:{lot_id}:available"


def init_lot_cache(lot_id: int, available: int):
    """
    初始化停车场车位缓存。
    服务启动时调用，将数据库中的可用车位数同步到 Redis。
    """
    r = get_redis()
    r.set(lot_key(lot_id), available)


def get_available_spaces(lot_id: int) -> int:
    """
    读取实时可用车位数。
    直接读 Redis，不查数据库，响应极快。
    """
    r = get_redis()
    val = r.get(lot_key(lot_id))
    return int(val) if val is not None else 0


def decr_space(lot_id: int) -> int:
    """
    车辆进场：可用车位 -1。
    使用 Redis 原子操作 DECR，天然线程安全，
    解决高并发场景下多辆车同时进场导致的超卖问题。

    Returns:
        操作后的剩余车位数，-1 表示车位已满
    """
    r = get_redis()
    remaining = r.decr(lot_key(lot_id))
    if remaining < 0:
        # 已超出，回滚
        r.incr(lot_key(lot_id))
        return -1
    return remaining


def incr_space(lot_id: int) -> int:
    """
    车辆出场：可用车位 +1。
    使用原子操作 INCR。
    """
    r = get_redis()
    return r.incr(lot_key(lot_id))
