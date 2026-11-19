"""诊断 Redis 是否满足 LangGraph RedisSaver 的运行前置条件。

RedisSaver 使用 RedisJSON 与 RediSearch；仅 PING 成功并不代表 checkpointer 可用。
本脚本只读取配置并调用官方高层 API，不打印 API Key 或连接密码。
"""

from __future__ import annotations

from langgraph.checkpoint.redis import RedisSaver
from redis import Redis

from app.core.config import get_settings


def main() -> None:
    """检查 Redis 版本、模块和官方 RedisSaver.setup()。"""

    settings = get_settings()
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        server = client.info("server")
        print(f"redis_version: {server.get('redis_version', 'unknown')}")
        modules = client.execute_command("MODULE", "LIST")
        print(f"redis_modules: {modules}")
    finally:
        client.close()

    try:
        with RedisSaver.from_conn_string(
            settings.redis_url,
            checkpoint_prefix=settings.checkpoint_key_prefix,
            checkpoint_write_prefix=f"{settings.checkpoint_key_prefix}write:",
        ) as saver:
            saver.setup()
    except Exception as exc:
        print(f"checkpointer: NOT_READY ({type(exc).__name__}: {exc})")
        raise SystemExit(1) from None

    print("checkpointer: READY")


if __name__ == "__main__":
    main()