import asyncio
import os

import pytest
from app.api.health import _check_milvus, _check_redis
from app.core.config import get_settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INFRA_TESTS") != "1",
        reason="设置 RUN_INFRA_TESTS=1 后才连接 Redis/Milvus",
    ),
]


def test_redis_is_reachable() -> None:
    assert asyncio.run(_check_redis(get_settings())) is True


def test_milvus_is_reachable() -> None:
    assert _check_milvus(get_settings()) is True