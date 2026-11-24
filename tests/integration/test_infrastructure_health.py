from __future__ import annotations

import os

import pytest
from app.config import get_settings
from app.rag import check_milvus

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INFRA_TESTS") != "1",
        reason="设置 RUN_INFRA_TESTS=1 后才连接 Milvus",
    ),
]


def test_milvus_is_reachable() -> None:
    assert check_milvus(get_settings()) is True
