from __future__ import annotations

import os

import pytest
from app.config import get_settings
from app.llm import create_chat_model, create_embeddings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_QWEN_SMOKE") != "1",
    reason="设置 RUN_QWEN_SMOKE=1 后才调用真实 Qwen API",
)


def test_real_qwen_chat_and_embedding() -> None:
    settings = get_settings()
    response = create_chat_model(settings).invoke("只回复：OK")
    assert response.content
    vector = create_embeddings(settings).embed_query("洛阳龙门石窟")
    assert len(vector) > 0
