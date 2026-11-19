import os

import pytest
from app.core.config import get_settings
from app.integrations.qwen import create_chat_model, create_embedding_model

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_QWEN_SMOKE") != "1",
    reason="设置 RUN_QWEN_SMOKE=1 后才调用真实 Qwen API",
)


def test_real_qwen_chat_and_embedding() -> None:
    settings = get_settings()
    response = create_chat_model(settings).invoke("只回复：OK")
    assert response.content

    vector = create_embedding_model(settings).embed_query("洛阳龙门石窟")
    assert len(vector) == settings.qwen_embedding_dim