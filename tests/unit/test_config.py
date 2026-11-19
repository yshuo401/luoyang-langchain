from app.core.config import get_settings


def test_qwen_configuration_is_locked() -> None:
    settings = get_settings()
    assert settings.qwen_chat_model == "qwen3.8-flash"
    assert settings.qwen_embedding_model == "qwen3-vl-embedding"
    assert settings.qwen_embedding_dim == 1024
    assert settings.fastapi_host == "127.0.0.1"