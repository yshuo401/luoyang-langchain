from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.config import get_settings
from app.rag import extract_markdown

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MINERU_SMOKE") != "1",
    reason="设置 RUN_MINERU_SMOKE=1 后才调用真实 MinerU API",
)


def test_mineru_parses_a_real_image() -> None:
    image = Path(__file__).resolve().parents[3] / "数据集-照片-文档" / "景点" / "龙门石窟.jpg"
    markdown = extract_markdown(image.read_bytes(), image.name, get_settings())
    assert markdown
