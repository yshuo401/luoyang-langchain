"""最小日志初始化；结构化脱敏日志在 M6 继续完善。"""

from __future__ import annotations

import logging

from app.core.constants import SERVICE_NAME


def configure_logging(level: str = "INFO") -> None:
    """配置标准输出日志，不记录请求体、Prompt 或任何密钥。"""

    normalized = level.upper()
    logging.basicConfig(
        level=getattr(logging, normalized, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s [%(process)d] %(message)s",
    )
    logging.getLogger(SERVICE_NAME).setLevel(getattr(logging, normalized, logging.INFO))