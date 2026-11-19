"""健康检查响应契约。"""

from typing import Literal

from pydantic import BaseModel


class LiveResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]