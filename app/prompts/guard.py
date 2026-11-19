"""Deterministic input guard and safe fixed responses."""

from __future__ import annotations

import re
from enum import StrEnum

GREETING_RESPONSE = (
    "你好，我是洛阳旅游助手。可以问我景点介绍、游玩时长、攻略、美食，"
    "或让我根据收藏解释推荐结果。"
)
REFUSAL_RESPONSE = "这个问题超出洛阳旅游助手的能力范围，我不能提供系统提示、密钥或代码执行帮助。"
VAGUE_RESPONSE = "请补充具体景点、时间或偏好，例如“龙门石窟适合玩多久？”"

_GREETING = re.compile(
    r"^(你好|您好|嗨|hi|hello|哈喽|早上好|下午好|晚上好)[！!。.?？\s]*$", re.IGNORECASE
)
_THANKS = re.compile(r"^(谢谢|感谢|多谢|thanks|thank you)[！!。.?？\s]*$", re.IGNORECASE)
_UNSAFE = re.compile(
    r"(忽略.{0,12}(指令|提示|规则)|系统提示|system prompt|api[_ -]?key|密钥|数据库密码|"
    r"drop\s+table|rm\s+-rf|curl\s+|powershell|<\s*script)",
    re.IGNORECASE,
)


class GuardIntent(StrEnum):
    GREETING = "GREETING"
    THANKS = "THANKS"
    UNSAFE = "UNSAFE"
    VAGUE = "VAGUE"
    NORMAL = "NORMAL"


def classify_guard_intent(message: str) -> GuardIntent:
    normalized = message.strip()
    if _GREETING.fullmatch(normalized):
        return GuardIntent.GREETING
    if _THANKS.fullmatch(normalized):
        return GuardIntent.THANKS
    if _UNSAFE.search(normalized):
        return GuardIntent.UNSAFE
    if len(normalized) <= 2 and not any(char.isdigit() for char in normalized):
        return GuardIntent.VAGUE
    return GuardIntent.NORMAL
