"""内部错误码与异常映射。

M6 实现 HMAC、幂等和内部响应信封时在此集中扩展 ServiceError。
异常消息不得包含 API Key、完整 Prompt、用户聊天原文或服务器绝对路径。
"""


class ServiceError(Exception):
    """AI 服务可分类错误的基类。"""