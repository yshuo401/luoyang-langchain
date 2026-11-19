"""内部 HMAC 鉴权与重放防护。

骨架阶段不暴露内部业务路由；M6 实现 HMAC-SHA256、时间窗和常量时间比较。
"""

HMAC_HEADER = "X-Internal-Signature"
TIMESTAMP_HEADER = "X-Internal-Timestamp"
REQUEST_ID_HEADER = "X-Request-Id"