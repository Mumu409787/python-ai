"""业务异常与错误码。

对应 Java 的 `BizException` + `ResultCode`。错误码与 HTTP 状态码解耦：
业务错误返回 HTTP 200 + `code != 0`，只有未预期的异常才返回 500。
"""

from __future__ import annotations

from enum import IntEnum


class ResultCode(IntEnum):
    SUCCESS = 0

    # 1xxx 调用方问题
    PARAM_INVALID = 1001
    DOC_NOT_FOUND = 1002
    DOC_ALREADY_EXISTS = 1003
    FILE_TYPE_UNSUPPORTED = 1004
    FILE_EMPTY = 1005
    SESSION_NOT_FOUND = 1006

    # 2xxx 外部依赖
    LLM_CALL_FAILED = 2001
    LLM_RATE_LIMITED = 2002
    LLM_TIMEOUT = 2003
    EMBEDDING_FAILED = 2004
    VECTOR_STORE_UNAVAILABLE = 2005
    VECTOR_STORE_WRITE_FAILED = 2006

    # 5xxx 服务端
    INTERNAL_ERROR = 5000
    INGEST_FAILED = 5001


_DEFAULT_MESSAGE: dict[ResultCode, str] = {
    ResultCode.SUCCESS: "成功",
    ResultCode.PARAM_INVALID: "参数不合法",
    ResultCode.DOC_NOT_FOUND: "文档不存在",
    ResultCode.DOC_ALREADY_EXISTS: "文档已存在",
    ResultCode.FILE_TYPE_UNSUPPORTED: "不支持的文件类型",
    ResultCode.FILE_EMPTY: "文件内容为空",
    ResultCode.SESSION_NOT_FOUND: "会话不存在",
    ResultCode.LLM_CALL_FAILED: "模型调用失败",
    ResultCode.LLM_RATE_LIMITED: "模型调用被限流",
    ResultCode.LLM_TIMEOUT: "模型响应超时",
    ResultCode.EMBEDDING_FAILED: "向量化失败",
    ResultCode.VECTOR_STORE_UNAVAILABLE: "向量库不可用",
    ResultCode.VECTOR_STORE_WRITE_FAILED: "向量写入失败",
    ResultCode.INTERNAL_ERROR: "系统内部错误",
    ResultCode.INGEST_FAILED: "文档入库失败",
}


class BizError(Exception):
    """业务异常。`retryable` 只影响日志级别，不改变返回给调用方的结果。"""

    def __init__(
        self,
        code: ResultCode,
        message: str | None = None,
        *,
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.message = message or _DEFAULT_MESSAGE.get(code, "未知错误")
        self.retryable = retryable
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{int(self.code)}] {self.message}"

    def as_retryable(self) -> BizError:
        """标记为可重试（链式调用，与 Java 的 `BizException.retryable()` 一致）。"""
        self.retryable = True
        return self


def friendly_llm_message(exc: Exception) -> str:
    """把底层 SDK 异常翻译成给用户看的中文提示。

    与 Java `ChatService.friendlyMessage()` 的映射保持一致。
    """
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return "模型响应超时，请重试或换一个更短的问题"
    if "401" in text or "unauthorized" in text or "invalid api key" in text:
        return "模型 API Key 无效或已过期，请检查 .env 配置"
    if "429" in text or "rate limit" in text:
        return "模型调用被限流，请稍后重试"
    return f"对话失败：{exc}"
