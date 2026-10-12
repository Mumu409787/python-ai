"""对话模型调用。

对应 Java 的 `ChatAssistant`。

关于 deepseek-flash 的推理内容：`reasoning_content` **不会**出现在
LangChain 的 `ChatOpenAI` 输出里 —— 实测 90 个流式 chunk 中，思维链阶段的
chunk 其 `content` 为空字符串，`additional_kwargs` 也是空的，
拼接后只有正文。所以这里只需跳过空片段，不必额外过滤。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .clients import chat_model
from .errors import BizError, ResultCode, llm_error
from .observability import generation_usage, traced

History = list[tuple[str, str]]


def _build_messages(system: str, user: str, history: History | None) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(system)]
    for question, answer in history or []:
        messages.append(HumanMessage(question))
        messages.append(AIMessage(answer))
    messages.append(HumanMessage(user))
    return messages


@traced("llm.generate", as_type="generation")
async def generate_text(system: str, user: str, history: History | None = None) -> str:
    """一次性生成完整回答。"""
    try:
        response = await chat_model().ainvoke(_build_messages(system, user, history))
        generation_usage(getattr(response, "usage_metadata", None))
        text = _flatten(response.content)
        if not text.strip():
            raise BizError(ResultCode.LLM_CALL_FAILED, "模型未返回有效正文，请重试")
        return text
    except BizError:
        raise
    except Exception as exc:
        raise llm_error(exc) from exc


@traced("llm.stream", as_type="generation")
async def stream_text(system: str, user: str, history: History | None = None) -> AsyncIterator[str]:
    """逐片段流式生成，只吐出非空正文。

    思维链阶段的 chunk 是空字符串，跳过即可 —— 既避免 SSE 发空事件，
    也保证用户看不到推理过程。
    """
    has_text = False
    try:
        async for chunk in chat_model().astream(_build_messages(system, user, history)):
            generation_usage(getattr(chunk, "usage_metadata", None))
            text = _flatten(chunk.content)
            if text:
                has_text = has_text or bool(text.strip())
                yield text
        if not has_text:
            raise BizError(ResultCode.LLM_CALL_FAILED, "模型未返回有效正文，请重试")
    except BizError:
        raise
    except Exception as exc:
        raise llm_error(exc) from exc


def _flatten(content: object) -> str:
    """`content` 可能是 str，也可能是分块列表（多模态场景），统一成字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")
