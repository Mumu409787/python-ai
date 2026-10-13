"""Versioned server-to-server question answering for trusted applications."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import Callable
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ConfigDict, Field, field_validator

from ..config import get_settings
from ..errors import BizError, ResultCode
from ..rag import answer, new_session_id
from ..schemas import CamelModel, ChatMode, ChatResponse, Result

SERVICE_ID = "zhixu.qa.v1"
PREFIX = "/api/integration/v1"
logger = logging.getLogger(__name__)


class IntegrationResult[T](Result[T]):
    service_id: Literal["zhixu.qa.v1"] = SERVICE_ID
    api_version: Literal["1"] = "1"
    request_id: str


class IntegrationQuestion(CamelModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    mode: ChatMode = ChatMode.RAG

    @field_validator("message")
    @classmethod
    def nonblank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能只包含空白字符")
        return value


class IntegrationAnswer(ChatResponse):
    client_id: str
    mode: ChatMode
    status: Literal["answered", "no_evidence", "chat"]


class IntegrationInfo(CamelModel):
    name: str = "知序问答服务"
    enabled: bool
    ask_path: str = PREFIX + "/ask"
    openapi_path: str = "/openapi.json"
    authentication: str = "Authorization: Bearer <INTEGRATION_API_KEY>"
    supported_modes: list[ChatMode] = [ChatMode.RAG, ChatMode.CHAT]
    knowledge_base_scope: str = "shared"
    session_scope: str = "clientId + sessionId + mode"
    session_ttl_seconds: int
    session_max_turns: int
    request_timeout_seconds: float
    streaming: bool = False


class IntegrationRoute(APIRoute):
    """Keep the external contract independent from legacy HTTP-200 errors."""

    def get_route_handler(self) -> Callable:
        original = super().get_route_handler()

        async def handle(request: Request):
            request.state.integration_request_id = uuid4().hex
            try:
                response = await original(request)
            except RequestValidationError as exc:
                # Never echo raw inputs, headers, or provider responses.
                details = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                )
                response = _error(request, 422, 1001, details or "参数不合法")
            except HTTPException as exc:
                response = _error(request, exc.status_code, exc.status_code, str(exc.detail))
                response.headers.update(exc.headers or {})
            except BizError as exc:
                status = {
                    ResultCode.PARAM_INVALID: 400,
                    ResultCode.LLM_RATE_LIMITED: 429,
                    ResultCode.LLM_TIMEOUT: 504,
                    ResultCode.VECTOR_STORE_UNAVAILABLE: 503,
                }.get(exc.code, 502 if 2000 <= int(exc.code) < 3000 else 500)
                response = _error(request, status, int(exc.code), exc.message)
            except TimeoutError:
                response = _error(request, 504, 2003, "问答处理超时")
            except Exception:
                logger.exception(
                    "External QA failed (requestId=%s)", request.state.integration_request_id
                )
                response = _error(request, 500, 5000, "系统内部错误")
            response.headers["X-Zhixu-Service"] = SERVICE_ID
            response.headers["X-Request-Id"] = request.state.integration_request_id
            response.headers["Cache-Control"] = "no-store"
            return response

        return handle


def _error(request: Request, status: int, code: int, message: str) -> JSONResponse:
    payload = IntegrationResult[None](
        code=code, message=message, request_id=request.state.integration_request_id
    )
    return JSONResponse(status_code=status, content=payload.model_dump(by_alias=True))


bearer = HTTPBearer(auto_error=False, scheme_name="ZhixuIntegrationKey")


async def authorize(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    configured = get_settings().integration_api_key.get_secret_value()
    if not configured:
        raise HTTPException(503, "外部问答未启用，请配置 INTEGRATION_API_KEY 并重启服务")
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode(), configured.encode()
    ):
        raise HTTPException(401, "问答接入密钥无效或缺失", headers={"WWW-Authenticate": "Bearer"})


router = APIRouter(prefix=PREFIX, tags=[SERVICE_ID], route_class=IntegrationRoute)


@router.get(
    "/info", response_model=IntegrationResult[IntegrationInfo], operation_id="zhixu_qa_v1_info"
)
async def info(request: Request):
    """公开的服务发现；不调用模型，enabled 不代表依赖服务已健康。"""
    settings = get_settings()
    return IntegrationResult(
        request_id=request.state.integration_request_id,
        data=IntegrationInfo(
            enabled=bool(settings.integration_api_key.get_secret_value()),
            session_ttl_seconds=settings.session_ttl_seconds,
            session_max_turns=settings.session_max_turns,
            request_timeout_seconds=settings.integration_timeout_seconds,
        ),
    )


def internal_session_id(client_id: str, session_id: str) -> str:
    # Structured encoding prevents delimiter collisions. Versioning keeps old UI
    # and future external API namespaces separate; this is not user authorization.
    identity = json.dumps([SERVICE_ID, client_id, session_id], separators=(",", ":"))
    return "integration-v1-" + hashlib.sha256(identity.encode()).hexdigest()


@router.post(
    "/ask",
    response_model=IntegrationResult[IntegrationAnswer],
    operation_id="zhixu_qa_v1_ask",
    dependencies=[Depends(authorize)],
    responses={
        status: {"model": IntegrationResult[None]}
        for status in (400, 401, 422, 429, 500, 502, 503, 504)
    },
)
async def ask(question: IntegrationQuestion, request: Request):
    """复用完整问答链路；clientId 隔离会话命名空间，不隔离知识库。"""
    session_id = question.session_id or new_session_id()
    async with asyncio.timeout(get_settings().integration_timeout_seconds):
        result = await answer(
            question.message,
            session_id=internal_session_id(question.client_id, session_id),
            mode=question.mode,
        )
    data = result.model_dump()
    data["session_id"] = session_id
    return IntegrationResult(
        request_id=request.state.integration_request_id,
        data=IntegrationAnswer(
            **data,
            client_id=question.client_id,
            mode=question.mode,
            status="chat"
            if question.mode == ChatMode.CHAT
            else "answered"
            if result.hits
            else "no_evidence",
        ),
    )
