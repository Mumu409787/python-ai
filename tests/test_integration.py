"""External QA contract, auth and session boundaries; no paid model calls."""

from __future__ import annotations

import asyncio
import importlib.util
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from python_ai.api import integration
from python_ai.config import PROJECT_ROOT, Settings
from python_ai.errors import BizError, ResultCode
from python_ai.main import app
from python_ai.schemas import ChatMode, ChatResponse, RetrievalHit


async def test_external_entry_reuses_actual_rag_orchestration(monkeypatch):
    from python_ai import config, rag
    from python_ai.schemas import QueryInfo
    from python_ai.vectorstore import SearchHit

    settings = Settings(_env_file=None, integration_api_key=KEY, langfuse_enabled=False)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    monkeypatch.setattr(integration, "get_settings", lambda: settings)
    history = AsyncMock(return_value=[])
    save = AsyncMock(return_value=True)
    generate = AsyncMock(return_value="18743 [1]")
    monkeypatch.setattr(rag, "load_history", history)
    monkeypatch.setattr(rag, "append_turn", save)
    monkeypatch.setattr(rag, "generate_text", generate)
    monkeypatch.setattr(
        rag,
        "rewrite_query",
        AsyncMock(
            return_value=QueryInfo(
                original_question="端口？", retrieval_query="端口？", rewrite_status="no_history"
            )
        ),
    )
    monkeypatch.setattr(
        rag,
        "retrieve",
        AsyncMock(return_value=[SearchHit(0.9, 12, "ports.md", 34, 0, "端口为 18743")]),
    )
    response = await request(
        "POST",
        ASK,
        headers=AUTH,
        json={"clientId": "actual-pipeline", "message": "端口？", "sessionId": "thread-1"},
    )
    assert response.status_code == 200
    assert response.json()["data"]["answer"] == "18743 [1]"
    assert response.json()["data"]["hits"][0]["docId"] == 12
    sid = integration.internal_session_id("actual-pipeline", "thread-1")
    history.assert_awaited_once_with(sid)
    save.assert_awaited_once_with(sid, "端口？", "18743 [1]")
    generate.assert_awaited_once()


KEY = "integration-test-key-" + "x" * 32
AUTH = {"Authorization": "Bearer " + KEY}
ASK = "/api/integration/v1/ask"


@pytest.fixture
def external(monkeypatch):
    settings = SimpleNamespace(
        integration_api_key=SecretStr(KEY),
        integration_timeout_seconds=2.0,
        session_ttl_seconds=3600,
        session_max_turns=10,
    )
    monkeypatch.setattr(integration, "get_settings", lambda: settings)
    engine = AsyncMock(
        return_value=ChatResponse(
            session_id="internal-only",
            answer="端口为 18743 [1]",
            hits=[RetrievalHit(doc_id=4, chunk_id=7, doc_name="mixed.pdf", source_page=2)],
            memory_saved=False,
        )
    )
    monkeypatch.setattr(integration, "answer", engine)
    return settings, engine


async def request(method, path, **kwargs):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.request(method, path, **kwargs)


async def test_public_discovery_identifies_service_without_calling_model(external):
    response = await request("GET", "/api/integration/v1/info")
    body = response.json()
    assert response.status_code == 200
    assert body["serviceId"] == response.headers["X-Zhixu-Service"] == "zhixu.qa.v1"
    assert body["requestId"] == response.headers["X-Request-Id"]
    assert response.headers["Cache-Control"] == "no-store"
    assert body["data"]["enabled"] is True
    assert body["data"]["knowledgeBaseScope"] == "shared"
    assert KEY not in response.text
    external[1].assert_not_awaited()


@pytest.mark.parametrize(
    "header", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic x"}]
)
async def test_authentication_prevents_engine_calls(external, header):
    response = await request(
        "POST", ASK, headers=header, json={"clientId": "crm", "message": "问题"}
    )
    assert response.status_code == 401
    assert response.json()["code"] == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.json()["serviceId"] == "zhixu.qa.v1"
    external[1].assert_not_awaited()


async def test_disabled_endpoint_and_discovery(external):
    external[0].integration_api_key = SecretStr("")
    response = await request("POST", ASK, headers=AUTH, json={"clientId": "crm", "message": "问题"})
    assert response.status_code == 503
    assert response.json()["code"] == 503
    info = await request("GET", "/api/integration/v1/info")
    assert info.json()["data"]["enabled"] is False
    external[1].assert_not_awaited()


async def test_answer_preserves_evidence_metadata_and_returns_external_session(external):
    response = await request(
        "POST",
        ASK,
        headers=AUTH,
        json={"clientId": "crm", "message": "端口多少？", "sessionId": "user-1-thread-1"},
    )
    assert response.status_code == 200
    body = response.json()
    data = body["data"]
    assert body["code"] == 0
    assert data["sessionId"] == "user-1-thread-1"
    assert data["clientId"] == "crm"
    assert data["mode"] == "RAG"
    assert data["status"] == "answered"
    assert data["answer"] == "端口为 18743 [1]"
    assert data["hits"][0]["sourcePage"] == 2
    assert data["memorySaved"] is False
    external[1].assert_awaited_once_with(
        "端口多少？",
        session_id=integration.internal_session_id("crm", "user-1-thread-1"),
        mode=ChatMode.RAG,
    )


async def test_client_namespaces_and_generated_sessions(external):
    for client_id in ("crm", "erp", "crm"):
        await request(
            "POST",
            ASK,
            headers=AUTH,
            json={"clientId": client_id, "message": "问题", "sessionId": "same"},
        )
    calls = external[1].await_args_list
    assert calls[0].kwargs["session_id"] != calls[1].kwargs["session_id"]
    assert calls[0].kwargs["session_id"] == calls[2].kwargs["session_id"]
    sessions = []
    for _ in range(2):
        response = await request(
            "POST", ASK, headers=AUTH, json={"clientId": "crm", "message": "新问题"}
        )
        sessions.append(response.json()["data"]["sessionId"])
    assert sessions[0] != sessions[1]
    assert all(s.startswith("s-") for s in sessions)


@pytest.mark.parametrize(
    "mode,status,used_rag", [("RAG", "no_evidence", True), ("CHAT", "chat", False)]
)
async def test_no_evidence_and_chat_statuses(external, mode, status, used_rag):
    external[1].return_value = ChatResponse(session_id="internal", answer="答复", used_rag=used_rag)
    response = await request(
        "POST", ASK, headers=AUTH, json={"clientId": "crm", "message": "问题", "mode": mode}
    )
    assert response.json()["data"]["status"] == status
    assert response.json()["data"]["usedRag"] == used_rag
    assert external[1].await_args.kwargs["mode"] == ChatMode(mode)


@pytest.mark.parametrize(
    "patch",
    [
        {"clientId": ""},
        {"clientId": "bad/id"},
        {"clientId": "x" * 65},
        {"message": " \n"},
        {"message": "x" * 4001},
        {"sessionId": ""},
        {"sessionId": "x" * 129},
        {"mode": "wrong"},
        {"useRag": False},
    ],
)
async def test_invalid_contract_never_calls_model(external, patch):
    response = await request(
        "POST", ASK, headers=AUTH, json={"clientId": "crm", "message": "问题", **patch}
    )
    assert response.status_code == 422
    assert response.json()["code"] == 1001
    assert response.headers["X-Request-Id"] == response.json()["requestId"]
    external[1].assert_not_awaited()


async def test_invalid_json_uses_versioned_error_envelope(external):
    response = await request(
        "POST", ASK, headers={**AUTH, "Content-Type": "application/json"}, content=b'{"broken"'
    )
    assert response.status_code == 422
    assert response.json()["serviceId"] == "zhixu.qa.v1"
    external[1].assert_not_awaited()


@pytest.mark.parametrize(
    "code,status",
    [
        (ResultCode.LLM_CALL_FAILED, 502),
        (ResultCode.LLM_RATE_LIMITED, 429),
        (ResultCode.LLM_TIMEOUT, 504),
        (ResultCode.EMBEDDING_FAILED, 502),
        (ResultCode.VECTOR_STORE_UNAVAILABLE, 503),
    ],
)
async def test_provider_errors_have_http_statuses(external, code, status):
    external[1].side_effect = BizError(code)
    response = await request("POST", ASK, headers=AUTH, json={"clientId": "crm", "message": "问题"})
    assert response.status_code == status
    assert response.json()["code"] == int(code)
    assert response.json()["data"] is None


async def test_deadline_cancels_engine_and_has_no_success_response(external):
    external[0].integration_timeout_seconds = 0.01
    cancelled = asyncio.Event()

    async def wait(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    external[1].side_effect = wait
    response = await request("POST", ASK, headers=AUTH, json={"clientId": "crm", "message": "问题"})
    assert response.status_code == 504
    assert response.json()["code"] == 2003
    assert cancelled.is_set()


async def test_unexpected_failure_does_not_echo_exception(external):
    external[1].side_effect = RuntimeError("private provider data")
    response = await request("POST", ASK, headers=AUTH, json={"clientId": "crm", "message": "问题"})
    assert response.status_code == 500
    assert "private provider data" not in response.text


async def test_legacy_endpoint_retains_legacy_validation_status(external):
    response = await request("POST", "/api/chat", json={"message": " "})
    assert response.status_code == 200
    assert response.json()["code"] == 1001
    assert "serviceId" not in response.json()


def test_openapi_describes_auth_and_stable_operations():
    spec = app.openapi()
    post = spec["paths"][ASK]["post"]
    assert post["operationId"] == "zhixu_qa_v1_ask"
    assert post["security"] == [{"ZhixuIntegrationKey": []}]
    assert "401" in post["responses"]


def test_integration_key_is_validated_and_hidden():
    with pytest.raises(ValidationError, match="至少 32 字符"):
        Settings(_env_file=None, integration_api_key="short")
    settings = Settings(_env_file=None, integration_api_key=KEY)
    assert KEY not in repr(settings)


def test_setup_preserves_existing_configuration_and_key(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "setup_integration", PROJECT_ROOT / "scripts/setup_integration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = tmp_path / ".env"
    original = b"# keep\r\nDEEPSEEK_API_KEY=unchanged\r\nINTEGRATION_API_KEY=\r\n"
    env.write_bytes(original)
    assert module.configure(env) is True
    after = env.read_bytes()
    assert b"# keep\r\nDEEPSEEK_API_KEY=unchanged\r\n" in after
    assert b"INTEGRATION_API_KEY=" in after
    assert module.configure(env) is False
    assert env.read_bytes() == after


def test_namespace_encoding_is_unambiguous():
    assert integration.internal_session_id("a", "b:c") != integration.internal_session_id(
        "a:b", "c"
    )
