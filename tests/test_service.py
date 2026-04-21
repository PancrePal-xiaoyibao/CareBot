from pathlib import Path

import httpx
import pytest
from anyio import ClosedResourceError
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from care_chat.service import CareChatService
from care_chat.config import Settings


class _HealthyServer:
    def __init__(self) -> None:
        self.cleaned = False

    async def list_tools(self) -> list[object]:
        return []

    async def cleanup(self) -> None:
        self.cleaned = True


class _FailingServer:
    def __init__(self) -> None:
        self.cleaned = False

    async def list_tools(self) -> list[object]:
        raise ClosedResourceError

    async def cleanup(self) -> None:
        self.cleaned = True


def _build_service() -> CareChatService:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
    )
    return CareChatService(settings=settings, session_id="service-test")


def test_service_detects_recoverable_mcp_errors() -> None:
    service = _build_service()

    mcp_error = McpError(ErrorData(code=-1, message="mcp failed"))
    timeout_error = httpx.ConnectTimeout("timed out")
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = mcp_error

    assert service._is_recoverable_mcp_error(mcp_error) is True
    assert service._is_recoverable_mcp_error(timeout_error) is True
    assert service._is_recoverable_mcp_error(wrapped) is True
    assert service._is_recoverable_mcp_error(ValueError("plain error")) is False


@pytest.mark.anyio
async def test_prime_mcp_tool_cache_drops_failed_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(CareChatService, "_refresh_agent", lambda self: None)
    service = _build_service()

    healthy = _HealthyServer()
    failing = _FailingServer()
    service._active_mcp_servers = [healthy, failing]

    await service._prime_mcp_tool_cache()

    assert service._active_mcp_servers == [healthy]
    assert healthy.cleaned is False
    assert failing.cleaned is True
