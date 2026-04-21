from __future__ import annotations

import asyncio
import contextlib
import warnings
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
from anyio import ClosedResourceError

T_co = TypeVar("T_co")

from agents import (
    flush_traces,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
    SQLiteSession,
)
from agents.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent
from mcp.shared.exceptions import McpError
from openai.types.responses import ResponseTextDeltaEvent

from .agents import build_care_agent, build_run_config, configure_openai_runtime
from .config import Settings
from .mcp import CareChatMCPRegistry, build_mcp_registry
from .safety import build_guardrail_response, build_output_block_response, detect_local_crisis
from .schemas import CareChatContext, CareRoleHint, InputSafetyAssessment


def run_sync_coro(coro: Coroutine[Any, Any, T_co]) -> T_co:
    """Run *coro* on the thread default loop without closing it (matches ``Runner.run_sync``).

    Using ``asyncio.run()`` for each reply tears down the loop while httpx/OpenAI async clients
    may still schedule ``aclose()``, which triggers ``RuntimeError: Event loop is closed``.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_sync_coro() cannot be used while an event loop is already running; "
            "call the async API with await instead."
        )

    policy = asyncio.get_event_loop_policy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            loop = policy.get_event_loop()
        except RuntimeError:
            loop = policy.new_event_loop()
            policy.set_event_loop(loop)

    if loop.is_closed():
        loop = policy.new_event_loop()
        policy.set_event_loop(loop)

    try:
        return loop.run_until_complete(coro)
    finally:
        # Keep the event loop alive for the lifetime of the process. Explicit resource cleanup
        # happens at the service layer, and forcing async-generator shutdown here conflicts with
        # the MCP streamable-http transport's task-affinity requirements.
        pass


@dataclass(slots=True)
class CareChatService:
    settings: Settings
    session_id: str
    db_path: Path | None = None
    session: SQLiteSession = field(init=False)
    agent: object = field(init=False)
    _mcp_registry: CareChatMCPRegistry | None = field(init=False, default=None, repr=False)
    _mcp_connected: bool = field(init=False, default=False, repr=False)
    _mcp_manager: object | None = field(init=False, default=None, repr=False)
    _active_mcp_servers: list[object] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        configure_openai_runtime(self.settings)

        session_db_path = self.db_path or self.settings.care_chat_session_db_path
        if str(session_db_path) != ":memory:":
            session_db_path.parent.mkdir(parents=True, exist_ok=True)

        self.session = SQLiteSession(
            session_id=self.session_id,
            db_path=session_db_path,
        )
        self._mcp_registry = build_mcp_registry(self.settings)
        if self._mcp_registry is not None:
            self._mcp_manager = self._mcp_registry.build_manager(self.settings)
        self._refresh_agent()

    def _refresh_agent(self) -> None:
        if self._mcp_registry is not None and self._mcp_connected and self._mcp_manager is not None:
            self.agent = build_care_agent(
                self.settings,
                mcp_registry=self._mcp_registry,
                active_mcp_servers=self._active_mcp_servers,
            )
            return
        self.agent = build_care_agent(self.settings)

    async def _prime_mcp_tool_cache(self) -> None:
        if not self._active_mcp_servers:
            return

        healthy_servers: list[object] = []
        failures: list[tuple[object, Exception]] = []
        for server in self._active_mcp_servers:
            try:
                await server.list_tools()
            except Exception as exc:
                failures.append((server, exc))
            else:
                healthy_servers.append(server)

        if not failures:
            return

        for failed_server, _ in failures:
            with contextlib.suppress(Exception):
                await failed_server.cleanup()

        if self.settings.care_chat_mcp_strict:
            _, first_error = failures[0]
            raise first_error

        self._active_mcp_servers = healthy_servers
        self._refresh_agent()

    async def _rebuild_mcp_runtime(self) -> None:
        old_manager = self._mcp_manager
        self._mcp_connected = False
        self._active_mcp_servers = []

        if old_manager is not None:
            with contextlib.suppress(Exception):
                await old_manager.cleanup_all()

        self._mcp_registry = build_mcp_registry(self.settings)
        self._mcp_manager = (
            self._mcp_registry.build_manager(self.settings)
            if self._mcp_registry is not None
            else None
        )
        self._refresh_agent()

    async def _ensure_agent_ready(self, *, force_reconnect: bool = False) -> None:
        if self._mcp_registry is None or self._mcp_manager is None:
            return
        if self._mcp_connected and not force_reconnect:
            return

        if force_reconnect:
            await self._rebuild_mcp_runtime()
            if self._mcp_registry is None or self._mcp_manager is None:
                return

        await self._mcp_manager.connect_all()
        self._mcp_connected = True
        self._active_mcp_servers = list(self._mcp_manager.active_servers)
        await self._prime_mcp_tool_cache()
        self._refresh_agent()

    async def aclose(self) -> None:
        if self._mcp_manager is None or not self._mcp_connected:
            return

        await self._mcp_manager.cleanup_all()
        self._mcp_connected = False
        self._active_mcp_servers = []
        self._refresh_agent()

    async def clear_session(self) -> None:
        await self.session.clear_session()

    def _prepare_message(self, message: str) -> tuple[str, str | None]:
        text = message.strip()
        if not text:
            raise ValueError("Message cannot be empty.")

        local_alert = detect_local_crisis(text)
        if local_alert:
            return (
                text,
                build_guardrail_response(
                    category=local_alert.category.value,
                    language=self.settings.care_chat_language,
                ),
            )

        return text, None

    async def _session_item_count(self) -> int:
        return len(await self.session.get_items())

    async def _rewind_session_to_count(self, item_count: int) -> None:
        while await self._session_item_count() > item_count:
            await self.session.pop_item()

    def _resolve_role_hint(self, role_hint: CareRoleHint | None) -> CareRoleHint:
        return role_hint or self.settings.care_chat_default_role_hint

    def _build_context(self, role_hint: CareRoleHint | None) -> CareChatContext:
        return CareChatContext(
            role_hint=self._resolve_role_hint(role_hint),
            session_id=self.session_id,
        )

    def _build_trace_metadata(self, role_hint: CareRoleHint | None) -> dict[str, str]:
        return {
            "app": "care-chat",
            "session_id": self.session_id,
            "role_hint": self._resolve_role_hint(role_hint),
        }

    def _flush_traces(self) -> None:
        if self.settings.tracing_effective_disabled:
            return
        flush_traces()

    def _is_recoverable_mcp_error(self, exc: BaseException) -> bool:
        seen: set[int] = set()
        stack: list[BaseException] = [exc]

        while stack:
            current = stack.pop()
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)

            if isinstance(
                current,
                (
                    McpError,
                    ClosedResourceError,
                    TimeoutError,
                    httpx.TimeoutException,
                    httpx.ConnectError,
                ),
            ):
                return True

            cause = getattr(current, "__cause__", None)
            if isinstance(cause, BaseException):
                stack.append(cause)

            context = getattr(current, "__context__", None)
            if isinstance(context, BaseException):
                stack.append(context)

        return False

    def _reply_via_runner_run_sync(self, text: str, *, role_hint: CareRoleHint | None = None) -> str:
        run_sync_coro(self._ensure_agent_ready())
        try:
            result = Runner.run_sync(
                self.agent,
                text,
                context=self._build_context(role_hint),
                session=self.session,
                max_turns=16,
                run_config=build_run_config(
                    self.settings,
                    workflow_name="Care Chat Oncology Companion",
                    group_id=self.session_id,
                    trace_metadata=self._build_trace_metadata(role_hint),
                ),
            )
        except InputGuardrailTripwireTriggered as exc:
            assessment = exc.guardrail_result.output.output_info
            if isinstance(assessment, InputSafetyAssessment):
                return build_guardrail_response(
                    category=assessment.category,
                    reason=assessment.reason,
                    language=self.settings.care_chat_language,
                )
            return build_guardrail_response(
                category="medical_emergency",
                language=self.settings.care_chat_language,
            )
        except OutputGuardrailTripwireTriggered:
            return build_output_block_response(language=self.settings.care_chat_language)
        except Exception as exc:
            if not self._is_recoverable_mcp_error(exc):
                raise

            run_sync_coro(self._ensure_agent_ready(force_reconnect=True))
            result = Runner.run_sync(
                self.agent,
                text,
                context=self._build_context(role_hint),
                session=self.session,
                max_turns=16,
                run_config=build_run_config(
                    self.settings,
                    workflow_name="Care Chat Oncology Companion",
                    group_id=self.session_id,
                    trace_metadata=self._build_trace_metadata(role_hint),
                ),
            )

        final_output = str(result.final_output).strip()
        if not final_output:
            return build_output_block_response(language=self.settings.care_chat_language)

        self._flush_traces()
        return final_output

    def _should_prefer_streamed_reply(self) -> bool:
        return (
            self.settings.care_chat_openai_api == "chat_completions"
            and self.settings.care_chat_enable_thinking
            and "kimi-k2.5" in self.settings.care_chat_model.lower()
        )

    async def _collect_streamed_reply(
        self,
        message: str,
        *,
        role_hint: CareRoleHint | None = None,
    ) -> str:
        chunks: list[str] = []
        async for chunk in self.stream_reply(
            message,
            include_reasoning=False,
            role_hint=role_hint,
        ):
            chunks.append(chunk.text)
        final_output = "".join(chunks).strip()
        if not final_output:
            return build_output_block_response(language=self.settings.care_chat_language)
        return final_output

    def reply(self, message: str, *, role_hint: CareRoleHint | None = None) -> str:
        text, local_response = self._prepare_message(message)
        if local_response is not None:
            return local_response

        if self._should_prefer_streamed_reply():
            return run_sync_coro(self._collect_streamed_reply(text, role_hint=role_hint))

        return self._reply_via_runner_run_sync(text, role_hint=role_hint)

    @dataclass(slots=True)
    class StreamChunk:
        kind: Literal["reasoning", "text", "tool"]
        text: str

    def _format_tool_event(self, event: RunItemStreamEvent | AgentUpdatedStreamEvent) -> str | None:
        if isinstance(event, AgentUpdatedStreamEvent):
            return f"切换到 Agent: {event.new_agent.name}"

        item = event.item
        item_agent = getattr(item, "agent", None)
        agent_name = getattr(item_agent, "name", "Unknown Agent")

        def _raw_attr(name: str) -> Any:
            raw_item = getattr(item, "raw_item", None)
            if isinstance(raw_item, dict):
                return raw_item.get(name)
            return getattr(raw_item, name, None)

        def _tool_name() -> str:
            tool_origin = getattr(item, "tool_origin", None)
            origin_tool_name = getattr(tool_origin, "agent_tool_name", None)
            if isinstance(origin_tool_name, str) and origin_tool_name.strip():
                return origin_tool_name.strip()
            for key in ("name", "tool_name", "server_label"):
                candidate = _raw_attr(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            title = getattr(item, "title", None)
            if isinstance(title, str) and title.strip():
                return title.strip()
            description = getattr(item, "description", None)
            if isinstance(description, str) and description.strip() and len(description.strip()) <= 80:
                return description.strip()
            if hasattr(item, "name"):
                candidate = getattr(item, "name", None)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            origin_server = getattr(tool_origin, "mcp_server_name", None)
            if isinstance(origin_server, str) and origin_server.strip():
                return f"mcp:{origin_server.strip()}"
            return "unknown_tool"

        if event.name == "tool_called":
            return f"{agent_name} 调用工具: {_tool_name()}"
        if event.name == "tool_output":
            return f"{agent_name} 工具完成: {_tool_name()}"
        if event.name == "tool_search_called":
            return f"{agent_name} 发起工具搜索"
        if event.name == "tool_search_output_created":
            return f"{agent_name} 收到工具搜索结果"
        if event.name == "mcp_list_tools":
            server_label = _raw_attr("server_label")
            if isinstance(server_label, str) and server_label.strip():
                return f"{agent_name} 列出 MCP 工具: {server_label.strip()}"
            return f"{agent_name} 列出 MCP 工具"
        if event.name == "handoff_requested":
            return f"{agent_name} 请求转接"
        if event.name == "handoff_occured":
            target_agent = getattr(item, "target_agent", None)
            target_name = getattr(target_agent, "name", None)
            if isinstance(target_name, str) and target_name.strip():
                return f"{agent_name} 已转接到: {target_name}"
            return f"{agent_name} 已完成转接"
        return None

    async def stream_reply(
        self,
        message: str,
        *,
        include_reasoning: bool = False,
        include_tool_activity: bool = False,
        role_hint: CareRoleHint | None = None,
    ) -> AsyncIterator["CareChatService.StreamChunk"]:
        text, local_response = self._prepare_message(message)
        if local_response is not None:
            yield self.StreamChunk(kind="text", text=local_response)
            return

        await self._ensure_agent_ready()
        item_count_before = await self._session_item_count()
        streamed_text = ""

        try:
            result = Runner.run_streamed(
                self.agent,
                text,
                context=self._build_context(role_hint),
                session=self.session,
                max_turns=16,
                run_config=build_run_config(
                    self.settings,
                    workflow_name="Care Chat Oncology Companion",
                    group_id=self.session_id,
                    trace_metadata=self._build_trace_metadata(role_hint),
                ),
            )

            async for event in result.stream_events():
                if include_tool_activity:
                    tool_event_text = None
                    if isinstance(event, (RunItemStreamEvent, AgentUpdatedStreamEvent)):
                        tool_event_text = self._format_tool_event(event)
                    if tool_event_text:
                        yield self.StreamChunk(kind="tool", text=tool_event_text)

                if event.type != "raw_response_event":
                    continue

                event_type = getattr(event.data, "type", None)
                if include_reasoning and event_type in {
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                }:
                    delta = getattr(event.data, "delta", "") or ""
                    if delta:
                        yield self.StreamChunk(kind="reasoning", text=delta)
                    continue

                if not isinstance(event.data, ResponseTextDeltaEvent):
                    continue

                delta = event.data.delta or ""
                if not delta:
                    continue

                streamed_text += delta
                yield self.StreamChunk(kind="text", text=delta)

            final_output = str(result.final_output or "").strip()
            if not final_output:
                if not streamed_text:
                    raise RuntimeError("Streaming run finished without visible output.")
                self._flush_traces()
                return

            if final_output.startswith(streamed_text):
                tail = final_output[len(streamed_text) :]
                if tail:
                    yield self.StreamChunk(kind="text", text=tail)
                self._flush_traces()
                return

            if not streamed_text:
                yield self.StreamChunk(kind="text", text=final_output)
                self._flush_traces()
                return
        except InputGuardrailTripwireTriggered as exc:
            assessment = exc.guardrail_result.output.output_info
            if isinstance(assessment, InputSafetyAssessment):
                yield self.StreamChunk(
                    kind="text",
                    text=build_guardrail_response(
                        category=assessment.category,
                        reason=assessment.reason,
                        language=self.settings.care_chat_language,
                    ),
                )
                return

            yield self.StreamChunk(
                kind="text",
                text=build_guardrail_response(
                    category="medical_emergency",
                    language=self.settings.care_chat_language,
                ),
            )
            return
        except OutputGuardrailTripwireTriggered:
            yield self.StreamChunk(
                kind="text",
                text=build_output_block_response(language=self.settings.care_chat_language),
            )
            return
        except Exception as exc:
            if streamed_text:
                raise

            if self._is_recoverable_mcp_error(exc):
                await self._ensure_agent_ready(force_reconnect=True)
            await self._rewind_session_to_count(item_count_before)
            yield self.StreamChunk(
                kind="text",
                text=await asyncio.to_thread(
                    self._reply_via_runner_run_sync,
                    text,
                    role_hint=role_hint,
                ),
            )
