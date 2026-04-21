from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from agents.mcp import (
    MCPServer,
    MCPServerManager,
    MCPServerSse,
    MCPServerStdio,
    MCPServerStreamableHttp,
    create_static_tool_filter,
)
from pydantic import BaseModel, Field, ValidationError, model_validator

from .config import PROJECT_ROOT, Settings

ROUTER_AGENT_KEY = "router"
PATIENT_COORDINATOR_AGENT_KEY = "patient_coordinator"
PATIENT_EMOTIONAL_AGENT_KEY = "patient_emotional"
PATIENT_NAVIGATION_AGENT_KEY = "patient_navigation"
PATIENT_URGENT_AGENT_KEY = "patient_urgent"
CAREGIVER_COORDINATOR_AGENT_KEY = "caregiver_coordinator"
CAREGIVER_EMOTIONAL_AGENT_KEY = "caregiver_emotional"
CAREGIVER_COORDINATION_AGENT_KEY = "caregiver_coordination"
CAREGIVER_URGENT_AGENT_KEY = "caregiver_urgent"
VOLUNTEER_COORDINATOR_AGENT_KEY = "volunteer_coordinator"
VOLUNTEER_TASK_AGENT_KEY = "volunteer_task"
VOLUNTEER_BOUNDARY_AGENT_KEY = "volunteer_boundary"
VOLUNTEER_ESCALATION_AGENT_KEY = "volunteer_escalation"

ALL_AGENT_KEYS = {
    ROUTER_AGENT_KEY,
    PATIENT_COORDINATOR_AGENT_KEY,
    PATIENT_EMOTIONAL_AGENT_KEY,
    PATIENT_NAVIGATION_AGENT_KEY,
    PATIENT_URGENT_AGENT_KEY,
    CAREGIVER_COORDINATOR_AGENT_KEY,
    CAREGIVER_EMOTIONAL_AGENT_KEY,
    CAREGIVER_COORDINATION_AGENT_KEY,
    CAREGIVER_URGENT_AGENT_KEY,
    VOLUNTEER_COORDINATOR_AGENT_KEY,
    VOLUNTEER_TASK_AGENT_KEY,
    VOLUNTEER_BOUNDARY_AGENT_KEY,
    VOLUNTEER_ESCALATION_AGENT_KEY,
}

TARGET_GROUPS: dict[str, set[str]] = {
    "all": set(ALL_AGENT_KEYS),
    "patient": {
        PATIENT_COORDINATOR_AGENT_KEY,
        PATIENT_EMOTIONAL_AGENT_KEY,
        PATIENT_NAVIGATION_AGENT_KEY,
        PATIENT_URGENT_AGENT_KEY,
    },
    "caregiver": {
        CAREGIVER_COORDINATOR_AGENT_KEY,
        CAREGIVER_EMOTIONAL_AGENT_KEY,
        CAREGIVER_COORDINATION_AGENT_KEY,
        CAREGIVER_URGENT_AGENT_KEY,
    },
    "volunteer": {
        VOLUNTEER_COORDINATOR_AGENT_KEY,
        VOLUNTEER_TASK_AGENT_KEY,
        VOLUNTEER_BOUNDARY_AGENT_KEY,
        VOLUNTEER_ESCALATION_AGENT_KEY,
    },
    "coordinators": {
        PATIENT_COORDINATOR_AGENT_KEY,
        CAREGIVER_COORDINATOR_AGENT_KEY,
        VOLUNTEER_COORDINATOR_AGENT_KEY,
    },
    "specialists": {
        PATIENT_EMOTIONAL_AGENT_KEY,
        PATIENT_NAVIGATION_AGENT_KEY,
        PATIENT_URGENT_AGENT_KEY,
        CAREGIVER_EMOTIONAL_AGENT_KEY,
        CAREGIVER_COORDINATION_AGENT_KEY,
        CAREGIVER_URGENT_AGENT_KEY,
        VOLUNTEER_TASK_AGENT_KEY,
        VOLUNTEER_BOUNDARY_AGENT_KEY,
        VOLUNTEER_ESCALATION_AGENT_KEY,
    },
}

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
MCPTransport = Literal["stdio", "sse", "streamable_http"]


def _expand_env_placeholders(text: str) -> str:
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in os.environ:
            missing.append(key)
            return ""
        return os.environ[key]

    expanded = _ENV_PATTERN.sub(_replace, text)
    if missing:
        missing_list = ", ".join(sorted(set(missing)))
        raise ValueError(f"MCP config references missing environment variables: {missing_list}")
    return expanded


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_placeholders(value)
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_env_values(item) for key, item in value.items()}
    return value


def _expand_targets(targets: list[str]) -> frozenset[str]:
    resolved: set[str] = set()
    for raw_target in targets:
        target = raw_target.strip()
        if not target:
            continue
        if target in TARGET_GROUPS:
            resolved.update(TARGET_GROUPS[target])
            continue
        if target in ALL_AGENT_KEYS:
            resolved.add(target)
            continue
        supported = ", ".join(sorted({*TARGET_GROUPS.keys(), *ALL_AGENT_KEYS}))
        raise ValueError(
            f"Unsupported MCP target '{target}'. Supported values: {supported}"
        )
    if not resolved:
        raise ValueError("MCP server targets cannot be empty.")
    return frozenset(resolved)


class MCPServerDefinition(BaseModel):
    name: str
    enabled: bool = True
    transport: MCPTransport
    targets: list[str] = Field(default_factory=lambda: ["all"])

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None

    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float | None = None
    sse_read_timeout: float | None = None
    terminate_on_close: bool | None = None
    ignore_initialized_notification_failure: bool | None = None

    cache_tools_list: bool = True
    client_session_timeout_seconds: float | None = 5.0
    use_structured_content: bool = False
    max_retry_attempts: int = 0
    retry_backoff_seconds_base: float = 1.0
    allowed_tool_names: list[str] = Field(default_factory=list)
    blocked_tool_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> MCPServerDefinition:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP server requires 'command'.")
        if self.transport in {"sse", "streamable_http"} and not self.url:
            raise ValueError(f"{self.transport} MCP server requires 'url'.")
        return self

    def build_server(self) -> MCPServer:
        tool_filter = create_static_tool_filter(
            allowed_tool_names=self.allowed_tool_names or None,
            blocked_tool_names=self.blocked_tool_names or None,
        )
        common_kwargs = {
            "cache_tools_list": self.cache_tools_list,
            "name": self.name,
            "client_session_timeout_seconds": self.client_session_timeout_seconds,
            "tool_filter": tool_filter,
            "use_structured_content": self.use_structured_content,
            "max_retry_attempts": self.max_retry_attempts,
            "retry_backoff_seconds_base": self.retry_backoff_seconds_base,
        }

        if self.transport == "stdio":
            params: dict[str, Any] = {"command": self.command}
            if self.args:
                params["args"] = self.args
            if self.env:
                params["env"] = self.env
            if self.cwd:
                params["cwd"] = self.cwd
            return MCPServerStdio(params=params, **common_kwargs)

        params = {"url": self.url}
        if self.headers:
            params["headers"] = self.headers
        if self.timeout is not None:
            params["timeout"] = self.timeout
        if self.sse_read_timeout is not None:
            params["sse_read_timeout"] = self.sse_read_timeout

        if self.transport == "sse":
            return MCPServerSse(params=params, **common_kwargs)

        if self.terminate_on_close is not None:
            params["terminate_on_close"] = self.terminate_on_close
        if self.ignore_initialized_notification_failure is not None:
            params["ignore_initialized_notification_failure"] = (
                self.ignore_initialized_notification_failure
            )
        return MCPServerStreamableHttp(params=params, **common_kwargs)


@dataclass(frozen=True, slots=True)
class MCPServerBinding:
    name: str
    targets: frozenset[str]
    server: MCPServer


@dataclass(frozen=True, slots=True)
class CareChatMCPRegistry:
    bindings: tuple[MCPServerBinding, ...]

    @property
    def server_names(self) -> list[str]:
        return [binding.name for binding in self.bindings]

    @property
    def all_servers(self) -> list[MCPServer]:
        servers: list[MCPServer] = []
        seen: set[MCPServer] = set()
        for binding in self.bindings:
            if binding.server in seen:
                continue
            seen.add(binding.server)
            servers.append(binding.server)
        return servers

    def resolve_for(
        self,
        agent_key: str,
        *,
        active_mcp_servers: list[MCPServer] | None = None,
    ) -> list[MCPServer]:
        allowed_servers = set(active_mcp_servers) if active_mcp_servers is not None else None
        resolved: list[MCPServer] = []
        seen: set[MCPServer] = set()
        for binding in self.bindings:
            if agent_key not in binding.targets:
                continue
            if allowed_servers is not None and binding.server not in allowed_servers:
                continue
            if binding.server in seen:
                continue
            seen.add(binding.server)
            resolved.append(binding.server)
        return resolved

    def build_manager(self, settings: Settings) -> MCPServerManager:
        return MCPServerManager(
            self.all_servers,
            connect_timeout_seconds=settings.care_chat_mcp_connect_timeout_seconds,
            cleanup_timeout_seconds=settings.care_chat_mcp_cleanup_timeout_seconds,
            strict=settings.care_chat_mcp_strict,
            connect_in_parallel=settings.care_chat_mcp_connect_in_parallel,
        )


def _load_config_text(settings: Settings) -> str:
    config_path = settings.care_chat_mcp_config_path
    config_json = settings.care_chat_mcp_config_json
    if config_path is not None and config_json:
        raise ValueError(
            "CARE_CHAT_MCP_CONFIG_PATH and CARE_CHAT_MCP_CONFIG_JSON cannot be set together."
        )
    if config_path is not None:
        resolved_path = config_path if config_path.is_absolute() else PROJECT_ROOT / config_path
        if not resolved_path.exists():
            raise ValueError(f"MCP config file does not exist: {resolved_path}")
        return resolved_path.read_text(encoding="utf-8")
    if config_json:
        return config_json
    raise ValueError(
        "CARE_CHAT_MCP_ENABLED=true but no MCP config source was provided. "
        "Set CARE_CHAT_MCP_CONFIG_PATH or CARE_CHAT_MCP_CONFIG_JSON."
    )


def _load_server_items(settings: Settings) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_load_config_text(settings))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse MCP config JSON: {exc}") from exc

    if isinstance(payload, dict):
        payload = payload.get("servers")

    if not isinstance(payload, list):
        raise ValueError("MCP config must be a JSON array or an object with a 'servers' array.")

    items: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"MCP config entry #{index} must be a JSON object.")
        items.append(item)
    return items


def build_mcp_registry(settings: Settings) -> CareChatMCPRegistry | None:
    if not settings.care_chat_mcp_enabled:
        return None

    names: set[str] = set()
    bindings: list[MCPServerBinding] = []
    for item in _load_server_items(settings):
        try:
            definition = MCPServerDefinition.model_validate(_expand_env_values(item))
        except ValidationError as exc:
            raise ValueError(f"Invalid MCP server config: {exc}") from exc

        if not definition.enabled:
            continue
        if definition.name in names:
            raise ValueError(f"Duplicate MCP server name: {definition.name}")

        names.add(definition.name)
        bindings.append(
            MCPServerBinding(
                name=definition.name,
                targets=_expand_targets(definition.targets),
                server=definition.build_server(),
            )
        )

    if not bindings:
        return None
    return CareChatMCPRegistry(bindings=tuple(bindings))
