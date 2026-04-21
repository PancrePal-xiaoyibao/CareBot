import json
from pathlib import Path

from care_chat.agents import build_care_agent
from care_chat.config import Settings
from care_chat.mcp import (
    CAREGIVER_COORDINATION_AGENT_KEY,
    PATIENT_COORDINATOR_AGENT_KEY,
    PATIENT_NAVIGATION_AGENT_KEY,
    ROUTER_AGENT_KEY,
    VOLUNTEER_TASK_AGENT_KEY,
    build_mcp_registry,
)


def test_build_mcp_registry_supports_shared_and_dedicated_targets() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
        care_chat_mcp_enabled=True,
        care_chat_mcp_config_json=json.dumps(
            [
                {
                    "name": "shared-kb",
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["shared-server"],
                    "targets": ["patient", "caregiver"],
                },
                {
                    "name": "patient-only",
                    "transport": "streamable_http",
                    "url": "https://patient.example.com/mcp",
                    "targets": ["patient_navigation"],
                },
                {
                    "name": "router-common",
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["router-server"],
                    "targets": ["router"],
                },
            ]
        ),
    )

    registry = build_mcp_registry(settings)

    assert registry is not None
    assert [server.name for server in registry.resolve_for(PATIENT_COORDINATOR_AGENT_KEY)] == [
        "shared-kb"
    ]
    assert [server.name for server in registry.resolve_for(PATIENT_NAVIGATION_AGENT_KEY)] == [
        "shared-kb",
        "patient-only",
    ]
    assert [server.name for server in registry.resolve_for(CAREGIVER_COORDINATION_AGENT_KEY)] == [
        "shared-kb"
    ]
    assert [server.name for server in registry.resolve_for(ROUTER_AGENT_KEY)] == [
        "router-common"
    ]
    assert registry.resolve_for(VOLUNTEER_TASK_AGENT_KEY) == []


def test_build_care_agent_attaches_mcp_servers_to_matching_agents() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
        care_chat_mcp_enabled=True,
        care_chat_mcp_config_json=json.dumps(
            [
                {
                    "name": "shared-kb",
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["shared-server"],
                    "targets": ["patient"],
                },
                {
                    "name": "patient-navigation-only",
                    "transport": "stdio",
                    "command": "uvx",
                    "args": ["patient-navigation-server"],
                    "targets": ["patient_navigation"],
                },
            ]
        ),
    )

    registry = build_mcp_registry(settings)
    assert registry is not None

    agent = build_care_agent(
        settings,
        mcp_registry=registry,
        active_mcp_servers=registry.all_servers,
    )

    patient_coordinator = agent.handoffs[0]
    patient_navigation = patient_coordinator.handoffs[1]
    volunteer_coordinator = agent.handoffs[2]

    assert [server.name for server in patient_coordinator.mcp_servers] == ["shared-kb"]
    assert [server.name for server in patient_navigation.mcp_servers] == [
        "shared-kb",
        "patient-navigation-only",
    ]
    assert volunteer_coordinator.mcp_servers == []
