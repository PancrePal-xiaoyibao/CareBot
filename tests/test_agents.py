from pathlib import Path

from care_chat.agents import (
    _build_guardrail_model_settings,
    _build_model_settings,
    build_care_agent,
    build_run_config,
)
from care_chat.config import Settings


def test_kimi_model_settings_disables_thinking_with_nested_handoffs() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
        care_chat_model="moonshot/kimi-k2.5",
        care_chat_openai_api="chat_completions",
        care_chat_enable_thinking=True,
    )

    model_settings = _build_model_settings(settings)

    assert model_settings.parallel_tool_calls is False
    assert model_settings.reasoning is None
    assert model_settings.extra_body == {"thinking": {"type": "disabled"}}


def test_kimi_sets_thinking_disabled_when_user_disables_thinking_flag() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
        care_chat_model="moonshot/kimi-k2.5",
        care_chat_openai_api="chat_completions",
        care_chat_enable_thinking=False,
    )

    assert _build_model_settings(settings).extra_body == {"thinking": {"type": "disabled"}}


def test_guardrail_model_settings_stay_lightweight() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
        care_chat_model="moonshot/kimi-k2.5",
        care_chat_guardrail_model="moonshot/kimi-k2.5",
    )

    model_settings = _build_guardrail_model_settings(settings)

    assert model_settings.parallel_tool_calls is False
    assert model_settings.extra_body == {"thinking": {"type": "disabled"}}


def test_build_care_agent_creates_nested_routing_hierarchy() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
    )

    agent = build_care_agent(settings)

    assert agent.name == "Care Chat Router"

    coordinator_names = [h.name for h in agent.handoffs]
    assert coordinator_names == [
        "Patient Companion",
        "Caregiver Support Coach",
        "Community Volunteer Guide",
    ]

    patient_coordinator = agent.handoffs[0]
    assert [h.name for h in patient_coordinator.handoffs] == [
        "Patient Emotional Support",
        "Patient Care Navigation",
        "Patient Urgent Support",
    ]

    caregiver_coordinator = agent.handoffs[1]
    assert [h.name for h in caregiver_coordinator.handoffs] == [
        "Caregiver Emotional Support",
        "Caregiver Care Coordination",
        "Caregiver Urgent Support",
    ]

    volunteer_coordinator = agent.handoffs[2]
    assert [h.name for h in volunteer_coordinator.handoffs] == [
        "Volunteer Task Guide",
        "Volunteer Boundary Coach",
        "Volunteer Escalation Guide",
    ]


def test_sub_agents_have_expected_tools() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
    )

    agent = build_care_agent(settings)

    patient = agent.handoffs[0]
    patient_emotional = patient.handoffs[0]
    patient_navigation = patient.handoffs[1]
    patient_urgent = patient.handoffs[2]
    assert [t.name for t in patient_emotional.tools] == ["grounding_exercise", "community_peer_referral"]
    assert [t.name for t in patient_navigation.tools] == [
        "doctor_question_builder",
        "symptom_journal_template",
    ]
    assert [t.name for t in patient_urgent.tools] == ["urgent_support_playbook"]

    caregiver = agent.handoffs[1]
    caregiver_coordination = caregiver.handoffs[1]
    assert [t.name for t in caregiver_coordination.tools] == [
        "doctor_question_builder",
        "symptom_journal_template",
        "caregiver_coordination_plan",
        "community_help_request",
    ]

    volunteer = agent.handoffs[2]
    volunteer_task = volunteer.handoffs[0]
    volunteer_boundary = volunteer.handoffs[1]
    volunteer_escalation = volunteer.handoffs[2]
    assert [t.name for t in volunteer_task.tools] == ["community_help_request"]
    assert [t.name for t in volunteer_boundary.tools] == ["volunteer_support_boundaries"]
    assert [t.name for t in volunteer_escalation.tools] == ["urgent_support_playbook"]


def test_build_run_config_uses_official_tracing_fields() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="trace-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
    )

    run_config = build_run_config(
        settings,
        workflow_name="Care Chat Test Workflow",
        group_id="session-123",
        trace_metadata={"role_hint": "patient"},
    )

    assert run_config.tracing_disabled is False
    assert run_config.tracing == {"api_key": "trace-key"}
    assert run_config.group_id == "session-123"
    assert run_config.trace_metadata == {"role_hint": "patient"}
