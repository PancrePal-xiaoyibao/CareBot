from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from agents import (
    Agent,
    GuardrailFunctionOutput,
    ModelSettings,
    RunConfig,
    Runner,
    input_guardrail,
    output_guardrail,
    set_default_openai_api,
    set_default_openai_client,
)
from agents.models.interface import ModelProvider
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider
from openai import AsyncOpenAI
from openai.types.shared.reasoning import Reasoning
from pydantic import ValidationError

from .config import Settings
from .kimi_compat import install_kimi_chat_compatibility, should_replay_kimi_reasoning_content
from .mcp import (
    CAREGIVER_COORDINATION_AGENT_KEY,
    CAREGIVER_COORDINATOR_AGENT_KEY,
    CAREGIVER_EMOTIONAL_AGENT_KEY,
    CAREGIVER_URGENT_AGENT_KEY,
    PATIENT_COORDINATOR_AGENT_KEY,
    PATIENT_EMOTIONAL_AGENT_KEY,
    PATIENT_NAVIGATION_AGENT_KEY,
    PATIENT_URGENT_AGENT_KEY,
    ROUTER_AGENT_KEY,
    VOLUNTEER_BOUNDARY_AGENT_KEY,
    VOLUNTEER_COORDINATOR_AGENT_KEY,
    VOLUNTEER_ESCALATION_AGENT_KEY,
    VOLUNTEER_TASK_AGENT_KEY,
)
from .prompts import (
    CRISIS_ALERT_TOOL_POLICY,
    caregiver_care_coordination_prompt,
    caregiver_coordinator_prompt,
    caregiver_emotional_support_prompt,
    caregiver_urgent_support_prompt,
    crisis_monitor_prompt,
    input_guardrail_prompt,
    output_guardrail_prompt,
    patient_care_navigation_prompt,
    patient_coordinator_prompt,
    patient_emotional_support_prompt,
    patient_urgent_support_prompt,
    role_router_prompt,
    volunteer_boundary_coach_prompt,
    volunteer_coordinator_prompt,
    volunteer_escalation_guide_prompt,
    volunteer_task_guide_prompt,
)
from .safety import detect_local_crisis
from .schemas import (
    CareChatContext,
    CrisisAssessment,
    InputSafetyAssessment,
    OutputSafetyAssessment,
)
from .tools import (
    build_crisis_alert_tool,
    caregiver_coordination_plan,
    community_help_request,
    community_peer_referral,
    doctor_question_builder,
    grounding_exercise,
    symptom_journal_template,
    urgent_support_playbook,
    volunteer_support_boundaries,
)

if TYPE_CHECKING:
    from agents.mcp import MCPServer

    from .mcp import CareChatMCPRegistry


def _is_kimi_k25(model_name: str) -> bool:
    return "kimi-k2.5" in model_name.lower()


def _kimi_must_disable_thinking_mode(settings: Settings) -> bool:
    """Nested handoff architecture always produces multi-turn histories where the SDK
    does not reliably replay Kimi reasoning on tool messages."""

    if settings.care_chat_enable_llm_output_guardrail:
        return True
    # Two-level handoffs are always active in the current architecture.
    return True


def configure_openai_runtime(settings: Settings) -> None:
    settings.ensure_ready()
    install_kimi_chat_compatibility()

    # The app already retries/rebuilds MCP runtime on recoverable transport failures.
    # Suppress raw transport stack traces from bubbling into the CLI.
    logging.getLogger("mcp.client.streamable_http").setLevel(logging.CRITICAL)
    logging.getLogger("mcp.client.sse").setLevel(logging.CRITICAL)

    client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        client_kwargs["base_url"] = settings.openai_base_url

    set_default_openai_client(AsyncOpenAI(**client_kwargs))
    set_default_openai_api(settings.care_chat_openai_api)


class CareChatModelProvider(ModelProvider):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._responses_provider = OpenAIProvider(
            openai_client=self._client,
            use_responses=True,
        )

    def get_model(self, model_name: str | None):
        resolved_model_name = model_name or self.settings.care_chat_model
        if self.settings.care_chat_openai_api == "responses":
            return self._responses_provider.get_model(resolved_model_name)

        return OpenAIChatCompletionsModel(
            model=resolved_model_name,
            openai_client=self._client,
            should_replay_reasoning_content=should_replay_kimi_reasoning_content,
        )

    async def aclose(self) -> None:
        await self._responses_provider.aclose()


def build_model_provider(settings: Settings) -> CareChatModelProvider:
    return CareChatModelProvider(settings)


def build_run_config(
    settings: Settings,
    *,
    workflow_name: str,
    tracing_disabled: bool | None = None,
    group_id: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
) -> RunConfig:
    resolved_tracing_disabled = (
        settings.tracing_effective_disabled
        if tracing_disabled is None
        else (tracing_disabled or settings.tracing_effective_disabled)
    )

    return RunConfig(
        model_provider=build_model_provider(settings),
        session_settings=settings.build_session_settings(),
        tracing_disabled=resolved_tracing_disabled,
        tracing=None if resolved_tracing_disabled else settings.build_tracing_config(),
        trace_include_sensitive_data=settings.care_chat_trace_include_sensitive_data,
        workflow_name=workflow_name,
        group_id=group_id,
        trace_metadata=trace_metadata,
    )


def _build_model_settings_for_model(settings: Settings, model_name: str) -> ModelSettings:
    temperature = settings.care_chat_temperature
    if model_name == "moonshot/kimi-k2.5":
        temperature = 1.0

    reasoning: Reasoning | None = Reasoning(effort=settings.care_chat_reasoning_effort)
    extra_body: dict[str, Any] | None = None
    if _is_kimi_k25(model_name):
        reasoning = None
        if settings.care_chat_openai_api == "chat_completions":
            thinking_on = (
                settings.care_chat_enable_thinking
                and not _kimi_must_disable_thinking_mode(settings)
            )
            extra_body = (
                {"thinking": {"type": "enabled"}}
                if thinking_on
                else {"thinking": {"type": "disabled"}}
            )

    return ModelSettings(
        temperature=temperature,
        max_tokens=settings.care_chat_max_tokens,
        verbosity=settings.care_chat_verbosity,
        parallel_tool_calls=False,
        reasoning=reasoning,
        extra_body=extra_body,
    )


def _build_model_settings(settings: Settings) -> ModelSettings:
    return _build_model_settings_for_model(settings, settings.care_chat_model)


def _build_role_router_model_settings(settings: Settings) -> ModelSettings:
    return _build_model_settings_for_model(settings, settings.role_router_model)


def _build_guardrail_model_settings(settings: Settings) -> ModelSettings:
    temperature = 0.0
    if settings.guardrail_model == "moonshot/kimi-k2.5":
        temperature = 1.0

    reasoning: Reasoning | None = Reasoning(effort=settings.care_chat_guardrail_reasoning_effort)
    extra_body: dict[str, Any] | None = None
    if _is_kimi_k25(settings.guardrail_model):
        reasoning = None
        extra_body = {"thinking": {"type": "disabled"}}

    return ModelSettings(
        temperature=temperature,
        max_tokens=250,
        verbosity="low",
        parallel_tool_calls=False,
        reasoning=reasoning,
        extra_body=extra_body,
    )


def _stringify_input(raw_input: str | list[Any]) -> str:
    if isinstance(raw_input, str):
        return raw_input

    return "\n".join(str(item) for item in raw_input)


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        cleaned = cleaned[start : end + 1]

    return json.loads(cleaned)


def _parse_input_assessment(raw_output: str, original_input: str) -> InputSafetyAssessment:
    try:
        payload = _extract_json_object(raw_output)
        return InputSafetyAssessment.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        local_alert = detect_local_crisis(original_input)
        if local_alert:
            return InputSafetyAssessment(
                tripwire_triggered=True,
                category=local_alert.category.value,
                reason="Fallback local crisis detection matched emergency keywords.",
            )

        return InputSafetyAssessment(
            tripwire_triggered=False,
            category="none",
            reason="Classifier output could not be parsed; local fallback found no crisis keywords.",
        )


def _parse_output_assessment(raw_output: str) -> OutputSafetyAssessment:
    try:
        payload = _extract_json_object(raw_output)
        return OutputSafetyAssessment.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return OutputSafetyAssessment(
            unsafe=False,
            reason="Classifier output could not be parsed; leaving output unblocked.",
        )


def _parse_crisis_assessment(raw_output: str) -> CrisisAssessment:
    try:
        payload = _extract_json_object(raw_output)
        return CrisisAssessment.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return CrisisAssessment(
            risk_level="none",
            reason="Classifier output could not be parsed; defaulting to no risk.",
        )


def build_crisis_monitor_agent(settings: Settings) -> Agent[CareChatContext]:
    return Agent(
        name="Crisis Risk Monitor",
        instructions=crisis_monitor_prompt(),
        model=settings.guardrail_model,
        model_settings=_build_guardrail_model_settings(settings),
    )


def build_care_agent(
    settings: Settings,
    *,
    mcp_registry: CareChatMCPRegistry | None = None,
    active_mcp_servers: list[MCPServer] | None = None,
) -> Agent[CareChatContext]:
    guardrail_model_settings = _build_guardrail_model_settings(settings)

    input_guardrails = []
    output_guardrails = []

    if settings.care_chat_enable_llm_input_guardrail:
        input_classifier = Agent(
            name="Critical Risk Input Classifier",
            instructions=input_guardrail_prompt(),
            model=settings.guardrail_model,
            model_settings=guardrail_model_settings,
        )

        @input_guardrail(name="critical-risk-filter", run_in_parallel=False)
        async def critical_risk_filter(ctx, agent, raw_input):
            original_input = _stringify_input(raw_input)
            if not original_input.strip():
                return GuardrailFunctionOutput(
                    output_info=InputSafetyAssessment(
                        tripwire_triggered=False,
                        category="none",
                        reason="Empty input; skipping LLM classifier.",
                    ),
                    tripwire_triggered=False,
                )
            result = await Runner.run(
                input_classifier,
                original_input,
                context=ctx.context,
                max_turns=1,
                run_config=build_run_config(
                    settings,
                    workflow_name="CareChat input safety",
                    tracing_disabled=True,
                ),
            )
            assessment = _parse_input_assessment(str(result.final_output), original_input)
            return GuardrailFunctionOutput(
                output_info=assessment,
                tripwire_triggered=assessment.tripwire_triggered,
            )

        input_guardrails.append(critical_risk_filter)

    if settings.care_chat_enable_llm_output_guardrail:
        output_classifier = Agent(
            name="Medical Safety Output Classifier",
            instructions=output_guardrail_prompt(),
            model=settings.guardrail_model,
            model_settings=guardrail_model_settings,
        )

        @output_guardrail(name="medical-safety-filter")
        async def medical_safety_filter(ctx, agent, agent_output):
            text_out = str(agent_output).strip()
            if not text_out:
                return GuardrailFunctionOutput(
                    output_info=OutputSafetyAssessment(
                        unsafe=False,
                        reason="Empty assistant output chunk; skipping LLM classifier.",
                    ),
                    tripwire_triggered=False,
                )
            result = await Runner.run(
                output_classifier,
                text_out,
                context=ctx.context,
                max_turns=1,
                run_config=build_run_config(
                    settings,
                    workflow_name="CareChat output safety",
                    tracing_disabled=True,
                ),
            )
            assessment = _parse_output_assessment(str(result.final_output))
            return GuardrailFunctionOutput(
                output_info=assessment,
                tripwire_triggered=assessment.unsafe,
            )

        output_guardrails.append(medical_safety_filter)

    shared_agent_config = {
        "model": settings.care_chat_model,
        "model_settings": _build_model_settings(settings),
        "output_guardrails": output_guardrails,
        "mcp_config": {
            "convert_schemas_to_strict": settings.care_chat_mcp_convert_schemas_to_strict,
        },
    }

    crisis_tool = (
        build_crisis_alert_tool(settings)
        if settings.care_chat_crisis_alert_enabled
        else None
    )
    crisis_policy_suffix = (
        f"\n\n{CRISIS_ALERT_TOOL_POLICY}" if crisis_tool else ""
    )

    def _agent_kwargs(agent_key: str) -> dict[str, Any]:
        if mcp_registry is None:
            return {}
        return {
            "mcp_servers": mcp_registry.resolve_for(
                agent_key,
                active_mcp_servers=active_mcp_servers,
            )
        }

    # === Level 2: Patient sub-agents ===

    patient_emotional = Agent(
        name="Patient Emotional Support",
        handoff_description=(
            "Best for patient fear, sadness, loneliness, overwhelm, grief, "
            "and emotional containment."
        ),
        instructions=patient_emotional_support_prompt() + crisis_policy_suffix,
        tools=[grounding_exercise, community_peer_referral] + ([crisis_tool] if crisis_tool else []),
        **shared_agent_config,
        **_agent_kwargs(PATIENT_EMOTIONAL_AGENT_KEY),
    )

    patient_navigation = Agent(
        name="Patient Care Navigation",
        handoff_description=(
            "Best for patient visit prep, symptom tracking, and questions "
            "for clinicians."
        ),
        instructions=patient_care_navigation_prompt(),
        tools=[doctor_question_builder, symptom_journal_template],
        **shared_agent_config,
        **_agent_kwargs(PATIENT_NAVIGATION_AGENT_KEY),
    )

    patient_urgent = Agent(
        name="Patient Urgent Support",
        handoff_description=(
            "Best for patient new or worsening symptoms and escalation decisions."
        ),
        instructions=patient_urgent_support_prompt() + crisis_policy_suffix,
        tools=[urgent_support_playbook] + ([crisis_tool] if crisis_tool else []),
        **shared_agent_config,
        **_agent_kwargs(PATIENT_URGENT_AGENT_KEY),
    )

    # === Level 2: Caregiver sub-agents ===

    caregiver_emotional = Agent(
        name="Caregiver Emotional Support",
        handoff_description=(
            "Best for caregiver burnout, strain, guilt, and emotional support."
        ),
        instructions=caregiver_emotional_support_prompt() + crisis_policy_suffix,
        tools=[grounding_exercise, community_peer_referral] + ([crisis_tool] if crisis_tool else []),
        **shared_agent_config,
        **_agent_kwargs(CAREGIVER_EMOTIONAL_AGENT_KEY),
    )

    caregiver_coordination = Agent(
        name="Caregiver Care Coordination",
        handoff_description=(
            "Best for home coordination, observation tracking, care team "
            "communication, and asking for help."
        ),
        instructions=caregiver_care_coordination_prompt(),
        tools=[
            doctor_question_builder,
            symptom_journal_template,
            caregiver_coordination_plan,
            community_help_request,
        ],
        **shared_agent_config,
        **_agent_kwargs(CAREGIVER_COORDINATION_AGENT_KEY),
    )

    caregiver_urgent = Agent(
        name="Caregiver Urgent Support",
        handoff_description=(
            "Best for concerning patient symptoms observed by caregiver "
            "and escalation decisions."
        ),
        instructions=caregiver_urgent_support_prompt() + crisis_policy_suffix,
        tools=[urgent_support_playbook] + ([crisis_tool] if crisis_tool else []),
        **shared_agent_config,
        **_agent_kwargs(CAREGIVER_URGENT_AGENT_KEY),
    )

    # === Level 2: Volunteer sub-agents ===

    volunteer_task = Agent(
        name="Volunteer Task Guide",
        handoff_description=(
            "Best for practical help logistics: transport, meals, errands, "
            "companionship."
        ),
        instructions=volunteer_task_guide_prompt(),
        tools=[community_help_request],
        **shared_agent_config,
        **_agent_kwargs(VOLUNTEER_TASK_AGENT_KEY),
    )

    volunteer_boundary = Agent(
        name="Volunteer Boundary Coach",
        handoff_description=(
            "Best for scope questions, privacy, and what volunteers should "
            "or shouldn't do."
        ),
        instructions=volunteer_boundary_coach_prompt(),
        tools=[volunteer_support_boundaries],
        **shared_agent_config,
        **_agent_kwargs(VOLUNTEER_BOUNDARY_AGENT_KEY),
    )

    volunteer_escalation = Agent(
        name="Volunteer Escalation Guide",
        handoff_description=(
            "Best for concerning observations and when or how to escalate "
            "to family or medical team."
        ),
        instructions=volunteer_escalation_guide_prompt() + crisis_policy_suffix,
        tools=[urgent_support_playbook] + ([crisis_tool] if crisis_tool else []),
        **shared_agent_config,
        **_agent_kwargs(VOLUNTEER_ESCALATION_AGENT_KEY),
    )

    # === Level 1: Coordinators ===

    patient_coordinator = Agent(
        name="Patient Companion",
        handoff_description=(
            "Best for cancer patients asking for emotional support, symptom "
            "organization, visit prep, or self-management guidance."
        ),
        instructions=patient_coordinator_prompt(),
        handoffs=[patient_emotional, patient_navigation, patient_urgent],
        **shared_agent_config,
        **_agent_kwargs(PATIENT_COORDINATOR_AGENT_KEY),
    )

    caregiver_coordinator = Agent(
        name="Caregiver Support Coach",
        handoff_description=(
            "Best for family caregivers or care partners coordinating home "
            "care, monitoring changes, and communicating with clinicians."
        ),
        instructions=caregiver_coordinator_prompt(),
        handoffs=[caregiver_emotional, caregiver_coordination, caregiver_urgent],
        **shared_agent_config,
        **_agent_kwargs(CAREGIVER_COORDINATOR_AGENT_KEY),
    )

    volunteer_coordinator = Agent(
        name="Community Volunteer Guide",
        handoff_description=(
            "Best for volunteers, neighbors, or non-family helpers who need "
            "boundaries, logistics support, or escalation guidance."
        ),
        instructions=volunteer_coordinator_prompt(),
        handoffs=[volunteer_task, volunteer_boundary, volunteer_escalation],
        **shared_agent_config,
        **_agent_kwargs(VOLUNTEER_COORDINATOR_AGENT_KEY),
    )

    # === Level 0: Triage Router ===

    role_router_config = {
        **shared_agent_config,
        "model": settings.role_router_model,
        "model_settings": _build_role_router_model_settings(settings),
    }

    return Agent(
        name="Care Chat Router",
        instructions=role_router_prompt,
        handoffs=[
            patient_coordinator,
            caregiver_coordinator,
            volunteer_coordinator,
        ],
        input_guardrails=input_guardrails,
        **role_router_config,
        **_agent_kwargs(ROUTER_AGENT_KEY),
    )
