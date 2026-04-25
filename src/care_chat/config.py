from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from agents import SessionSettings

from .schemas import CareRoleHint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_DB = PROJECT_ROOT / ".local" / "care_chat_sessions.sqlite3"

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
Verbosity = Literal["low", "medium", "high"]
OpenAIApi = Literal["responses", "chat_completions"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    care_chat_model: str = Field(
        default="moonshot/kimi-k2.5",
        validation_alias="CARE_CHAT_MODEL",
    )
    care_chat_guardrail_model: str | None = Field(
        default=None,
        validation_alias="CARE_CHAT_GUARDRAIL_MODEL",
    )
    care_chat_role_router_model: str | None = Field(
        default=None,
        validation_alias="CARE_CHAT_ROLE_ROUTER_MODEL",
    )
    care_chat_openai_api: OpenAIApi = Field(
        default="chat_completions",
        validation_alias="CARE_CHAT_OPENAI_API",
    )
    care_chat_reasoning_effort: ReasoningEffort = Field(
        default="minimal",
        validation_alias="CARE_CHAT_REASONING_EFFORT",
    )
    care_chat_guardrail_reasoning_effort: ReasoningEffort = Field(
        default="minimal",
        validation_alias="CARE_CHAT_GUARDRAIL_REASONING_EFFORT",
    )
    care_chat_temperature: float | None = Field(
        default=0.7,
        validation_alias="CARE_CHAT_TEMPERATURE",
    )
    care_chat_max_tokens: int | None = Field(
        default=4000,
        validation_alias="CARE_CHAT_MAX_TOKENS",
    )
    care_chat_verbosity: Verbosity = Field(
        default="high",
        validation_alias="CARE_CHAT_VERBOSITY",
    )
    care_chat_session_db_path: Path = Field(
        default=DEFAULT_SESSION_DB,
        validation_alias="CARE_CHAT_SESSION_DB_PATH",
    )
    care_chat_session_history_limit: int | None = Field(
        default=None,
        validation_alias="CARE_CHAT_SESSION_HISTORY_LIMIT",
    )
    care_chat_enable_thinking: bool = Field(
        default=True,
        validation_alias="CARE_CHAT_ENABLE_THINKING",
    )
    care_chat_default_role_hint: CareRoleHint = Field(
        default="auto",
        validation_alias="CARE_CHAT_DEFAULT_ROLE_HINT",
    )
    care_chat_enable_llm_input_guardrail: bool = Field(
        default=False,
        validation_alias="CARE_CHAT_ENABLE_LLM_INPUT_GUARDRAIL",
    )
    care_chat_enable_llm_output_guardrail: bool = Field(
        default=False,
        validation_alias="CARE_CHAT_ENABLE_LLM_OUTPUT_GUARDRAIL",
    )
    care_chat_tracing_api_key: str = Field(
        default="",
        validation_alias="CARE_CHAT_TRACING_API_KEY",
    )
    care_chat_tracing_disabled: bool = Field(
        default=False,
        validation_alias="CARE_CHAT_TRACING_DISABLED",
    )
    care_chat_trace_include_sensitive_data: bool = Field(
        default=False,
        validation_alias="CARE_CHAT_TRACE_INCLUDE_SENSITIVE_DATA",
    )
    care_chat_mcp_enabled: bool = Field(
        default=False,
        validation_alias="CARE_CHAT_MCP_ENABLED",
    )
    care_chat_mcp_config_path: Path | None = Field(
        default=None,
        validation_alias="CARE_CHAT_MCP_CONFIG_PATH",
    )
    care_chat_mcp_config_json: str | None = Field(
        default=None,
        validation_alias="CARE_CHAT_MCP_CONFIG_JSON",
    )
    care_chat_mcp_strict: bool = Field(
        default=False,
        validation_alias="CARE_CHAT_MCP_STRICT",
    )
    care_chat_mcp_connect_timeout_seconds: float | None = Field(
        default=10.0,
        validation_alias="CARE_CHAT_MCP_CONNECT_TIMEOUT_SECONDS",
    )
    care_chat_mcp_cleanup_timeout_seconds: float | None = Field(
        default=10.0,
        validation_alias="CARE_CHAT_MCP_CLEANUP_TIMEOUT_SECONDS",
    )
    care_chat_mcp_connect_in_parallel: bool = Field(
        default=True,
        validation_alias="CARE_CHAT_MCP_CONNECT_IN_PARALLEL",
    )
    care_chat_mcp_convert_schemas_to_strict: bool = Field(
        default=False,
        validation_alias="CARE_CHAT_MCP_CONVERT_SCHEMAS_TO_STRICT",
    )
    care_chat_language: str = Field(
        default="zh-CN",
        validation_alias="CARE_CHAT_LANGUAGE",
    )
    care_chat_crisis_alert_enabled: bool = Field(
        default=False,
        validation_alias="CARE_CHAT_CRISIS_ALERT_ENABLED",
    )
    care_chat_crisis_alert_smtp_host: str = Field(
        default="",
        validation_alias="CARE_CHAT_CRISIS_ALERT_SMTP_HOST",
    )
    care_chat_crisis_alert_smtp_port: int = Field(
        default=465,
        validation_alias="CARE_CHAT_CRISIS_ALERT_SMTP_PORT",
    )
    care_chat_crisis_alert_smtp_user: str = Field(
        default="",
        validation_alias="CARE_CHAT_CRISIS_ALERT_SMTP_USER",
    )
    care_chat_crisis_alert_smtp_password: str = Field(
        default="",
        validation_alias="CARE_CHAT_CRISIS_ALERT_SMTP_PASSWORD",
    )
    care_chat_crisis_alert_recipient: str = Field(
        default="",
        validation_alias="CARE_CHAT_CRISIS_ALERT_RECIPIENT",
    )
    care_chat_crisis_alert_cooldown_minutes: int = Field(
        default=10,
        validation_alias="CARE_CHAT_CRISIS_ALERT_COOLDOWN_MINUTES",
    )

    @field_validator("care_chat_session_history_limit", mode="before")
    @classmethod
    def _empty_limit_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator("care_chat_mcp_config_path", "care_chat_mcp_config_json", mode="before")
    @classmethod
    def _empty_mcp_config_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @property
    def guardrail_model(self) -> str:
        return self.care_chat_guardrail_model or self.care_chat_model

    @property
    def role_router_model(self) -> str:
        return self.care_chat_role_router_model or self.care_chat_model

    def ensure_ready(self) -> None:
        if not self.openai_api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to .env before running care-chat."
            )

    @property
    def uses_custom_openai_base_url(self) -> bool:
        if not self.openai_base_url:
            return False
        return "api.openai.com" not in self.openai_base_url

    @property
    def tracing_export_api_key(self) -> str | None:
        explicit_key = self.care_chat_tracing_api_key.strip()
        if explicit_key:
            return explicit_key

        if self.uses_custom_openai_base_url:
            return None

        default_key = self.openai_api_key.strip()
        return default_key or None

    @property
    def tracing_effective_disabled(self) -> bool:
        if self.care_chat_tracing_disabled:
            return True
        return self.tracing_export_api_key is None

    def build_tracing_config(self) -> dict[str, str] | None:
        api_key = self.tracing_export_api_key
        if not api_key:
            return None
        return {"api_key": api_key}

    def tracing_warning(self) -> str | None:
        if self.care_chat_tracing_disabled:
            return None
        if self.uses_custom_openai_base_url and not self.care_chat_tracing_api_key.strip():
            return (
                "当前模型请求走第三方 OPENAI_BASE_URL。若要把 traces 发到 OpenAI Traces dashboard，"
                "请设置 CARE_CHAT_TRACING_API_KEY 为官方 OpenAI API key。"
            )
        return None

    @property
    def mcp_config_source(self) -> str:
        if self.care_chat_mcp_config_path is not None:
            return f"path:{self.care_chat_mcp_config_path}"
        if self.care_chat_mcp_config_json:
            return "inline_json"
        return "unset"

    def safe_summary(self) -> dict[str, str]:
        return {
            "model": self.care_chat_model,
            "guardrail_model": self.guardrail_model,
            "role_router_model": self.role_router_model,
            "openai_api": self.care_chat_openai_api,
            "base_url": self.openai_base_url or "default",
            "session_db_path": str(self.care_chat_session_db_path),
            "session_history_limit": (
                "all"
                if self.care_chat_session_history_limit is None
                else str(self.care_chat_session_history_limit)
            ),
            "thinking": str(self.care_chat_enable_thinking).lower(),
            "default_role_hint": self.care_chat_default_role_hint,
            "llm_input_guardrail": str(self.care_chat_enable_llm_input_guardrail).lower(),
            "llm_output_guardrail": str(self.care_chat_enable_llm_output_guardrail).lower(),
            "language": self.care_chat_language,
            "tracing_disabled": str(self.tracing_effective_disabled).lower(),
            "tracing_export_api_key_configured": (
                "yes" if bool(self.tracing_export_api_key) else "no"
            ),
            "trace_include_sensitive_data": (
                str(self.care_chat_trace_include_sensitive_data).lower()
            ),
            "mcp_enabled": str(self.care_chat_mcp_enabled).lower(),
            "mcp_config_source": self.mcp_config_source,
            "mcp_strict": str(self.care_chat_mcp_strict).lower(),
            "mcp_connect_in_parallel": str(self.care_chat_mcp_connect_in_parallel).lower(),
            "mcp_convert_schemas_to_strict": (
                str(self.care_chat_mcp_convert_schemas_to_strict).lower()
            ),
            "crisis_alert_enabled": str(self.care_chat_crisis_alert_enabled).lower(),
            "api_key_configured": "yes" if bool(self.openai_api_key.strip()) else "no",
        }

    def build_session_settings(self) -> SessionSettings | None:
        if self.care_chat_session_history_limit is None:
            return None

        return SessionSettings(limit=self.care_chat_session_history_limit)


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings()
