from pathlib import Path

import pytest

from care_chat.config import Settings


def test_guardrail_model_defaults_to_main_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_model="gpt-5.4-mini",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
    )

    assert settings.guardrail_model == "gpt-5.4-mini"
    assert settings.role_router_model == "gpt-5.4-mini"
    assert settings.safe_summary()["api_key_configured"] == "yes"


def test_role_router_model_can_override_main_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_model="moonshot/kimi-k2.5",
        care_chat_role_router_model="gpt-4.1-mini",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
    )

    assert settings.role_router_model == "gpt-4.1-mini"
    assert settings.safe_summary()["role_router_model"] == "gpt-4.1-mini"


def test_build_session_settings_uses_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
        care_chat_session_history_limit=25,
    )

    session_settings = settings.build_session_settings()
    assert session_settings is not None
    assert session_settings.limit == 25


def test_empty_session_limit_is_treated_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
        care_chat_session_history_limit="",
    )

    assert settings.care_chat_session_history_limit is None
    assert settings.build_session_settings() is None


def test_ensure_ready_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        openai_api_key="",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.ensure_ready()


def test_safe_summary_includes_lightweight_runtime_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
    )

    summary = settings.safe_summary()

    assert summary["role_router_model"] == summary["model"]
    assert summary["thinking"] == "true"
    assert summary["default_role_hint"] == "auto"
    assert summary["llm_input_guardrail"] == "false"
    assert summary["llm_output_guardrail"] == "false"
    assert summary["tracing_disabled"] == "false"
    assert summary["tracing_export_api_key_configured"] == "yes"
    assert summary["trace_include_sensitive_data"] == "false"


def test_custom_gateway_without_tracing_key_disables_official_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,
        openai_api_key="gateway-key",
        openai_base_url="https://gateway.example.com/v1",
        care_chat_session_db_path=Path("tmp/test.sqlite3"),
    )

    assert settings.uses_custom_openai_base_url is True
    assert settings.tracing_export_api_key is None
    assert settings.tracing_effective_disabled is True
    assert settings.build_tracing_config() is None
    assert settings.tracing_warning() is not None
