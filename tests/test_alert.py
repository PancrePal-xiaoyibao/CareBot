from datetime import datetime, timedelta
from unittest.mock import patch

from care_chat.alert import (
    CrisisAlertPayload,
    _BEIJING_TZ,
    _recent_alerts,
    _send_email_sync,
    _within_cooldown,
)
from care_chat.config import Settings
from pathlib import Path


def _test_settings(**overrides) -> Settings:
    defaults = {
        "_env_file": None,
        "openai_api_key": "test-key",
        "care_chat_session_db_path": Path("tmp/test.sqlite3"),
        "care_chat_crisis_alert_enabled": True,
        "care_chat_crisis_alert_smtp_host": "smtp.example.com",
        "care_chat_crisis_alert_smtp_port": 465,
        "care_chat_crisis_alert_smtp_user": "alert@example.com",
        "care_chat_crisis_alert_smtp_password": "secret",
        "care_chat_crisis_alert_recipient": "admin@example.com",
        "care_chat_crisis_alert_cooldown_minutes": 10,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_crisis_alert_payload_has_timestamp() -> None:
    payload = CrisisAlertPayload(
        session_id="test-session",
        user_message="我不想活了",
        risk_level="critical",
        risk_reason="Self-harm keywords detected",
    )

    assert payload.session_id == "test-session"
    assert payload.risk_level == "critical"
    assert len(payload.timestamp) > 0


def test_within_cooldown_returns_false_for_new_session() -> None:
    _recent_alerts.clear()
    assert _within_cooldown("brand-new-session", 10) is False


def test_within_cooldown_returns_true_within_window() -> None:
    _recent_alerts.clear()
    _recent_alerts["s1"] = datetime.now(_BEIJING_TZ)
    assert _within_cooldown("s1", 10) is True


def test_within_cooldown_returns_false_after_window() -> None:
    _recent_alerts.clear()
    _recent_alerts["s2"] = datetime.now(_BEIJING_TZ) - timedelta(minutes=15)
    assert _within_cooldown("s2", 10) is False


def test_send_email_skips_when_disabled() -> None:
    _recent_alerts.clear()
    settings = _test_settings(care_chat_crisis_alert_enabled=False)
    payload = CrisisAlertPayload(
        session_id="s",
        user_message="test",
        risk_level="high",
        risk_reason="test",
    )

    with patch("care_chat.alert.smtplib") as mock_smtp:
        _send_email_sync(payload, settings)
        mock_smtp.SMTP_SSL.assert_not_called()


def test_send_email_skips_when_in_cooldown() -> None:
    _recent_alerts.clear()
    _recent_alerts["s"] = datetime.now(_BEIJING_TZ)
    settings = _test_settings()
    payload = CrisisAlertPayload(
        session_id="s",
        user_message="test",
        risk_level="high",
        risk_reason="test",
    )

    with patch("care_chat.alert.smtplib") as mock_smtp:
        _send_email_sync(payload, settings)
        mock_smtp.SMTP_SSL.assert_not_called()


def test_send_email_skips_when_no_recipient() -> None:
    _recent_alerts.clear()
    settings = _test_settings(care_chat_crisis_alert_recipient="")
    payload = CrisisAlertPayload(
        session_id="s",
        user_message="test",
        risk_level="high",
        risk_reason="test",
    )

    with patch("care_chat.alert.smtplib") as mock_smtp:
        _send_email_sync(payload, settings)
        mock_smtp.SMTP_SSL.assert_not_called()
