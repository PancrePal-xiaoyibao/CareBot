from __future__ import annotations

import asyncio
import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)

_BEIJING_TZ = timezone(timedelta(hours=8))

_recent_alerts: dict[str, datetime] = {}


@dataclass(frozen=True, slots=True)
class CrisisAlertPayload:
    session_id: str
    user_message: str
    risk_level: str
    risk_reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")
    )


def _within_cooldown(session_id: str, cooldown_minutes: int) -> bool:
    last = _recent_alerts.get(session_id)
    if last is None:
        return False
    return (datetime.now(_BEIJING_TZ) - last) < timedelta(minutes=cooldown_minutes)


def _send_email_sync(payload: CrisisAlertPayload, settings: Settings) -> None:
    if not settings.care_chat_crisis_alert_enabled:
        return

    if _within_cooldown(payload.session_id, settings.care_chat_crisis_alert_cooldown_minutes):
        logger.info("Crisis alert for session %s skipped (cooldown).", payload.session_id)
        return

    recipients = [
        a.strip()
        for a in settings.care_chat_crisis_alert_recipient.split(",")
        if a.strip()
    ]
    if not recipients:
        logger.warning("Crisis alert enabled but no recipient configured.")
        return

    subject = f"[小馨宝危机预警] 会话 {payload.session_id} - 风险等级: {payload.risk_level}"
    body = (
        f"危机预警通知\n{'=' * 40}\n\n"
        f"会话 ID: {payload.session_id}\n"
        f"检测时间: {payload.timestamp}\n"
        f"风险等级: {payload.risk_level}\n"
        f"风险原因: {payload.risk_reason}\n\n"
        f"用户消息:\n{payload.user_message}\n\n"
        f"{'=' * 40}\n"
        f"请尽快安排人工干预。\n"
        f"此邮件由 Care Chat（小馨宝）危机监测系统自动发送。"
    )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = settings.care_chat_crisis_alert_smtp_user
    msg["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP_SSL(
            settings.care_chat_crisis_alert_smtp_host,
            settings.care_chat_crisis_alert_smtp_port,
            timeout=10,
        ) as server:
            server.login(
                settings.care_chat_crisis_alert_smtp_user,
                settings.care_chat_crisis_alert_smtp_password,
            )
            server.sendmail(
                settings.care_chat_crisis_alert_smtp_user,
                recipients,
                msg.as_string(),
            )
        _recent_alerts[payload.session_id] = datetime.now(_BEIJING_TZ)
        logger.info("Crisis alert email sent for session %s", payload.session_id)
    except Exception:
        logger.exception(
            "Failed to send crisis alert email for session %s", payload.session_id
        )


async def send_crisis_alert(payload: CrisisAlertPayload, settings: Settings) -> None:
    await asyncio.to_thread(_send_email_sync, payload, settings)
