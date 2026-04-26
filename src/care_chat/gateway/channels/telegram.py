from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..adapter import ChannelAdapter
from ..models import InboundMessage, OutboundReply, split_text

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


@dataclass
class TelegramAdapter(ChannelAdapter):
    """Telegram Bot API adapter.

    Receives updates via webhook and sends replies via the Bot API.
    """

    bot_token: str
    webhook_secret: str = ""
    on_message: Callable[[InboundMessage], Coroutine] | None = None
    _http: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=15)

    @property
    def channel_name(self) -> str:
        return "telegram"

    @property
    def _api_base(self) -> str:
        return f"{TELEGRAM_API_BASE}/bot{self.bot_token}"

    async def handle_webhook(self, request: Request) -> Response:
        if self.webhook_secret:
            header_secret = request.headers.get("x-telegram-bot-api-secret-token", "")
            if header_secret != self.webhook_secret:
                return JSONResponse({"error": "unauthorized"}, status_code=403)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)

        message = self._parse_update(body)
        if message is not None and self.on_message is not None:
            asyncio.create_task(self.on_message(message))

        return JSONResponse({"ok": True})

    def _parse_update(self, body: dict) -> InboundMessage | None:
        msg = body.get("message") or body.get("edited_message")
        if not msg:
            return None

        text = msg.get("text", "").strip()
        if not text:
            return None

        from_user = msg.get("from", {})
        chat = msg.get("chat", {})

        return InboundMessage(
            channel="telegram",
            user_id=str(from_user.get("id", "")),
            text=text,
            message_id=str(msg.get("message_id", "")),
            chat_id=str(chat.get("id", "")),
            extra={
                "chat_type": chat.get("type", ""),
                "from_username": from_user.get("username", ""),
                "from_first_name": from_user.get("first_name", ""),
            },
        )

    async def send_reply(self, reply: OutboundReply) -> None:
        chunks = split_text(reply.text, max_length=4096)
        for chunk in chunks:
            payload: dict = {
                "chat_id": reply.chat_id or reply.user_id,
                "text": chunk,
            }
            if reply.message_id:
                payload["reply_to_message_id"] = int(reply.message_id)

            await self._http.post(f"{self._api_base}/sendMessage", json=payload)

    async def set_webhook(self, url: str) -> dict:
        """Register webhook URL with Telegram. Call once during deployment."""
        payload: dict = {"url": url}
        if self.webhook_secret:
            payload["secret_token"] = self.webhook_secret
        resp = await self._http.post(f"{self._api_base}/setWebhook", json=payload)
        return resp.json()

    async def delete_webhook(self) -> dict:
        resp = await self._http.post(f"{self._api_base}/deleteWebhook")
        return resp.json()

    async def aclose(self) -> None:
        await self._http.aclose()
