from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..config import Settings
from ..service import CareChatService
from .adapter import ChannelAdapter
from .models import InboundMessage, OutboundReply

logger = logging.getLogger(__name__)


@dataclass
class Dispatcher:
    """Routes messages between IM channel adapters and CareChatService instances."""

    settings: Settings
    _adapters: dict[str, ChannelAdapter] = field(default_factory=dict, init=False)
    _services: dict[str, CareChatService] = field(default_factory=dict, init=False)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict, init=False)

    def register_adapter(self, adapter: ChannelAdapter) -> None:
        self._adapters[adapter.channel_name] = adapter
        logger.info("Registered channel adapter: %s", adapter.channel_name)

    def get_adapter(self, channel: str) -> ChannelAdapter | None:
        return self._adapters.get(channel)

    @property
    def adapters(self) -> dict[str, ChannelAdapter]:
        return dict(self._adapters)

    def _get_or_create_service(self, session_key: str) -> CareChatService:
        if session_key not in self._services:
            self._services[session_key] = CareChatService(
                settings=self.settings,
                session_id=session_key,
            )
            logger.debug("Created new service for session: %s", session_key)
        return self._services[session_key]

    def _get_lock(self, session_key: str) -> asyncio.Lock:
        if session_key not in self._locks:
            self._locks[session_key] = asyncio.Lock()
        return self._locks[session_key]

    async def dispatch(self, message: InboundMessage) -> None:
        """Route an inbound message to the agent, then reply via the channel adapter."""
        adapter = self._adapters.get(message.channel)
        if adapter is None:
            logger.warning("No adapter for channel: %s", message.channel)
            return

        session_key = message.session_key
        lock = self._get_lock(session_key)

        async with lock:
            service = self._get_or_create_service(session_key)
            try:
                reply_text = await self._get_reply(service, message.text)
            except Exception:
                logger.exception("Agent reply failed for session %s", session_key)
                reply_text = "抱歉，系统暂时无法处理您的消息，请稍后再试。"

        reply = OutboundReply(
            text=reply_text,
            channel=message.channel,
            user_id=message.user_id,
            message_id=message.message_id,
            chat_id=message.chat_id,
            extra=message.extra,
        )

        try:
            await adapter.send_reply(reply)
        except Exception:
            logger.exception("Failed to send reply via %s for session %s", message.channel, session_key)

    async def _get_reply(self, service: CareChatService, text: str) -> str:
        chunks: list[str] = []
        async for chunk in service.stream_reply(text):
            if chunk.kind == "text":
                chunks.append(chunk.text)
        result = "".join(chunks).strip()
        return result or "抱歉，我暂时无法回复，请稍后再试。"

    async def aclose(self) -> None:
        for key, service in self._services.items():
            try:
                await service.aclose()
            except Exception:
                logger.debug("Error closing service %s", key, exc_info=True)
        self._services.clear()
        self._locks.clear()
        logger.info("Dispatcher shut down, all services closed.")
