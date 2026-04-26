from __future__ import annotations

from abc import ABC, abstractmethod

from starlette.requests import Request
from starlette.responses import Response

from .models import InboundMessage, OutboundReply


class ChannelAdapter(ABC):
    """Abstract base for IM channel adapters.

    Each adapter handles one messaging platform by implementing:
    - webhook parsing (platform -> InboundMessage)
    - reply sending (OutboundReply -> platform API)
    """

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Unique identifier, e.g. ``'feishu'``, ``'telegram'``."""
        ...

    @property
    def webhook_path(self) -> str:
        return f"/webhook/{self.channel_name}"

    @abstractmethod
    async def handle_webhook(self, request: Request) -> Response:
        """Handle the raw HTTP request from the platform.

        Must return a response quickly (< 3 s for most platforms).
        Long-running work should be dispatched via the *on_message* callback
        provided at init time.
        """
        ...

    @abstractmethod
    async def send_reply(self, reply: OutboundReply) -> None:
        """Deliver a reply back to the platform."""
        ...
