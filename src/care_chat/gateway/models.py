from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class InboundMessage:
    """Unified inbound message from any channel."""

    channel: str
    user_id: str
    text: str
    message_id: str = ""
    chat_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict = field(default_factory=dict)

    @property
    def session_key(self) -> str:
        return f"{self.channel}:{self.user_id}"


@dataclass(slots=True)
class OutboundReply:
    """Unified outbound reply to send back to a channel."""

    text: str
    channel: str
    user_id: str
    message_id: str = ""
    chat_id: str | None = None
    extra: dict = field(default_factory=dict)


def split_text(text: str, max_length: int = 4096) -> list[str]:
    """Split long text into chunks, preferring paragraph/sentence boundaries."""
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        split_pos = remaining.rfind("\n\n", 0, max_length)
        if split_pos == -1:
            split_pos = remaining.rfind("\n", 0, max_length)
        if split_pos == -1:
            for sep in ("\u3002", ".", "\uff01", "!", "\uff1f", "?"):
                split_pos = remaining.rfind(sep, 0, max_length)
                if split_pos != -1:
                    split_pos += len(sep)
                    break
        if split_pos <= 0:
            split_pos = max_length

        chunks.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip()

    return chunks
