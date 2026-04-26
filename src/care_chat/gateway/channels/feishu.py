from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Literal

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    P2ImMessageReceiveV1,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..adapter import ChannelAdapter
from ..models import InboundMessage, OutboundReply
from .feishu_card import markdown_to_card_json, split_card_content

logger = logging.getLogger(__name__)

FeishuMode = Literal["websocket", "webhook"]


@dataclass
class FeishuAdapter(ChannelAdapter):
    app_id: str
    app_secret: str
    verification_token: str = ""
    encrypt_key: str = ""
    mode: FeishuMode = "websocket"
    on_message: Callable[[InboundMessage], Coroutine] | None = None

    _lark_client: lark.Client = field(init=False, repr=False)
    _event_handler: lark.EventDispatcherHandler = field(init=False, repr=False)
    _ws_client: lark.ws.Client | None = field(init=False, default=None, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._lark_client = (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        self._event_handler = (
            lark.EventDispatcherHandler.builder(
                self.encrypt_key,
                self.verification_token,
            )
            .register_p2_im_message_receive_v1(self._on_sdk_event)
            .build()
        )

    @property
    def channel_name(self) -> str:
        return "feishu"

    # ── SDK event callback (called from sync context) ─────────────────

    def _on_sdk_event(self, data: P2ImMessageReceiveV1) -> None:
        message = self._parse_sdk_event(data)
        if message is None or self.on_message is None:
            return

        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.on_message(message), self._loop)
        else:
            logger.warning("No running event loop; dropping Feishu message from %s", message.user_id)

    def _parse_sdk_event(self, data: P2ImMessageReceiveV1) -> InboundMessage | None:
        event = data.event
        if event is None or event.message is None:
            return None

        msg = event.message
        if msg.message_type != "text":
            logger.debug("Skipping non-text Feishu message type: %s", msg.message_type)
            return None

        try:
            content = json.loads(msg.content or "{}")
        except json.JSONDecodeError:
            return None

        text = content.get("text", "").strip()
        if not text:
            return None

        text = re.sub(r"@_user_\d+\s*", "", text).strip()
        if not text:
            return None

        sender = event.sender
        open_id = ""
        sender_type = ""
        if sender is not None:
            sender_type = sender.sender_type or ""
            if sender.sender_id is not None:
                open_id = sender.sender_id.open_id or ""

        return InboundMessage(
            channel="feishu",
            user_id=open_id,
            text=text,
            message_id=msg.message_id or "",
            chat_id=msg.chat_id,
            extra={
                "chat_type": msg.chat_type or "",
                "sender_type": sender_type,
            },
        )

    # ── WebSocket mode ────────────────────────────────────────────────

    def start_websocket(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._ws_client = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=self._event_handler,
            log_level=lark.LogLevel.INFO,
        )
        thread = threading.Thread(
            target=self._ws_client.start,
            name="feishu-ws",
            daemon=True,
        )
        thread.start()
        logger.info("Feishu WebSocket client started in background thread")

    # ── Webhook mode ──────────────────────────────────────────────────

    async def handle_webhook(self, request: Request) -> Response:
        try:
            body = await request.body()
        except Exception:
            return JSONResponse({"error": "read error"}, status_code=400)

        raw_req = lark.RawRequest()
        raw_req.uri = str(request.url.path)
        raw_req.headers = dict(request.headers)
        raw_req.body = body

        raw_resp = self._event_handler.do(raw_req)

        return Response(
            content=raw_resp.content,
            status_code=raw_resp.status_code,
            headers=raw_resp.headers,
        )

    # ── Send reply via SDK ────────────────────────────────────────────

    async def send_reply(self, reply: OutboundReply) -> None:
        chunks = split_text(reply.text, max_length=30000)
        for chunk in chunks:
            content = json.dumps({"text": chunk}, ensure_ascii=False)

            if reply.message_id:
                req = (
                    ReplyMessageRequest.builder()
                    .message_id(reply.message_id)
                    .request_body(
                        ReplyMessageRequestBody.builder()
                        .content(content)
                        .msg_type("text")
                        .build()
                    )
                    .build()
                )
                resp = await asyncio.to_thread(
                    self._lark_client.im.v1.message.reply, req
                )
            elif reply.chat_id:
                req = (
                    CreateMessageRequest.builder()
                    .receive_id_type("chat_id")
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(reply.chat_id)
                        .content(content)
                        .msg_type("text")
                        .build()
                    )
                    .build()
                )
                resp = await asyncio.to_thread(
                    self._lark_client.im.v1.message.create, req
                )
            else:
                req = (
                    CreateMessageRequest.builder()
                    .receive_id_type("open_id")
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(reply.user_id)
                        .content(content)
                        .msg_type("text")
                        .build()
                    )
                    .build()
                )
                resp = await asyncio.to_thread(
                    self._lark_client.im.v1.message.create, req
                )

            if resp and hasattr(resp, "code") and resp.code != 0:
                logger.error(
                    "Feishu send_reply failed: code=%s msg=%s",
                    getattr(resp, "code", "?"),
                    getattr(resp, "msg", "?"),
                )
