from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from care_chat.gateway.models import InboundMessage, OutboundReply, split_text
from care_chat.gateway.channels.feishu import FeishuAdapter
from care_chat.gateway.channels.telegram import TelegramAdapter
from care_chat.gateway.dispatcher import Dispatcher


# ── split_text ──────────────────────────────────────────────────────


def test_split_text_short():
    assert split_text("hello") == ["hello"]


def test_split_text_exact_limit():
    text = "a" * 100
    assert split_text(text, max_length=100) == [text]


def test_split_text_splits_at_paragraph():
    text = "first paragraph.\n\nsecond paragraph."
    chunks = split_text(text, max_length=25)
    assert len(chunks) == 2
    assert chunks[0] == "first paragraph."
    assert chunks[1] == "second paragraph."


def test_split_text_splits_at_sentence():
    text = "句子一。句子二。句子三。"
    chunks = split_text(text, max_length=15)
    assert all(len(c) <= 15 for c in chunks)
    assert "".join(chunks) == text


def test_split_text_hard_split():
    text = "a" * 200
    chunks = split_text(text, max_length=100)
    assert len(chunks) == 2
    assert chunks[0] == "a" * 100
    assert chunks[1] == "a" * 100


# ── InboundMessage ──────────────────────────────────────────────────


def test_inbound_message_session_key():
    msg = InboundMessage(channel="feishu", user_id="ou_123", text="hi")
    assert msg.session_key == "feishu:ou_123"


# ── FeishuAdapter: SDK event parsing ────────────────────────────────


def _feishu_adapter() -> FeishuAdapter:
    return FeishuAdapter(app_id="test_id", app_secret="test_secret")


def _build_sdk_event(
    *,
    open_id: str = "ou_abc",
    sender_type: str = "user",
    message_id: str = "om_1",
    chat_id: str = "oc_1",
    chat_type: str = "p2p",
    message_type: str = "text",
    content: str | None = None,
):
    from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
    from lark_oapi.api.im.v1.model.p2_im_message_receive_v1 import P2ImMessageReceiveV1Data
    from lark_oapi.api.im.v1.model import EventSender, EventMessage
    from lark_oapi.api.im.v1.model.user_id import UserId

    if content is None:
        content = json.dumps({"text": "你好"})

    uid = UserId()
    uid.open_id = open_id

    sender = EventSender()
    sender.sender_id = uid
    sender.sender_type = sender_type

    msg = EventMessage()
    msg.message_id = message_id
    msg.chat_id = chat_id
    msg.chat_type = chat_type
    msg.message_type = message_type
    msg.content = content

    event_data = P2ImMessageReceiveV1Data()
    event_data.sender = sender
    event_data.message = msg

    event = P2ImMessageReceiveV1()
    event.event = event_data
    return event


def test_feishu_parse_text_message():
    adapter = _feishu_adapter()
    event = _build_sdk_event()
    msg = adapter._parse_sdk_event(event)
    assert msg is not None
    assert msg.channel == "feishu"
    assert msg.user_id == "ou_abc"
    assert msg.text == "你好"
    assert msg.message_id == "om_1"
    assert msg.chat_id == "oc_1"


def test_feishu_parse_strips_mentions():
    adapter = _feishu_adapter()
    event = _build_sdk_event(
        chat_type="group",
        content=json.dumps({"text": "@_user_1 帮帮我"}),
    )
    msg = adapter._parse_sdk_event(event)
    assert msg is not None
    assert msg.text == "帮帮我"


def test_feishu_parse_ignores_non_text():
    adapter = _feishu_adapter()
    event = _build_sdk_event(message_type="image", content="{}")
    assert adapter._parse_sdk_event(event) is None


def test_feishu_parse_ignores_empty_text():
    adapter = _feishu_adapter()
    event = _build_sdk_event(content=json.dumps({"text": ""}))
    assert adapter._parse_sdk_event(event) is None


def test_feishu_parse_ignores_none_event():
    adapter = _feishu_adapter()
    from lark_oapi.api.im.v1 import P2ImMessageReceiveV1

    event = P2ImMessageReceiveV1()
    event.event = None
    assert adapter._parse_sdk_event(event) is None


def test_feishu_mode_default_websocket():
    adapter = _feishu_adapter()
    assert adapter.mode == "websocket"


def test_feishu_mode_webhook():
    adapter = FeishuAdapter(app_id="x", app_secret="y", mode="webhook")
    assert adapter.mode == "webhook"


# ── TelegramAdapter parsing ────────────────────────────────────────


def _telegram_adapter() -> TelegramAdapter:
    return TelegramAdapter(bot_token="123:ABC")


def test_telegram_parse_message():
    adapter = _telegram_adapter()
    update = {
        "update_id": 1,
        "message": {
            "message_id": 42,
            "from": {"id": 100, "first_name": "Alice"},
            "chat": {"id": 100, "type": "private"},
            "text": "hello",
        },
    }
    msg = adapter._parse_update(update)
    assert msg is not None
    assert msg.channel == "telegram"
    assert msg.user_id == "100"
    assert msg.text == "hello"
    assert msg.chat_id == "100"


def test_telegram_parse_edited_message():
    adapter = _telegram_adapter()
    update = {
        "update_id": 2,
        "edited_message": {
            "message_id": 43,
            "from": {"id": 200, "first_name": "Bob"},
            "chat": {"id": 200, "type": "private"},
            "text": "edited text",
        },
    }
    msg = adapter._parse_update(update)
    assert msg is not None
    assert msg.text == "edited text"


def test_telegram_parse_no_text():
    adapter = _telegram_adapter()
    update = {"update_id": 3, "message": {"message_id": 44, "from": {"id": 1}, "chat": {"id": 1}}}
    assert adapter._parse_update(update) is None


def test_telegram_parse_no_message():
    adapter = _telegram_adapter()
    update = {"update_id": 4, "channel_post": {"message_id": 45, "text": "post"}}
    assert adapter._parse_update(update) is None


# ── Dispatcher ──────────────────────────────────────────────────────


@pytest.fixture
def settings():
    from care_chat.config import Settings

    return Settings(
        openai_api_key="test-key",
        care_chat_tracing_disabled=True,
    )


def test_dispatcher_register_adapter(settings):
    dispatcher = Dispatcher(settings=settings)
    adapter = _feishu_adapter()
    dispatcher.register_adapter(adapter)
    assert dispatcher.get_adapter("feishu") is adapter
    assert "feishu" in dispatcher.adapters


def test_dispatcher_get_adapter_returns_none(settings):
    dispatcher = Dispatcher(settings=settings)
    assert dispatcher.get_adapter("unknown") is None
