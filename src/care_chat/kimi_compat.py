from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agents.models.chatcmpl_converter import Converter
from agents.models.reasoning_content_replay import (
    ReasoningContentReplayContext,
    default_should_replay_reasoning_content,
)
from openai.types.responses import ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

_PATCHED = False


def _is_kimi_model_name(model_name: str | None) -> bool:
    return bool(model_name and "kimi-k2.5" in model_name.lower())


def should_replay_kimi_reasoning_content(context: ReasoningContentReplayContext) -> bool:
    if _is_kimi_model_name(context.model):
        return True

    return default_should_replay_reasoning_content(context)


def _extract_reasoning_text_from_message(message: Any) -> str | None:
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        return reasoning_content

    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        return reasoning

    reasoning_details = getattr(message, "reasoning_details", None) or []
    texts = []
    for detail in reasoning_details:
        if isinstance(detail, Mapping) and detail.get("text"):
            texts.append(str(detail["text"]))
    if texts:
        return "\n".join(texts)

    return None


def _has_reasoning_item(items: list[Any]) -> bool:
    for item in items:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            return True
        if isinstance(item, Mapping) and item.get("type") == "reasoning":
            return True
    return False


def _normalize_reasoning_item_for_replay(item: Any) -> Any:
    payload: Any = item
    if hasattr(item, "model_dump"):
        payload = item.model_dump()

    if not isinstance(payload, dict) or payload.get("type") != "reasoning":
        return item

    if payload.get("summary"):
        return payload

    content_items = payload.get("content") or []
    reasoning_texts = []
    for content_item in content_items:
        if (
            isinstance(content_item, Mapping)
            and content_item.get("type") == "reasoning_text"
            and content_item.get("text")
        ):
            reasoning_texts.append(str(content_item["text"]))

    if not reasoning_texts:
        return payload

    normalized = dict(payload)
    normalized["summary"] = [
        {
            "text": "\n".join(reasoning_texts),
            "type": "summary_text",
        }
    ]
    return normalized


def _ensure_kimi_reasoning_content_on_tool_calls(messages: list[Any]) -> None:
    """Moonshot requires ``reasoning_content`` on every assistant message that carries
    ``tool_calls`` when thinking mode is active.  Gateways may force thinking on for
    kimi-k2.5 regardless of the ``extra_body`` we send, so patch any message that is
    still missing the field after the normal replay logic runs."""

    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and not msg.get("reasoning_content")
        ):
            msg["reasoning_content"] = ""


def install_kimi_chat_compatibility() -> None:
    global _PATCHED
    if _PATCHED:
        return

    original_message_to_output_items = Converter.message_to_output_items.__func__
    original_items_to_messages = Converter.items_to_messages.__func__

    def patched_message_to_output_items(
        cls,
        message: Any,
        provider_data: dict[str, Any] | None = None,
    ) -> list[Any]:
        items = original_message_to_output_items(cls, message, provider_data)
        if _has_reasoning_item(items):
            return items

        reasoning_text = _extract_reasoning_text_from_message(message)
        if not reasoning_text:
            return items

        reasoning_kwargs: dict[str, Any] = {
            "id": "fake_resp_reasoning",
            "summary": [Summary(text=reasoning_text, type="summary_text")],
            "type": "reasoning",
        }
        if provider_data:
            reasoning_kwargs["provider_data"] = provider_data

        return [ResponseReasoningItem(**reasoning_kwargs), *items]

    def patched_items_to_messages(
        cls,
        items: str | Iterable[Any],
        model: str | None = None,
        preserve_thinking_blocks: bool = False,
        preserve_tool_output_all_content: bool = False,
        base_url: str | None = None,
        should_replay_reasoning_content=None,
    ) -> list[Any]:
        if isinstance(items, str):
            return original_items_to_messages(
                cls,
                items,
                model=model,
                preserve_thinking_blocks=preserve_thinking_blocks,
                preserve_tool_output_all_content=preserve_tool_output_all_content,
                base_url=base_url,
                should_replay_reasoning_content=should_replay_reasoning_content,
            )

        normalized_items = [_normalize_reasoning_item_for_replay(item) for item in items]
        replay_callback = should_replay_reasoning_content
        if replay_callback is None and _is_kimi_model_name(model):
            replay_callback = should_replay_kimi_reasoning_content

        messages = original_items_to_messages(
            cls,
            normalized_items,
            model=model,
            preserve_thinking_blocks=preserve_thinking_blocks,
            preserve_tool_output_all_content=preserve_tool_output_all_content,
            base_url=base_url,
            should_replay_reasoning_content=replay_callback,
        )

        if _is_kimi_model_name(model):
            _ensure_kimi_reasoning_content_on_tool_calls(messages)

        return messages

    Converter.message_to_output_items = classmethod(patched_message_to_output_items)
    Converter.items_to_messages = classmethod(patched_items_to_messages)
    _PATCHED = True
