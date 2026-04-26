from __future__ import annotations

import re

_CARD_CONTENT_LIMIT = 28000
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TEMPLATE_COLORS = (
    "blue", "wathet", "turquoise", "green", "yellow",
    "orange", "red", "carmine", "violet", "purple", "indigo", "grey",
)


def markdown_to_card(
    text: str,
    *,
    header_template: str = "blue",
    default_title: str = "",
) -> dict:
    """Convert LLM Markdown output to a Feishu Card JSON 2.0 structure.

    The first heading (if any) is extracted as the card header title.
    The remaining text is placed into one or more ``markdown`` body elements.
    """
    title, body = _extract_title(text)
    if not title:
        title = default_title

    card: dict = {"schema": "2.0"}

    card["config"] = {"width_mode": "fill"}

    if title:
        card["header"] = {
            "title": {"tag": "plain_text", "content": title},
            "template": header_template if header_template in _TEMPLATE_COLORS else "blue",
        }

    elements = _build_body_elements(body)
    card["body"] = {"elements": elements}

    return card


def markdown_to_card_json(text: str, **kwargs: object) -> str:
    """Return the card dict serialised as a JSON string (for the Feishu API)."""
    import json

    return json.dumps(markdown_to_card(text, **kwargs), ensure_ascii=False)


def split_card_content(text: str, max_length: int = _CARD_CONTENT_LIMIT) -> list[str]:
    """Split long markdown into chunks that each fit a single card."""
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
        if split_pos <= 0:
            split_pos = max_length

        chunks.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip()

    return chunks


def _extract_title(text: str) -> tuple[str, str]:
    """Pull the first heading out as a title; return (title, rest)."""
    m = _HEADING_RE.search(text)
    if m is None:
        return ("", text.strip())

    title = m.group(2).strip()
    body = (text[: m.start()] + text[m.end() :]).strip()
    return (title, body)


def _build_body_elements(md: str) -> list[dict]:
    if not md:
        return [{"tag": "markdown", "content": " "}]

    sections = _split_by_headings(md)

    elements: list[dict] = []
    for section in sections:
        heading, content = section
        if heading:
            elements.append({
                "tag": "markdown",
                "content": f"**{heading}**",
            })
            if content:
                elements.append({"tag": "markdown", "content": content})
        else:
            elements.append({"tag": "markdown", "content": content})

    if len(elements) > 190:
        merged = "\n\n".join(e["content"] for e in elements)
        elements = [{"tag": "markdown", "content": merged}]

    return elements


def _split_by_headings(md: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, content) sections.

    Sections without a heading get heading=''.
    """
    parts = _HEADING_RE.split(md)

    sections: list[tuple[str, str]] = []

    idx = 0
    if parts[0].strip():
        sections.append(("", parts[0].strip()))
    idx = 1

    while idx + 2 < len(parts):
        heading_text = parts[idx + 1].strip()
        content = parts[idx + 2].strip() if idx + 2 < len(parts) else ""
        sections.append((heading_text, content))
        idx += 3

    if not sections:
        sections.append(("", md.strip()))

    return sections
