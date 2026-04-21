from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.text import Text

from .config import load_settings
from .schemas import CareRoleHint
from .service import CareChatService, run_sync_coro


async def _print_streaming_reply(
    service: CareChatService,
    message: str,
    *,
    console: Console,
    show_reasoning: bool,
    show_tools: bool,
    plain: bool,
    role_hint: CareRoleHint,
) -> None:
    last_kind: str | None = None
    async for chunk in service.stream_reply(
        message,
        include_reasoning=show_reasoning,
        include_tool_activity=show_tools,
        role_hint=role_hint,
    ):
        if chunk.kind == "tool":
            if plain:
                if last_kind is not None:
                    print()
                print(f"Tool> {chunk.text}")
            else:
                if last_kind is not None:
                    console.print()
                console.print(f"[bold yellow]Tool> [/bold yellow][yellow]{chunk.text}[/yellow]")
            last_kind = None
            continue

        if plain:
            if chunk.kind != last_kind:
                if last_kind is not None:
                    print()
                label = (
                    "Thinking"
                    if chunk.kind == "reasoning"
                    else "Care"
                )
                print(f"{label}> ", end="", flush=True)
                last_kind = chunk.kind
            print(chunk.text, end="", flush=True)
            continue

        if chunk.kind != last_kind:
            if last_kind is not None:
                console.print()
            label = (
                "Thinking"
                if chunk.kind == "reasoning"
                else "Care"
            )
            style = (
                "dim cyan"
                if chunk.kind == "reasoning"
                else "bold green"
            )
            console.print(f"[{style}]{label}> [/{style}]", end="")
            last_kind = chunk.kind

        style = (
            "dim cyan"
            if chunk.kind == "reasoning"
            else "white"
        )
        console.print(Text(chunk.text, style=style), end="")

    if plain:
        print()
    else:
        console.print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Care Chat: a cancer-support companion built with the OpenAI Agents SDK.",
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="Optional one-shot user message. If omitted, an interactive session starts.",
    )
    parser.add_argument(
        "--session-id",
        default="care-chat-local",
        help="Persistent conversation session id stored in SQLite.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override the SQLite session database path.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved runtime config summary.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the assistant reply when the provider supports it.",
    )
    parser.add_argument(
        "--clear-session",
        action="store_true",
        help="Clear the persisted SQLite session before running.",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Show model-supplied reasoning/thinking text when the provider returns it.",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Show handoff and tool-calling activity during streamed runs.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Disable Rich formatting and print plain terminal output.",
    )
    parser.add_argument(
        "--role-hint",
        choices=("auto", "patient", "caregiver", "volunteer"),
        default=None,
        help="Hint which audience this session is serving.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings()
    console = Console()
    resolved_role_hint: CareRoleHint = args.role_hint or settings.care_chat_default_role_hint
    tracing_warning = settings.tracing_warning()

    if args.show_reasoning:
        args.stream = True
    if args.show_tools:
        args.stream = True

    if args.show_config:
        if args.plain:
            print(json.dumps(settings.safe_summary(), ensure_ascii=False, indent=2))
            if tracing_warning:
                print(f"\n[tracing-warning] {tracing_warning}", file=sys.stderr)
        else:
            console.print(JSON.from_data(settings.safe_summary()))
            if tracing_warning:
                console.print(Panel.fit(tracing_warning, title="Tracing Warning", border_style="yellow"))
        if not args.message:
            return

    try:
        service = CareChatService(
            settings=settings,
            session_id=args.session_id,
            db_path=args.db_path,
        )
    except Exception as exc:
        if args.plain:
            print(f"配置或初始化失败: {exc}", file=sys.stderr)
        else:
            console.print(Panel.fit(str(exc), title="配置或初始化失败", border_style="red"))
        raise SystemExit(1) from exc

    try:
        if args.clear_session:
            run_sync_coro(service.clear_session())
            if not args.plain:
                console.print("[yellow]Session history cleared.[/yellow]")

        if args.message:
            try:
                message = " ".join(args.message)
                if args.stream:
                    run_sync_coro(
                        _print_streaming_reply(
                            service,
                            message,
                            console=console,
                            show_reasoning=args.show_reasoning,
                            show_tools=args.show_tools,
                            plain=args.plain,
                            role_hint=resolved_role_hint,
                        )
                    )
                else:
                    reply = service.reply(message, role_hint=resolved_role_hint)
                    if args.plain:
                        print(reply)
                    else:
                        console.print(Panel(reply, title="Care", border_style="green"))
            except Exception as exc:
                if args.plain:
                    print(f"运行失败: {exc}", file=sys.stderr)
                else:
                    console.print(Panel.fit(str(exc), title="运行失败", border_style="red"))
                raise SystemExit(1) from exc
            return

        if args.plain:
            print(
                f"Care Chat ready. session={args.session_id} model={settings.care_chat_model} stream={str(args.stream).lower()} role={resolved_role_hint}\n"
                "输入 exit、quit 或 Ctrl+C 结束。"
            )
            if tracing_warning:
                print(f"[tracing-warning] {tracing_warning}")
        else:
            features = (
                f"stream={str(args.stream).lower()} "
                f"thinking={str(settings.care_chat_enable_thinking).lower()} "
                f"show_reasoning={str(args.show_reasoning).lower()} "
                f"show_tools={str(args.show_tools).lower()} "
                f"role={resolved_role_hint}"
            )
            console.print(
                Panel.fit(
                    f"[bold]Care Chat[/bold]\nsession=[cyan]{args.session_id}[/cyan]\n"
                    f"model=[magenta]{settings.care_chat_model}[/magenta]\n{features}\n"
                    "输入 exit、quit 或 Ctrl+C 结束。",
                    border_style="blue",
                )
            )
            if tracing_warning:
                console.print(
                    Panel.fit(tracing_warning, title="Tracing Warning", border_style="yellow")
                )

        while True:
            try:
                if args.plain:
                    user_input = input("You> ").strip()
                else:
                    user_input = console.input("[bold cyan]You> [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                if args.plain:
                    print()
                else:
                    console.print()
                return

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", ":q"}:
                return

            try:
                if args.stream:
                    run_sync_coro(
                        _print_streaming_reply(
                            service,
                            user_input,
                            console=console,
                            show_reasoning=args.show_reasoning,
                            show_tools=args.show_tools,
                            plain=args.plain,
                            role_hint=resolved_role_hint,
                        )
                    )
                    if not args.plain:
                        console.print()
                    continue

                reply = service.reply(user_input, role_hint=resolved_role_hint)
            except Exception as exc:
                if args.plain:
                    print(f"Care> 运行失败: {exc}\n")
                else:
                    console.print(Panel.fit(str(exc), title="运行失败", border_style="red"))
                    console.print()
                continue

            if args.plain:
                print(f"Care> {reply}\n")
            else:
                console.print(Panel(reply, title="Care", border_style="green"))
                console.print()
    finally:
        try:
            run_sync_coro(service.aclose())
        except Exception:
            pass
