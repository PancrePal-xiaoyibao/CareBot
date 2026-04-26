from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..config import load_settings, Settings
from .dispatcher import Dispatcher
from .channels.feishu import FeishuAdapter
from .channels.telegram import TelegramAdapter

logger = logging.getLogger(__name__)

_dispatcher: Dispatcher | None = None


def _build_dispatcher(settings: Settings) -> Dispatcher:
    dispatcher = Dispatcher(settings=settings)

    if settings.care_chat_gateway_feishu_enabled:
        adapter = FeishuAdapter(
            app_id=settings.care_chat_gateway_feishu_app_id,
            app_secret=settings.care_chat_gateway_feishu_app_secret,
            verification_token=settings.care_chat_gateway_feishu_verification_token,
            encrypt_key=settings.care_chat_gateway_feishu_encrypt_key,
            mode=settings.care_chat_gateway_feishu_mode,
        )
        adapter.on_message = dispatcher.dispatch
        dispatcher.register_adapter(adapter)

    if settings.care_chat_gateway_telegram_enabled:
        adapter = TelegramAdapter(
            bot_token=settings.care_chat_gateway_telegram_bot_token,
            webhook_secret=settings.care_chat_gateway_telegram_webhook_secret,
        )
        adapter.on_message = dispatcher.dispatch
        dispatcher.register_adapter(adapter)

    return dispatcher


@asynccontextmanager
async def _lifespan(app: Starlette) -> AsyncIterator[None]:
    global _dispatcher
    logger.info("Gateway starting up ...")

    if _dispatcher is not None:
        feishu = _dispatcher.get_adapter("feishu")
        if isinstance(feishu, FeishuAdapter) and feishu.mode == "websocket":
            loop = asyncio.get_running_loop()
            feishu.start_websocket(loop)

    yield
    if _dispatcher is not None:
        await _dispatcher.aclose()
        _dispatcher = None
    logger.info("Gateway shut down.")


async def _health(request: Request) -> JSONResponse:
    adapters = list(_dispatcher.adapters.keys()) if _dispatcher else []
    return JSONResponse({"status": "ok", "channels": adapters})


async def _webhook_feishu(request: Request) -> JSONResponse:
    if _dispatcher is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    adapter = _dispatcher.get_adapter("feishu")
    if adapter is None:
        return JSONResponse({"error": "feishu not enabled"}, status_code=404)
    return await adapter.handle_webhook(request)


async def _webhook_telegram(request: Request) -> JSONResponse:
    if _dispatcher is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    adapter = _dispatcher.get_adapter("telegram")
    if adapter is None:
        return JSONResponse({"error": "telegram not enabled"}, status_code=404)
    return await adapter.handle_webhook(request)


def create_app(settings: Settings | None = None) -> Starlette:
    global _dispatcher
    if settings is None:
        settings = load_settings()
    _dispatcher = _build_dispatcher(settings)

    return Starlette(
        lifespan=_lifespan,
        routes=[
            Route("/health", _health, methods=["GET"]),
            Route("/webhook/feishu", _webhook_feishu, methods=["POST"]),
            Route("/webhook/telegram", _webhook_telegram, methods=["POST"]),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Care Chat Gateway: webhook server bridging IM channels to the Care Chat agent.",
    )
    parser.add_argument("--host", default=None, help="Bind host (default: from env or 0.0.0.0).")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: from env or 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    )

    settings = load_settings()
    settings.ensure_ready()

    host = args.host or settings.care_chat_gateway_host
    port = args.port or settings.care_chat_gateway_port

    enabled = []
    if settings.care_chat_gateway_feishu_enabled:
        enabled.append("feishu")
    if settings.care_chat_gateway_telegram_enabled:
        enabled.append("telegram")

    if not enabled:
        logger.error(
            "No channel adapter enabled. Set CARE_CHAT_GATEWAY_FEISHU_ENABLED=true "
            "or CARE_CHAT_GATEWAY_TELEGRAM_ENABLED=true in .env"
        )
        raise SystemExit(1)

    logger.info("Starting gateway on %s:%d with channels: %s", host, port, ", ".join(enabled))

    import uvicorn

    uvicorn.run(
        "care_chat.gateway.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=args.reload,
    )
