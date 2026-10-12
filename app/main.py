import asyncio
import logging
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from .config import Settings
from .chat import ChatEngine, ChatRequest
from .errors import ServiceError
from .inference import InferenceEngine
from .registry import MODELS

logger = logging.getLogger("lightbridge.asr")


async def decode_audio(*args, **kwargs):
    # Remote chat does not need NumPy, FFmpeg or any model libraries.
    from .audio import decode_audio as decode
    return await decode(*args, **kwargs)


def error_response(error, request_id):
    return JSONResponse(
        {"error": {"code": error.code, "message": error.message}, "request_id": request_id},
        status_code=error.status,
        headers={"Cache-Control": "no-store", "X-Request-ID": request_id},
    )


class RequestGuard:
    """Authenticate and bound multipart bytes BEFORE Starlette parses/uploads them."""

    def __init__(self, app, settings):
        self.app = app
        self.settings = settings
        self.busy = False

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        is_chat = scope["path"].rstrip("/") == "/chat"
        guarded = is_chat or scope["path"].rstrip("/") == "/transcribe"
        if not guarded:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        supplied = headers.get(b"authorization", b"")
        expected = ("Bearer " + self.settings.ai_service_token.get_secret_value()).encode()
        if not secrets.compare_digest(supplied, expected):
            return await error_response(ServiceError(401, "unauthorized", "Service authentication required."), request_id)(scope, receive, send)
        if self.busy:
            response = error_response(ServiceError(429, "service_busy", "Another AI request is processing; retry shortly."), request_id)
            response.headers["Retry-After"] = "5"
            return await response(scope, receive, send)
        self.busy = True  # No await between check and acquisition.
        started = time.monotonic()
        try:
            content_type = headers.get(b"content-type", b"").lower()
            if is_chat and content_type.split(b";")[0].strip() != b"application/json":
                raise ServiceError(415, "json_required", "Send a JSON chat request.")
            if not is_chat and b"multipart/form-data" not in content_type:
                raise ServiceError(415, "multipart_required", "Send a multipart audio file and language.")
            limit = 128 * 1024 if is_chat else self.settings.max_upload_bytes + 64 * 1024
            declared = headers.get(b"content-length")
            if declared:
                try:
                    length = int(declared)
                except ValueError:
                    raise ServiceError(400, "invalid_request", "Invalid Content-Length.") from None
                if length < 0:
                    raise ServiceError(400, "invalid_request", "Invalid Content-Length.")
                if length > limit:
                    raise ServiceError(413, "request_too_large" if is_chat else "audio_too_large", "The request exceeds the size limit.")
            body = bytearray()
            async with asyncio.timeout(self.settings.request_timeout_seconds if is_chat else self.settings.upload_timeout_seconds):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > limit:
                        raise ServiceError(413, "request_too_large" if is_chat else "audio_too_large", "The request exceeds the size limit.")
                    if not message.get("more_body", False):
                        break

            consumed = False

            async def buffered_receive():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, buffered_receive, send)
        except TimeoutError:
            await error_response(ServiceError(408, "upload_timeout", "The upload timed out."), request_id)(scope, receive, send)
        except ServiceError as exc:
            await error_response(exc, request_id)(scope, receive, send)
        except Exception:
            # Catch before ServerErrorMiddleware/Uvicorn can log a raw exception
            # containing multipart details, local paths or third-party secrets.
            logger.error("request_id=%s code=internal_error", request_id)
            await error_response(ServiceError(500, "internal_error", "The request could not be processed."), request_id)(scope, receive, send)
        finally:
            self.busy = False
            logger.info("request_id=%s processing_ms=%d", request_id, (time.monotonic() - started) * 1000)


def create_app(settings=None, engine=None, decoder=decode_audio, chat_engine=None, render_only=False):
    settings = settings or Settings()
    remote_mode = settings.natlas_backend == "remote"
    engine = None if render_only or remote_mode else (engine or InferenceEngine(settings))
    if chat_engine is None:
        if remote_mode:
            from .remote import RemoteChatEngine
            chat_engine = RemoteChatEngine(settings)
        else:
            chat_engine = ChatEngine(settings)

    @asynccontextmanager
    async def lifespan(app):
        # Bind PORT immediately while downloading/loading; readiness stays 503.
        # No success is reported until the model has actually loaded.
        if settings.natlas_backend == "gguf":
            startup = asyncio.create_task(chat_engine.start())
        else:
            await chat_engine.start()
            startup = None
        try:
            yield
        finally:
            if startup and not startup.done():
                startup.cancel()
                with suppress(asyncio.CancelledError):
                    await startup
            await chat_engine.close()
            if engine:
                await engine.close()

    app = FastAPI(title="LightBridge AI — NCAIR speech and chat", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestGuard, settings=settings)

    if render_only or remote_mode:
        @app.get("/live")
        async def live():
            # Process liveness only; this is NOT inference readiness.
            return JSONResponse({"status": "alive", "mode": "remote" if remote_mode else settings.natlas_setup_mode,
                                 "model_state": chat_engine.state}, headers={"Cache-Control": "no-store"})

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        logger.warning("request_id=%s code=%s", request.state.request_id, exc.code)
        return error_response(exc, request.state.request_id)

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(HTTPException)
    async def invalid_request(request, exc):
        return error_response(ServiceError(getattr(exc, "status_code", 422), "invalid_request", "Invalid request fields or multipart data."), request.state.request_id)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.error("request_id=%s code=internal_error", request.state.request_id)
        return error_response(ServiceError(500, "internal_error", "The request could not be processed."), request.state.request_id)

    @app.get("/health")
    async def health():
        if settings.natlas_backend in ("gguf", "remote"):
            return await chat_health()
        available = bool(shutil.which(settings.ffmpeg_binary))
        healthy = available and engine.last_error is None
        return JSONResponse({
            "status": "ok" if healthy else "degraded",
            "decoder_available": available,
            "model_state": "loaded" if engine.loaded else "not_loaded",
            "enabled_languages": list(settings.languages),
            "loaded_languages": sorted(engine.loaded),
            "last_error": engine.last_error,
        }, status_code=200 if healthy else 503, headers={"Cache-Control": "no-store"})

    @app.post("/transcribe")
    async def transcribe(request: Request):
        if remote_mode:
            raise ServiceError(503, "remote_speech_unavailable",
                               "Remote speech inference is not verified. Local speech models are disabled in remote mode.")
        if render_only:
            raise ServiceError(404, "not_found", "Endpoint not available on the chat service.")
        started = time.monotonic()
        async with request.form(max_files=1, max_fields=1, max_part_size=1024) as form:
            if len(form.multi_items()) != 2 or set(form) != {"audio", "language"}:
                raise ServiceError(422, "invalid_fields", "Provide exactly one audio file and one language field.")
            language = form.get("language")
            audio = form.get("audio")
            if not isinstance(language, str) or language not in settings.languages:
                raise ServiceError(422, "unsupported_language", "Select an enabled language: " + ", ".join(settings.languages))
            if not isinstance(audio, UploadFile):
                raise ServiceError(422, "invalid_audio", "The audio field must be a file.")
            data = await audio.read(settings.max_upload_bytes + 1)
            samples = await decoder(data, audio.content_type or "", settings)
            transcript = await engine.transcribe(language, samples)
        return JSONResponse({
            "transcript": transcript, "language": language,
            "model": MODELS[language].model_id, "model_revision": MODELS[language].revision,
            "processing_time_ms": round((time.monotonic() - started) * 1000, 2),
            "request_id": request.state.request_id,
        }, headers={"Cache-Control": "no-store", "X-Request-ID": request.state.request_id})

    @app.get("/health/chat")
    async def chat_health():
        health = {"status": chat_engine.state, "model": settings.natlas_model_id}
        if settings.natlas_backend == "gguf":
            health.update(runtime="gguf-cpu", artifact_sha256=settings.natlas_manifest_sha256)
        elif remote_mode:
            health.update(runtime="remote", last_error=chat_engine.last_error,
                          inference_verified=chat_engine.state == "ready",
                          speech_status="remote_speech_unavailable")
        return JSONResponse(health,
                            status_code=200 if chat_engine.state == "ready" else 503,
                            headers={"Cache-Control": "no-store"})

    @app.post("/chat")
    async def chat(payload: ChatRequest, request: Request):
        started = time.monotonic()
        reply = await chat_engine.generate(payload)
        return JSONResponse({"reply": reply, "language": payload.language,
                             "model": settings.natlas_model_id,
                             "processing_time_ms": round((time.monotonic() - started) * 1000, 2)},
                            headers={"Cache-Control": "no-store", "X-Request-ID": request.state.request_id})

    return app


def create_render_app():
    if Settings().natlas_backend == "remote":
        return create_remote_app()
    from .preparation import RenderChatEngine
    settings = Settings(natlas_backend="gguf", natlas_enabled=True,
                        natlas_managed_artifact=True, hf_local_files_only=True)
    return create_app(settings=settings, chat_engine=RenderChatEngine(settings), render_only=True)


def create_remote_app():
    """Free-tier factory: never create a local model or preparation worker."""
    return create_app(settings=Settings(natlas_backend="remote", natlas_enabled=True))
