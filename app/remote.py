"""HTTP-only adapter to OUR existing CPU FastAPI service (app/main.py).

No default provider, model downloads or substitution. The official HF model
had no inference provider on 2026-10-12; a separately hosted CPU service is
required. Readiness requires an actual completion, not a health response.
"""
import asyncio
import json

import httpx

from .errors import ServiceError

MAX_RESPONSE_BYTES = 64 * 1024


class RemoteChatEngine:
    def __init__(self, settings):
        self.settings = settings
        self.state = "disabled" if not settings.natlas_enabled else "unverified"
        self.last_error = None
        self.client = None
        self.busy = False

    async def start(self):
        if self.state == "disabled" or self.client is not None:
            return
        key = self.settings.natlas_remote_token
        missing = ("remote_endpoint_missing" if not self.settings.natlas_remote_url else
                   "remote_token_missing" if key is None or not key.get_secret_value().strip() else
                   "remote_artifact_pin_missing" if not self.settings.natlas_manifest_sha256 else None)
        if missing:
            self.state = "unavailable"
            self.last_error = missing
            return
        self.client = httpx.AsyncClient(
            base_url=self.settings.natlas_remote_url,
            headers={"Authorization": "Bearer " + key.get_secret_value()},
            timeout=httpx.Timeout(self.settings.natlas_timeout_seconds, connect=10),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            follow_redirects=False, trust_env=False,
        )
        self.state, self.last_error = "unverified", None

    async def _json(self, method, path, **kwargs):
        async with self.client.stream(method, path, **kwargs) as response:
            if response.status_code in (401, 403):
                raise ServiceError(503, "remote_auth_failed", "The CPU host rejected NATLAS_REMOTE_TOKEN.")
            if response.status_code == 429:
                raise ServiceError(429, "remote_rate_limited", "The CPU inference host is busy. Retry shortly.")
            if response.status_code == 422:
                raise ServiceError(422, "remote_request_rejected", "The CPU host rejected the conversation. Shorten it and retry.")
            if response.status_code == 504:
                raise ServiceError(504, "inference_timeout", "The CPU host timed out during inference.")
            if response.status_code != 200:
                raise ServiceError(503, "remote_unavailable", "The remote inference service is unavailable.")
            if response.headers.get("content-type", "").split(";")[0].strip() != "application/json":
                raise ServiceError(502, "remote_invalid_response", "The remote service did not return valid inference data.")
            body = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=8192):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ServiceError(502, "remote_invalid_response", "The remote response exceeded the size limit.")
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeError):
                payload = None
            if not isinstance(payload, dict) or "error" in payload:
                raise ServiceError(502, "remote_invalid_response", "The remote service did not return valid inference data.")
            return payload

    async def generate(self, request):
        if self.client is None:
            raise ServiceError(503, self.last_error or "chat_unavailable",
                               "Configure your own N-ATLaS CPU host URL, service token and verified artifact hash. No hosted provider is configured.")
        if self.busy:
            raise ServiceError(429, "service_busy", "Another AI request is processing; retry shortly.")
        self.busy = True
        try:
            async with asyncio.timeout(self.settings.natlas_timeout_seconds):
                # These endpoints are defined by our existing CPU application.
                health = await self._json("GET", "/health/chat")
                if (health.get("model"), health.get("status"), health.get("runtime"), health.get("artifact_sha256")) != (
                        self.settings.natlas_model_id, "ready", "gguf-cpu", self.settings.natlas_manifest_sha256):
                    raise ServiceError(503, "remote_model_mismatch", "The CPU host is not serving the expected verified N-ATLaS artifact.")
                result = await self._json("POST", "/chat", json=request.model_dump())
                reply = self._reply(result, request.language)
                self.state, self.last_error = "ready", None
                return reply
        except (TimeoutError, httpx.TimeoutException):
            self.state, self.last_error = "unavailable", "inference_timeout"
            raise ServiceError(504, self.last_error, "Remote inference timed out. Retry later.") from None
        except httpx.HTTPError:
            self.state, self.last_error = "unavailable", "remote_unavailable"
            raise ServiceError(503, self.last_error, "The remote inference service could not be reached.") from None
        except ServiceError as exc:
            self.state, self.last_error = "unavailable", exc.code
            raise
        except asyncio.CancelledError:
            self.state, self.last_error = "unverified", "remote_request_cancelled"
            raise
        finally:
            self.busy = False

    def _reply(self, result, language):
        try:
            reply = result["reply"]
            valid = (
                result["model"] == self.settings.natlas_model_id
                and result["language"] == language
                and isinstance(reply, str) and 0 < len(reply.strip()) <= 8000
            )
        except (KeyError, IndexError, TypeError):
            valid = False
        if not valid:
            raise ServiceError(502, "remote_invalid_response", "The CPU host did not return a usable N-ATLaS completion.")
        return reply.strip()

    async def close(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None
