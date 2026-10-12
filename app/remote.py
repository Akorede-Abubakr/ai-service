"""Adapter for the inspected zylvex Space API; never loads model weights.

Contract: Space app.py revision a2566d91fede85bf769daacdfb55d1678eafba2a.
Live unauthenticated inference returned 401 on 2026-10-12. The owner's key
is required; source inspection and health are NOT inference verification.
"""
import asyncio
import json

import httpx

from .chat import LANGUAGES
from .errors import ServiceError

SPACE_URL = "https://zylvex-natlas-zylcode-bridge.hf.space"
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
        key = self.settings.natlas_remote_api_key
        if key is None or not key.get_secret_value().strip():
            self.state = "unavailable"
            self.last_error = "remote_api_key_missing"
            return
        self.client = httpx.AsyncClient(
            base_url=SPACE_URL,
            headers={"Authorization": "Bearer " + key.get_secret_value()},
            timeout=httpx.Timeout(self.settings.natlas_timeout_seconds, connect=10),
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            follow_redirects=False, trust_env=False,
        )

    async def _json(self, method, path, **kwargs):
        async with self.client.stream(method, path, **kwargs) as response:
            if response.status_code in (401, 403):
                raise ServiceError(503, "remote_auth_failed", "The Space rejected its API key. Obtain access from the Space owner.")
            if response.status_code == 429:
                raise ServiceError(429, "remote_rate_limited", "The remote inference quota is exhausted. Retry later.")
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
                               "Remote N-ATLaS requires NATLAS_REMOTE_API_KEY issued by the Space owner.")
        if self.busy:
            raise ServiceError(429, "service_busy", "Another AI request is processing; retry shortly.")
        self.busy = True
        try:
            async with asyncio.timeout(self.settings.natlas_timeout_seconds):
                # The completion's model field echoes the requested model. Check
                # the actual configured LLM reported by health on every request.
                health = await self._json("GET", "/healthz")
                if health.get("model") != self.settings.natlas_model_id or health.get("status") != "healthy":
                    raise ServiceError(503, "remote_model_mismatch", "The Space is not reporting the expected N-ATLaS model.")
                result = await self._json("POST", "/v1/chat/completions", json={
                    "model": self.settings.natlas_model_id,
                    "messages": [{"role": "system", "content": (
                        "You are LightBridge AI, a helpful assistant on MUSLIM GUARDIAN, powered by N-ATLaS. "
                        f"Respond in {LANGUAGES[request.language]}. Be clear and honest about uncertainty. "
                        "Do not invent religious citations or claim access to the user's books or account."
                    )}] + [message.model_dump() for message in request.messages],
                    "max_tokens": self.settings.natlas_max_new_tokens,
                    "temperature": 0.1, "repetition_penalty": 1.12, "stream": False,
                })
                reply = self._reply(result)
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

    def _reply(self, result):
        try:
            choices = result["choices"]
            message = choices[0]["message"]
            reply = message["content"]
            usage = result["usage"]
            valid = (
                result["model"] == self.settings.natlas_model_id
                and result["object"] == "chat.completion"
                and len(choices) == 1 and message["role"] == "assistant"
                and choices[0]["finish_reason"] in ("stop", "length")
                and isinstance(reply, str) and 0 < len(reply.strip()) <= 8000
                and type(usage["completion_tokens"]) is int and usage["completion_tokens"] > 0
                and type(usage["prompt_tokens"]) is int and usage["prompt_tokens"] > 0
            )
        except (KeyError, IndexError, TypeError):
            valid = False
        if not valid:
            raise ServiceError(502, "remote_invalid_response", "The Space did not return a usable N-ATLaS completion.")
        return reply.strip()

    async def close(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None
