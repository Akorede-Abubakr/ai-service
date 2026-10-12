"""Transport doubles test adapter behavior only; NOT real model verification."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.chat import ChatRequest, LANGUAGES
from app.errors import ServiceError
from app.main import create_app
from app.remote import RemoteChatEngine, SPACE_URL


def request(language="en"):
    return ChatRequest(language=language, messages=[{"role": "user", "content": "Hello"}])


def completion():
    return {"model": "NCAIR1/N-ATLaS", "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "TEST ONLY: not real inference"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6}}


async def connected(settings, handler):
    from pydantic import SecretStr
    settings.natlas_backend = "remote"
    settings.natlas_enabled = True
    settings.natlas_remote_api_key = SecretStr("unit-test-space-key")
    engine = RemoteChatEngine(settings)
    await engine.start()
    await engine.client.aclose()
    engine.client = httpx.AsyncClient(base_url=SPACE_URL, transport=httpx.MockTransport(handler),
                                    headers={"Authorization": "Bearer unit-test-space-key"})
    return engine


def test_missing_dependency_auth_and_no_local_load(settings, headers, monkeypatch):
    from unittest.mock import Mock
    import app.main as main
    forbidden = Mock(side_effect=AssertionError("No local engine may be constructed"))
    monkeypatch.setattr(main, "InferenceEngine", forbidden)
    monkeypatch.setattr(main, "ChatEngine", forbidden)
    settings.natlas_backend = "remote"
    settings.natlas_enabled = True
    settings.natlas_remote_api_key = None
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/live").status_code == 200
        for path in ("/health", "/health/chat"):
            health = client.get(path)
            assert health.status_code == 503
            assert health.json()["last_error"] == "remote_api_key_missing"
            assert health.json()["inference_verified"] is False
        assert client.post("/chat", json=request().model_dump()).status_code == 401
        for language in LANGUAGES:
            response = client.post("/chat", json=request(language).model_dump(), headers=headers)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "remote_api_key_missing"
        assert client.post("/transcribe", files={"audio": ("test.wav", b"x")}, headers=headers).json()["error"]["code"] == "remote_speech_unavailable"
        assert client.post("/transcribe").status_code == 401
    forbidden.assert_not_called()


@pytest.mark.asyncio
async def test_production_client_uses_only_owner_key(settings):
    from pydantic import SecretStr
    settings.natlas_enabled = True
    settings.natlas_remote_api_key = SecretStr("owner-issued-test-key")
    settings.hf_token = SecretStr("hub-token-must-not-be-forwarded")
    engine = RemoteChatEngine(settings)
    try:
        await engine.start()
        outgoing = engine.client.build_request("POST", "/v1/chat/completions")
        assert outgoing.headers["authorization"] == "Bearer owner-issued-test-key"
        assert str(outgoing.url) == SPACE_URL + "/v1/chat/completions"
        assert engine.client.follow_redirects is False
        assert engine.client.trust_env is False
        assert engine.state == "unverified"
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_request_contract_languages_history_and_readiness(settings):
    seen = []
    def handler(req):
        assert req.url.host == "zylvex-natlas-zylcode-bridge.hf.space"
        assert req.headers["authorization"] == "Bearer unit-test-space-key"
        if req.url.path == "/healthz":
            return httpx.Response(200, json={"model": "NCAIR1/N-ATLaS", "status": "healthy"})
        assert req.url.path == "/v1/chat/completions" and req.method == "POST"
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=completion())
    engine = await connected(settings, handler)
    try:
        assert engine.state == "unverified"
        for language, name in LANGUAGES.items():
            payload = request(language)
            payload = ChatRequest(language=language, messages=payload.model_dump()["messages"] + [
                {"role": "assistant", "content": "Earlier reply"}, {"role": "user", "content": "Follow up"}])
            assert await engine.generate(payload) == "TEST ONLY: not real inference"
            assert name in seen[-1]["messages"][0]["content"]
            assert seen[-1]["messages"][1:] == payload.model_dump()["messages"]
            assert seen[-1]["stream"] is False
            assert seen[-1]["model"] == "NCAIR1/N-ATLaS"
            assert seen[-1]["max_tokens"] == settings.natlas_max_new_tokens
        assert engine.state == "ready"
    finally:
        await engine.close()


@pytest.mark.parametrize("status,expected,code", [
    (401, 503, "remote_auth_failed"), (403, 503, "remote_auth_failed"),
    (429, 429, "remote_rate_limited"), (503, 503, "remote_unavailable"),
    (302, 503, "remote_unavailable"),
])
@pytest.mark.asyncio
async def test_safe_failures_and_recovery(settings, status, expected, code):
    failing = True
    def handler(req):
        if req.url.path == "/healthz":
            return httpx.Response(200, json={"model": "NCAIR1/N-ATLaS", "status": "healthy"})
        return httpx.Response(status, text="PRIVATE UPSTREAM SECRET") if failing else httpx.Response(200, json=completion())
    engine = await connected(settings, handler)
    try:
        with pytest.raises(ServiceError) as failure:
            await engine.generate(request())
        assert (failure.value.status, failure.value.code) == (expected, code)
        assert "PRIVATE" not in str(failure.value)
        assert engine.state == "unavailable" and not engine.busy
        failing = False
        await engine.generate(request())
        assert engine.state == "ready" and engine.last_error is None
    finally:
        await engine.close()


@pytest.mark.parametrize("invalid", [
    {}, {**completion(), "model": "other/model"}, {**completion(), "choices": []},
    {**completion(), "usage": {"prompt_tokens": 1, "completion_tokens": 0}},
    {**completion(), "error": "private"}, {**completion(), "choices": "wrong"},
    {**completion(), "choices": [{"message": {"role": "assistant", "content": " "}, "finish_reason": "stop"}]},
])
@pytest.mark.asyncio
async def test_invalid_completions_never_succeed(settings, invalid):
    def handler(req):
        return httpx.Response(200, json={"model": "NCAIR1/N-ATLaS", "status": "healthy"} if req.url.path == "/healthz" else invalid)
    engine = await connected(settings, handler)
    try:
        with pytest.raises(ServiceError) as failure:
            await engine.generate(request())
        assert failure.value.code == "remote_invalid_response"
        assert engine.state != "ready"
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_health_cannot_substitute_for_inference_or_claim_other_model(settings):
    engine = await connected(settings, lambda req: httpx.Response(200, json={"model": "other/model", "status": "healthy"}))
    try:
        with pytest.raises(ServiceError) as failure:
            await engine.generate(request())
        assert failure.value.code == "remote_model_mismatch"
        assert engine.state != "ready"
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_deadline_and_concurrent_requests(settings):
    entered = asyncio.Event()
    async def handler(req):
        entered.set()
        await asyncio.sleep(5)
    settings.natlas_timeout_seconds = 0.05
    engine = await connected(settings, handler)
    task = asyncio.create_task(engine.generate(request()))
    try:
        await entered.wait()
        with pytest.raises(ServiceError) as busy:
            await engine.generate(request())
        assert busy.value.status == 429
        with pytest.raises(ServiceError) as timeout:
            await task
        assert timeout.value.status == 504
        assert not engine.busy and engine.state != "ready"
    finally:
        await engine.close()


@pytest.mark.parametrize("kind", ["html", "large", "network", "malformed"])
@pytest.mark.asyncio
async def test_bounded_responses_and_network_errors(settings, kind):
    def handler(req):
        if kind == "network":
            raise httpx.ConnectError("SECRET", request=req)
        if kind == "html":
            return httpx.Response(200, text="<html>Login</html>")
        return httpx.Response(200, content=b"x" * (70000 if kind == "large" else 4), headers={"content-type": "application/json"})
    engine = await connected(settings, handler)
    try:
        with pytest.raises(ServiceError) as failure:
            await engine.generate(request())
        assert failure.value.status == (503 if kind == "network" else 502)
        assert "SECRET" not in str(failure.value)
    finally:
        await engine.close()


def test_remote_factory_imports_no_local_model_or_audio_libraries():
    # Fresh interpreter so earlier CPU/speech tests cannot mask eager imports.
    code = '''
import sys, asyncio
from app.main import create_remote_app
app = create_remote_app()
async def check():
    async with app.router.lifespan_context(app):
        forbidden = ('torch', 'transformers', 'llama_cpp', 'huggingface_hub', 'numpy', 'app.cpu', 'app.preparation', 'app.audio')
        assert not any(name in sys.modules for name in forbidden)
asyncio.run(check())
'''
    import os
    env = dict(os.environ, AI_SERVICE_TOKEN="test-token-at-least-thirty-two-characters", NATLAS_REMOTE_API_KEY="")
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
