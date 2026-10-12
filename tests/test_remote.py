"""Fail-closed checks; no fabricated inference responses."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.chat import LANGUAGES
from app.config import Settings
from app.errors import ServiceError
from app.main import create_app
from app.remote import RemoteChatEngine


def test_missing_host(settings, headers):
    settings.natlas_backend = "remote"
    settings.natlas_enabled = True
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/live").status_code == 200
        for path in ("/health", "/health/chat"):
            health = client.get(path)
            assert health.status_code == 503
            assert health.json()["last_error"] == "remote_endpoint_missing"
            assert health.json()["inference_verified"] is False
        assert client.post("/chat", json={}).status_code == 401
        for language in LANGUAGES:
            response = client.post("/chat", json={"language": language, "messages": [{"role": "user", "content": "Hello"}]}, headers=headers)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "remote_endpoint_missing"
        assert client.post("/transcribe", files={"audio": ("test.wav", b"x")}, headers=headers).json()["error"]["code"] == "remote_speech_unavailable"
        assert client.post("/transcribe").status_code == 401


@pytest.mark.asyncio
async def test_host_credentials_and_pin_required(settings):
    settings.natlas_enabled = True
    settings.natlas_remote_url = "https://cpu.example.com"
    engine = RemoteChatEngine(settings)
    await engine.start()
    assert engine.last_error == "remote_token_missing" and engine.client is None
    settings.natlas_remote_token = SecretStr("our-cpu-service-token")
    await engine.start()
    assert engine.last_error == "remote_artifact_pin_missing" and engine.client is None
    settings.natlas_manifest_sha256 = "a" * 64
    settings.hf_token = SecretStr("hub-token-never-forwarded")
    try:
        await engine.start()
        outgoing = engine.client.build_request("POST", "/chat")
        assert outgoing.headers["authorization"] == "Bearer our-cpu-service-token"
        assert str(outgoing.url) == "https://cpu.example.com/chat"
        assert engine.client.follow_redirects is False and engine.client.trust_env is False
        assert engine.state != "ready"
    finally:
        await engine.close()


@pytest.mark.parametrize("url", ["http://cpu.example.com", "https://user:secret@cpu.example.com", "https://cpu.example.com/chat", "https://cpu.example.com?token=x", "https://cpu.example.com#x", "file:///tmp/model", "https://"])
def test_reject_unsafe_urls(settings, url):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_service_token=settings.ai_service_token, natlas_remote_url=url)


@pytest.mark.parametrize("url", ["https://cpu.example.com/", "http://127.0.0.1:10000"])
def test_valid_urls(settings, url):
    configured = Settings(_env_file=None, ai_service_token=settings.ai_service_token, natlas_remote_url=url)
    assert configured.natlas_remote_url == url.rstrip("/")


@pytest.mark.parametrize("invalid", [{}, {"reply": ""}, {"model": "other/model", "reply": "invalid", "language": "en"}])
def test_invalid_payload(settings, invalid):
    engine = RemoteChatEngine(settings)
    with pytest.raises(ServiceError) as failure:
        engine._reply(invalid, "en")
    assert failure.value.code == "remote_invalid_response"
    assert engine.state != "ready"


def test_no_model_imports():
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
    env = dict(os.environ, AI_SERVICE_TOKEN="test-token-at-least-thirty-two-characters", NATLAS_REMOTE_URL="")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
