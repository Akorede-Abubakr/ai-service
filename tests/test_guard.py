import asyncio

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.main import RequestGuard, create_app
from conftest import TOKEN, wav_bytes


async def test_busy_rejects_without_queue_and_health_responds(settings, engine, headers):
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked(*args):
        entered.set()
        await release.wait()
        return "hello"

    engine.transcribe = blocked
    app = create_app(settings, engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        request = dict(headers=headers, data={"language": "yo"}, files={"audio": ("a.wav", wav_bytes(), "audio/wav")})
        first = asyncio.create_task(client.post("/transcribe", **request))
        await asyncio.wait_for(entered.wait(), 10)
        try:
            second = await client.post("/transcribe", **request)
            assert second.status_code == 429
            assert second.headers["retry-after"] == "5"
            assert (await client.get("/health")).status_code == 200
        finally:
            release.set()
            assert (await first).status_code == 200


@pytest.mark.parametrize("mode,expected", [("too_big", 413), ("slow", 408)])
async def test_stream_without_content_length(settings, mode, expected):
    settings.upload_timeout_seconds = 0.01
    called = False
    messages = []

    async def downstream(*_):
        nonlocal called
        called = True

    async def receive():
        if mode == "slow":
            await asyncio.sleep(1)
        return {"type": "http.request", "body": b"x" * 65536, "more_body": True}

    async def send(message):
        messages.append(message)

    guard = RequestGuard(downstream, settings)
    scope = {"type": "http", "path": "/transcribe", "headers": [
        (b"authorization", ("Bearer " + TOKEN).encode()),
        (b"content-type", b"multipart/form-data; boundary=x")], "state": {}}
    await guard(scope, receive, send)
    assert messages[0]["status"] == expected
    assert not called
    assert not guard.busy


def test_config_requires_service_secret_and_known_languages():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_service_token="short")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_service_token=TOKEN, enabled_languages="unknown")
    assert Settings(_env_file=None, ai_service_token=TOKEN, enabled_languages="yo,en-NG,ha,ig").languages == ("yo", "en-NG", "ha", "ig")
