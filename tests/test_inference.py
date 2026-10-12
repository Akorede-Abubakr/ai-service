import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.errors import ServiceError
from app.inference import InferenceEngine, ModelCache


def slow_worker(connection, settings):
    connection.recv()
    connection.send({"loaded": "yo"})
    time.sleep(30)


def exit_worker(connection, settings):
    connection.close()


def echo_worker(connection, settings):
    loaded = None
    while True:
        try:
            language, samples = connection.recv()
            connection.send({"loaded": language, "released": loaded if loaded != language else None})
            loaded = language
            connection.send({"transcript": "test transcription"})
        except EOFError:
            break


def test_model_cache_reuses_only_the_current_language(settings):
    calls = []
    cache = ModelCache(settings, loader=lambda spec, _: calls.append(spec.model_id) or object())
    assert cache.get("yo") is cache.get("yo")
    assert cache.get("en-NG") is cache.get("en-NG")
    assert cache.get("ig") is cache.get("ig")
    assert calls == ["NCAIR1/Yoruba-ASR", "NCAIR1/NigerianAccentedEnglish", "NCAIR1/Igbo-ASR"]
    assert len(cache.models) == 1 and cache.language == "ig"
    assert cache.released_language == "en-NG"


@pytest.mark.parametrize("status,code", [(401, "model_access_denied"), (403, "model_access_denied"),
    (500, "model_initialization_failed")])
def test_loader_failure_is_safe_and_backed_off(settings, status, code):
    calls = []

    def broken_loader(*_):
        calls.append(True)
        inner = RuntimeError("secret-token-do-not-log")
        inner.response = SimpleNamespace(status_code=status)
        raise OSError("private signed URL") from inner

    cache = ModelCache(settings, loader=broken_loader)
    for _ in range(2):
        with pytest.raises(ServiceError) as exc:
            cache.get("yo")
        assert exc.value.code == code
        assert "secret-token" not in str(exc.value)
    assert len(calls) == 1


async def test_real_process_timeout_terminates_worker(settings):
    settings.inference_timeout_seconds = 0.1
    engine = InferenceEngine(settings, worker_target=slow_worker)
    with pytest.raises(ServiceError) as exc:
        await engine.transcribe("yo", np.zeros(1600))
    assert exc.value.status == 504
    assert engine.process is None
    assert not engine.loaded
    # The same API instance recovers on the next request.
    engine.worker_target = echo_worker
    try:
        assert await engine.transcribe("yo", np.zeros(1600)) == "test transcription"
    finally:
        await engine.close()


async def test_worker_crash(settings):
    engine = InferenceEngine(settings, worker_target=exit_worker)
    with pytest.raises(ServiceError) as exc:
        await engine.transcribe("yo", np.zeros(1600))
    assert exc.value.status == 503
    assert engine.process is None


async def test_persistent_process(settings):
    engine = InferenceEngine(settings, worker_target=echo_worker)
    try:
        await engine.transcribe("yo", np.zeros(1600))
        pid = engine.process.pid
        await engine.transcribe("en-NG", np.zeros(1600))
        assert engine.process.pid == pid
        assert engine.loaded == {"en-NG"}
    finally:
        await engine.close()
