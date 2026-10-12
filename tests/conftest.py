import io
import wave

import imageio_ffmpeg
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

TOKEN = "test-service-token-that-is-at-least-32-characters"


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, ai_service_token=TOKEN, hf_home=str(tmp_path),
                    ffmpeg_binary=imageio_ffmpeg.get_ffmpeg_exe())


def wav_bytes(seconds=0.2, rate=44100, channels=2):
    samples = (np.sin(np.arange(int(seconds * rate)) * 440 * 2 * np.pi / rate) * 1000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.repeat(samples[:, None], channels, axis=1).tobytes())
    return buffer.getvalue()


class FakeEngine:
    def __init__(self):
        self.loaded = set()
        self.last_error = None
        self.calls = []
        self.error = None

    async def transcribe(self, language, samples):
        self.calls.append((language, samples))
        if self.error:
            raise self.error
        self.loaded.add(language)
        return "Ẹ káàárọ̀" if language == "yo" else "Good morning"

    async def close(self):
        pass


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(settings, engine):
    with TestClient(create_app(settings, engine)) as client:
        yield client


@pytest.fixture
def headers():
    return {"Authorization": "Bearer " + TOKEN}
