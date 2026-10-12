import subprocess

import numpy as np
import pytest

from app.audio import decode_audio
from app.errors import ServiceError
from conftest import wav_bytes


@pytest.mark.parametrize("container,mime", [("webm", "audio/webm;codecs=opus"), ("ogg", "audio/ogg")])
async def test_real_browser_codecs(settings, container, mime):
    encoded = subprocess.run([settings.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                              "-c:a", "libopus", "-f", container, "pipe:1"],
                             input=wav_bytes(), capture_output=True, check=True, timeout=10).stdout
    samples = await decode_audio(encoded, mime, settings)
    assert samples.ndim == 1
    assert samples.dtype == np.float32
    assert 3000 <= len(samples) <= 3400


async def test_decode_timeout(settings):
    settings.decode_timeout_seconds = 0.000001
    with pytest.raises(ServiceError) as exc:
        await decode_audio(wav_bytes(), "audio/wav", settings)
    assert exc.value.code == "decode_timeout"
