import asyncio
import shutil

import numpy as np

from .config import Settings
from .errors import ServiceError

SAMPLE_RATE = 16000
MIME_FORMATS = {
    "audio/webm": "matroska",
    "audio/ogg": "ogg",
    "application/ogg": "ogg",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
}


def audio_format(data: bytes, content_type: str) -> str:
    mime = content_type.split(";", 1)[0].strip().lower()
    expected = MIME_FORMATS.get(mime)
    actual = None
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        actual = "wav"
    elif data.startswith(b"OggS"):
        actual = "ogg"
    elif data.startswith(b"\x1a\x45\xdf\xa3") and b"webm" in data[:4096]:
        actual = "matroska"
    if expected is None or actual != expected:
        raise ServiceError(415, "unsupported_audio", "Upload valid WebM, WAV or OGG audio with its matching MIME type.")
    return actual


async def decode_audio(data: bytes, content_type: str, settings: Settings) -> np.ndarray:
    if not data:
        raise ServiceError(400, "empty_audio", "The audio file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise ServiceError(413, "audio_too_large", "The audio file exceeds the upload limit.")
    container = audio_format(data, content_type)
    binary = shutil.which(settings.ffmpeg_binary)
    if not binary:
        raise ServiceError(503, "decoder_unavailable", "FFmpeg is not available on the inference server.")
    # Explicit demuxer + pipe-only protocols prevent playlists/remote file reads.
    # Decode one extra interval to detect oversize audio without unbounded output.
    proc = await asyncio.create_subprocess_exec(
        binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
        "-protocol_whitelist", "pipe", "-f", container, "-i", "pipe:0",
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-t", str(settings.max_audio_seconds + 0.1), "-f", "f32le", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        output, _ = await asyncio.wait_for(proc.communicate(data), settings.decode_timeout_seconds)
    except (TimeoutError, asyncio.CancelledError) as exc:
        if proc.returncode is None:
            proc.kill()
        await proc.communicate()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise ServiceError(504, "decode_timeout", "Audio decoding timed out.") from None
    if proc.returncode or not output or len(output) % 4:
        raise ServiceError(422, "invalid_audio", "The audio could not be decoded.")
    samples = np.frombuffer(output, dtype="<f4").copy()
    if len(samples) > int(settings.max_audio_seconds * SAMPLE_RATE):
        raise ServiceError(422, "audio_too_long", "The recording exceeds the duration limit.")
    if not np.isfinite(samples).all():
        raise ServiceError(422, "invalid_audio", "The audio contains invalid samples.")
    return samples
