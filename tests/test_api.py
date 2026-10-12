import pytest

from app.errors import ServiceError
from app.registry import MODELS
from conftest import wav_bytes


def upload(client, headers, language="yo", data=None, mime="audio/wav"):
    return client.post("/transcribe", headers=headers, data={"language": language},
                       files={"audio": ("recording.wav", wav_bytes() if data is None else data, mime)})


@pytest.mark.parametrize("language", ["yo", "en-NG"])
def test_success(client, headers, engine, language):
    response = upload(client, headers, language)
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"]
    assert body["language"] == language
    assert body["model"] == MODELS[language].model_id
    assert body["processing_time_ms"] >= 0
    assert response.headers["cache-control"] == "no-store"
    # Real FFmpeg converted stereo 44.1 kHz into mono 16 kHz.
    assert engine.calls[0][1].shape == (3200,)
    assert client.get("/health").json()["loaded_languages"] == [language]


def test_health_is_lazy(client, engine):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_state"] == "not_loaded"
    assert engine.calls == []


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_auth_before_parsing(client, headers, engine):
    assert client.post("/transcribe", headers=headers, content=b"broken").status_code == 401
    assert engine.calls == []


@pytest.mark.parametrize("language", ["en", "Yoruba", "../../model", ""])
def test_unsupported_language(client, headers, language, engine):
    assert upload(client, headers, language).status_code == 422


@pytest.mark.parametrize("language", ["yo", "en-NG", "ha", "ig"])
def test_four_official_languages_are_enabled(client, headers, language, engine):
    assert upload(client, headers, language).status_code == 200
    assert engine.calls[0][0] == language
    assert client.get("/health").json()["loaded_languages"] == [language]


@pytest.mark.parametrize("data,mime,status", [(b"", "audio/wav", 400), (b"not audio", "audio/wav", 415),
    (b"OggSbroken", "audio/ogg", 422), (b"RIFFxxxxWAVEgarbage", "audio/wav", 422),
    (b"anything", "audio/mpeg", 415)])
def test_invalid_audio(client, headers, data, mime, status, engine):
    assert upload(client, headers, data=data, mime=mime).status_code == status
    assert engine.calls == []


def test_mime_spoof(client, headers):
    assert upload(client, headers, mime="audio/ogg").status_code == 415


def test_file_size_limit(client, headers, settings):
    settings.max_upload_bytes = 1024
    response = upload(client, headers)
    assert response.status_code == 413


def test_duration_limit(client, headers, settings):
    settings.max_audio_seconds = 0.1
    response = upload(client, headers)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "audio_too_long"


def test_request_size_before_parsing(client, headers, settings):
    response = client.post("/transcribe", headers={**headers, "Content-Type": "multipart/form-data; boundary=x"},
                           content=b"x" * (settings.max_upload_bytes + 65537))
    assert response.status_code == 413


def test_extra_fields_and_files(client, headers):
    response = client.post("/transcribe", headers=headers, data={"language": "yo", "extra": "bad"},
                           files={"audio": ("audio.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 400
    response = client.post("/transcribe", headers=headers, data={"language": "yo"},
                           files=[("audio", ("a.wav", wav_bytes(), "audio/wav")),
                                  ("audio", ("b.wav", wav_bytes(), "audio/wav"))])
    assert response.status_code == 400


@pytest.mark.parametrize("code,status", [("model_access_denied", 503), ("model_initialization_failed", 503),
    ("worker_unavailable", 503), ("inference_timeout", 504)])
def test_service_errors(client, headers, engine, code, status):
    engine.error = ServiceError(status, code, "Safe message")
    response = upload(client, headers)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_no_decoder(client, headers, settings):
    settings.ffmpeg_binary = "nonexistent-ffmpeg-binary"
    assert client.get("/health").status_code == 503
    assert upload(client, headers).status_code == 503


def test_logs_do_not_contain_audio_or_credentials(client, headers, caplog):
    with caplog.at_level("INFO", logger="lightbridge.asr"):
        upload(client, headers)
    assert "request_id=" in caplog.text
    assert headers["Authorization"] not in caplog.text
    assert "Ẹ káàárọ̀" not in caplog.text


def test_unexpected_failure_is_redacted(client, headers, engine, caplog):
    engine.error = RuntimeError("private-token-must-never-appear")
    response = upload(client, headers)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "private-token" not in response.text
    assert "private-token" not in caplog.text
