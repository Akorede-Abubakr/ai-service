from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Uvicorn and diagnostic scripts may be launched from the repository root.
    # Always read the service's env file; process environment still takes priority.
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True,
    )

    ai_service_token: SecretStr = Field(min_length=32)
    hf_token: SecretStr | None = None
    hf_home: str = ".cache/huggingface"
    hf_local_files_only: bool = False
    # Explicit opt-in on the GPU host only. Local development never loads the model by default.
    natlas_enabled: bool = False
    natlas_model_id: Literal["NCAIR1/N-ATLaS"] = "NCAIR1/N-ATLaS"
    natlas_revision: str = "main"
    # Existing installations default to GPU; Render has separate remote/CPU profiles.
    natlas_backend: Literal["transformers", "gguf", "remote"] = "transformers"
    # Our own CPU service's root URL and AI_SERVICE_TOKEN. No third-party default.
    natlas_remote_url: str = ""
    natlas_remote_token: SecretStr | None = None
    natlas_artifact_dir: Path = Path("/data/natlas")
    natlas_managed_artifact: bool = False
    natlas_require_verified_artifact: bool = False
    natlas_setup_mode: Literal["prepare", "serve"] = "serve"
    natlas_prepare_work_dir: Path = Path("/data/natlas/work")
    natlas_prepare_timeout_seconds: int = Field(default=21600, ge=600, le=86400)
    natlas_manifest_sha256: str = Field(default="", pattern=r"^([0-9a-f]{64})?$")
    natlas_gguf_repo: str = ""
    natlas_gguf_revision: str = Field(default="", pattern=r"^([0-9a-f]{40})?$")
    natlas_context_size: int = Field(default=2048, ge=512, le=8192)
    natlas_threads: int = Field(default=4, ge=1, le=64)
    natlas_batch_size: int = Field(default=64, ge=1, le=512)
    natlas_max_input_tokens: int = Field(default=1792, ge=256, le=7000)
    natlas_max_new_tokens: int = Field(default=128, ge=1, le=1000)
    natlas_timeout_seconds: float = Field(default=120, gt=0, le=300)
    request_timeout_seconds: float = Field(default=30, gt=0)
    asr_device: Literal["auto", "cpu", "cuda"] = "auto"
    enabled_languages: str = "yo,en-NG,ha,ig"
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=50 * 1024 * 1024)
    max_audio_seconds: float = Field(default=60, gt=0, le=300)
    upload_timeout_seconds: float = Field(default=30, gt=0)
    decode_timeout_seconds: float = Field(default=20, gt=0)
    inference_timeout_seconds: float = Field(default=120, gt=0)
    model_load_timeout_seconds: float = Field(default=600, gt=0)
    model_retry_seconds: float = Field(default=60, gt=0)
    ffmpeg_binary: str = "ffmpeg"

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(x.strip() for x in self.enabled_languages.split(",")))

    @model_validator(mode="after")
    def check_languages(self):
        from .registry import MODELS

        if not self.languages or any(x not in MODELS for x in self.languages):
            raise ValueError("ENABLED_LANGUAGES must contain registered language codes")
        if self.natlas_remote_url:
            url = urlsplit(self.natlas_remote_url)
            local = url.hostname in ("localhost", "127.0.0.1", "::1")
            if (not url.hostname or url.username or url.password or url.query or url.fragment
                    or url.path not in ("", "/")
                    or url.scheme != "https" and not (local and url.scheme == "http")):
                raise ValueError("NATLAS_REMOTE_URL must be an HTTPS root URL (HTTP only for localhost)")
            self.natlas_remote_url = self.natlas_remote_url.rstrip("/")
        return self
