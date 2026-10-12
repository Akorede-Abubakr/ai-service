"""One persistent, killable model process per API instance; no inference threads."""
import asyncio
import gc
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

from .config import Settings
from .errors import ServiceError, model_error
from .registry import MODELS


def load_pipeline(spec, settings):
    # Lazy imports keep API/mock tests lightweight. FFmpeg provides PCM directly,
    # so librosa, torchaudio, torchcodec and accelerate are not required.
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

    device = "cuda" if settings.asr_device == "auto" and torch.cuda.is_available() else settings.asr_device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    token = settings.hf_token.get_secret_value() if settings.hf_token else False
    options = dict(
        revision=spec.revision, token=token,
        cache_dir=str(Path(settings.hf_home) / "hub"),
        local_files_only=settings.hf_local_files_only,
    )
    processor = WhisperProcessor.from_pretrained(spec.model_id, **options)
    model = WhisperForConditionalGeneration.from_pretrained(
        spec.model_id, **options, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        weights_only=True,
    )
    model.eval()
    return pipeline(
        "automatic-speech-recognition", model=model,
        tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
        device=0 if device == "cuda" else -1,
    )


class ModelCache:
    def __init__(self, settings, loader=load_pipeline):
        self.settings = settings
        self.loader = loader
        self.models = {}
        self.failures = {}
        self.language = None
        self.released_language = None

    def get(self, language):
        if language in self.failures:
            until, error = self.failures[language]
            if time.monotonic() < until:
                raise error
            del self.failures[language]
        if language not in self.models:
            if self.language is not None:
                self.models.clear()
                gc.collect()
                self.released_language = self.language
            try:
                self.models[language] = self.loader(MODELS[language], self.settings)
            except Exception as exc:
                error = model_error(exc)
                self.failures[language] = (time.monotonic() + self.settings.model_retry_seconds, error)
                raise error from None
            self.language = language
        return self.models[language]


def worker_main(connection, settings):
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    # Third-party exceptions may contain signed URLs; only parent logs safe codes.
    logging.disable(logging.CRITICAL)
    cache = ModelCache(settings)
    try:
        while True:
            language, samples = connection.recv()
            try:
                asr = cache.get(language)
                connection.send({"loaded": language, "released": cache.released_language})
                cache.released_language = None
                import torch

                with torch.inference_mode():
                    result = asr(
                        {"raw": samples, "sampling_rate": 16000},
                        chunk_length_s=30, stride_length_s=(5, 5),
                        generate_kwargs={"task": "transcribe"},
                    )
                connection.send({"transcript": result["text"].strip()})
            except ServiceError as exc:
                connection.send({"error": (exc.status, exc.code, exc.message)})
            except Exception:
                connection.send({"error": (503, "inference_failed", "Speech transcription failed.")})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class InferenceEngine:
    def __init__(self, settings, worker_target=worker_main):
        self.settings = settings
        self.worker_target = worker_target
        self.process = None
        self.connection = None
        self.loaded = set()
        self.last_error = None

    def _start(self):
        context = mp.get_context("spawn")  # Safe for CUDA and Windows.
        self.connection, child = context.Pipe()
        self.process = context.Process(target=self.worker_target, args=(child, self.settings), daemon=True)
        self.process.start()
        child.close()

    async def close(self):
        process, connection = self.process, self.connection
        self.process = self.connection = None
        self.loaded.clear()
        if process is not None:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 3)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 3)
            process.close()
        if connection is not None:
            connection.close()

    async def transcribe(self, language, samples):
        try:
            if self.process is None or not self.process.is_alive():
                await self.close()
                self._start()
            # Pipe writes can block; keep the event loop available for health checks.
            async with asyncio.timeout(self.settings.model_load_timeout_seconds):
                await asyncio.to_thread(self.connection.send, (language, samples))
                result = await self._receive()
            if "error" in result:
                raise ServiceError(*result["error"])
            if result.get("released"):
                self.loaded.discard(result["released"])
            self.loaded.add(language)
            async with asyncio.timeout(self.settings.inference_timeout_seconds):
                result = await self._receive()
            if "error" in result:
                raise ServiceError(*result["error"])
            self.last_error = None
            return result["transcript"]
        except TimeoutError:
            self.last_error = "inference_timeout"
            await self.close()
            raise ServiceError(504, "inference_timeout", "Model loading or transcription timed out; retry the recording.") from None
        except asyncio.CancelledError:
            await self.close()
            raise
        except ServiceError as exc:
            self.last_error = exc.code
            raise
        except Exception:
            self.last_error = "worker_unavailable"
            await self.close()
            raise ServiceError(503, "worker_unavailable", "The inference worker is unavailable.") from None

    async def _receive(self):
        while not self.connection.poll():
            if not self.process.is_alive():
                raise RuntimeError("Worker exited")
            await asyncio.sleep(0.02)
        return self.connection.recv()
