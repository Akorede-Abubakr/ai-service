# LightBridge AI — N-ATLaS text service

For **Render Free (512 MB)** use the HTTP-only `Dockerfile.remote`, `requirements-remote.txt` and `render-free.env.example`. This mode preserves the FastAPI chat contract and four languages without importing, downloading or loading any local model. **Blocked: the inspected Space requires its owner's API key; live unauthenticated chat returned 401. Real remote inference has not passed and nothing has been deployed.** See [remote setup and verification](REMOTE.md). `HF_TOKEN` is not the Space API key.

The optional **paid CPU** Render path prepares and hosts `NCAIR1/N-ATLaS` entirely on Render. A finite preparation job downloads the official model, converts it to Q4_K_M GGUF, saves it on the service's persistent disk and runs genuine inference checks. Serving mode reuses that artifact with CPU-only llama.cpp. No Kaggle, tunnel or substitute language model is used by this path.

**Status: implementation prepared; CPU conversion, Docker build, and real generation are not yet verified. No Render deployment has been made.** See [the Render CPU guide](../docs/render-cpu.md) for evidence, conversion, resource estimates, secrets, smoke tests, cutover, rollback, and current blockers.

| File | Purpose |
| --- | --- |
| `Dockerfile.remote` / `requirements-remote.txt` | Free-tier HTTP adapter; no local model dependencies |
| `app/remote.py` / `render-free.env.example` | Verified Space API contract, separate owner-issued API key |
| `scripts/smoke_remote.py` | Real four-language remote inference and history check; fails when blocked |
| `Dockerfile` / `requirements-convert.txt` | Render preparation and serving; CPU torch only for conversion, no CUDA/bitsandbytes |
| `app/preparation.py` | Supervised first-run preparation and genuine inference verification |
| `Dockerfile.chat`, `Dockerfile.convert` | Historical standalone configurations; not used by the Render workflow |
| `render.env.example` | CPU settings template; populate secrets outside Git |
| `scripts/smoke_cpu.py` | Genuine local or remote inference in all four languages |

## Endpoints

- `POST /chat`: requires `Authorization: Bearer AI_SERVICE_TOKEN`; accepts `{"language":"en","messages":[{"role":"user","content":"Hello"}]}` and returns `reply`, `language`, `model`, and `processing_time_ms`.
- `GET /health/chat`: chat readiness (`ready` → 200; disabled/unavailable → 503).
- `GET /health`: speech infrastructure health in the default Transformers configuration; chat readiness in GGUF CPU mode.
- `POST /transcribe`: restored authenticated audio transcription, described below.

Both Render modes expose chat readiness at `/health` and process liveness at `/live`. In remote mode, `/transcribe` remains authenticated but returns 503 `remote_speech_unavailable`; the Space reports no loaded ASR models and speech inference has not been verified. Remote chat readiness stays 503 until an actual completion succeeds. There is no local speech/model fallback on the free instance.

For the paid CPU factory, `/transcribe` remains unavailable. Start with `NATLAS_SETUP_MODE=prepare` and Render health path `/live`, then switch to `serve` and `/health` only after verified preparation. A green preparation deploy is not a claim of model readiness. See [exact CPU instructions](../docs/render-cpu.md).

Languages are `en`, `yo`, `ha`, `ig`. Requests must alternate user/assistant turns, starting and ending with user. Limits: 21 messages, 8,000 characters per message, 24,000 total characters and 128 KiB JSON. The server owns the system prompt and generation settings. Auth and body limits apply before parsing. Requests are serialized, with safe errors and no prompt/token logging.

## Laptop development and tests

Keep `NATLAS_ENABLED=false`. The API/test requirements do not install model libraries or weights:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe scripts/start_local.py
```

The local launcher explicitly disables model loading. A disabled `/chat` returns 503, never a fake reply. Tests inject test-only engines. Speech decoding requires NumPy and FFmpeg; development tests use the bundled FFmpeg dependency.

## GPU deployment

See [the GPU deployment guide](../docs/lightbridge-chat.md). `Dockerfile.chat` installs `requirements-chat.txt`, which includes the pinned GPU model stack. Set `NATLAS_BACKEND=transformers` for this fallback. FFmpeg and multipart support are restored for speech. Enable `NATLAS_ENABLED`, use one Uvicorn worker without reload, and wait for readiness. Model loading occurs once in lifespan, never in `/chat`.

The model uses 4-bit NF4/double quantization, float16 compute and `device_map="auto"`. CPU/disk offload is rejected. The tokenizer's chat template and generation implementation in `app/chat.py` are preserved. Configure `AI_CHAT_API_FORMAT=native` on Express for this API; Kaggle uses the distinct `message`/`response` contract described in [kaggle-chat.md](../docs/kaggle-chat.md).

Secrets belong in the GPU host environment/secret manager, never browser variables or Docker layers. Copy `.env.example` only when no `.env` exists. HTTP request-read timeout is `REQUEST_TIMEOUT_SECONDS`; generation timeout is `NATLAS_TIMEOUT_SECONDS`. A timed-out worker retains its busy slot until decoding finishes. A stuck CUDA kernel requires restarting the service. No deployment or local model download is performed by cleanup/testing.

# LightBridge AI private NCAIR speech service


The optional local speech mode uses FastAPI + Hugging Face Transformers + PyTorch + FFmpeg. It has no remote inference dependency. This service has no browser credentials or MongoDB coupling. The following speech setup is for separately resourced local inference, not the free-tier remote profile.

## Official model access

| Language code | Checkpoint | Default |
| --- | --- | --- |
| `yo` | [NCAIR1/Yoruba-ASR](https://huggingface.co/NCAIR1/Yoruba-ASR) | Enabled |
| `en-NG` | [NCAIR1/NigerianAccentedEnglish](https://huggingface.co/NCAIR1/NigerianAccentedEnglish) | Enabled |
| `ha` | [NCAIR1/Hausa-ASR](https://huggingface.co/NCAIR1/Hausa-ASR) | Disabled |
| `ig` | [NCAIR1/Igbo-ASR](https://huggingface.co/NCAIR1/Igbo-ASR) | Disabled |

On 2026-10-09, public Hugging Face metadata confirmed all four repositories exist, are gated, and supply Whisper checkpoint/tokenizer files. Exact commits are pinned in `app/registry.py`. Public metadata access does **not** demonstrate account authorization or successful inference.

Log into Hugging Face, review and accept each enabled repository's terms, then create a read token authorized for those repositories. Set `HF_TOKEN` only in this service's environment. These model cards have a custom license, attribution requirements, a 1,000-active-user cap and commercial-use conditions; review the linked terms with Awarri before production/commercial release. This implementation does not accept those terms for you.

Attribution: Yoruba-ASR is developed by Awarri Technologies in partnership with the Federal Government of Nigeria. The Nigerian English, Hausa and Igbo models are also official NCAIR/Awarri ecosystem checkpoints; preserve each model card's required attribution in eventual public UI/docs.

## Local setup

Run commands from `ai-service`. Python 3.12 is the recommended baseline; the local automated suite also runs on Python 3.14. Install system FFmpeg and confirm `ffmpeg -version`, or set `FFMPEG_BINARY` to its full executable path. Docker includes FFmpeg.

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements pin Transformers 4.57.6 and PyTorch 2.9.1. Official repositories currently provide `.bin` weights; the loader uses `weights_only=True`, no remote repository code, and no optimizer/training files. FFmpeg directly supplies NumPy PCM, so librosa, torchaudio, torchcodec and accelerate are unnecessary. CPU is supported; `ASR_DEVICE=auto` selects CUDA only if PyTorch reports it available. For GPU, install the matching PyTorch wheel for your host's driver using [official PyTorch instructions](https://pytorch.org/get-started/locally/).

Copy `.env.example` to `.env` if no `.env` exists. Generate a server secret with:

```sh
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set `AI_SERVICE_TOKEN` to that value (at least 32 characters), and set `HF_TOKEN`. No default usable service secret is supplied. Do not commit either token. To verify account access using small configuration files only:

```sh
python scripts/check_models.py
```

Run one API worker bound to loopback:

```sh
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

For local development, `python scripts/start_local.py` invokes this same factory and can discover the dev-only bundled FFmpeg if system FFmpeg is unavailable. On Windows, do not use `--reload` or multiple Uvicorn workers: the installed Uvicorn chooses a Selector event loop in those modes, which cannot execute the asynchronous FFmpeg decoder. Stop the old development server before starting the helper on port 8000. See the [website integration guide](../docs/speech-to-text.md).

The first transcription loads that language's model; subsequent requests reuse it. Each Whisper Small weight file is roughly 1 GB on disk; plan RAM/VRAM for both models and inference overhead and measure on the target host. Models remain cached under `HF_HOME/hub`. After prewarming every enabled language at the pinned revisions, `HF_LOCAL_FILES_ONLY=true` permits offline startup/reuse. Restarts reload from disk, without redownloading cached weights. Do not use multiple Uvicorn workers on a constrained host: each would allocate its own model process/cache in memory.

## API contract

`GET /health` is a minimal unauthenticated infrastructure health check. It reports decoder availability, enabled/loaded languages and the last safe worker error code. `status=ok` with `model_state=not_loaded` means the API can accept requests, **not** that gated model access is verified. After model failure health returns 503 until a successful retry. Use real prewarming before routing production traffic.

`POST /transcribe` requires `Authorization: Bearer <AI_SERVICE_TOKEN>` and multipart fields:

- `audio`: exactly one WebM, WAV or OGG file.
- `language`: exactly `yo` or `en-NG` by default.

Accepted MIME types are `audio/webm` (including `;codecs=opus`), `audio/ogg`, `application/ogg`, `audio/wav`, `audio/wave`, `audio/x-wav`. MP3, MP4/AAC and mismatched container/MIME pairs are rejected. FFmpeg validates/decodes content, rather than trusting filenames or duration metadata. Default limits are 10 MiB per file and 60 seconds of decoded audio. Audio becomes mono 16 kHz float32; recordings over 30 seconds use overlapping Whisper pipeline chunks. The checkpoint's language configuration is preserved and the generation task is transcription.

```json
{
  "transcript": "Ẹ káàárọ̀",
  "language": "yo",
  "model": "NCAIR1/Yoruba-ASR",
  "model_revision": "d1ae7b8b79c2ccd547d8761effe5057433f3fc7f",
  "processing_time_ms": 1250.5,
  "request_id": "server-generated-id"
}
```

Errors use `{"error":{"code":"...","message":"..."},"request_id":"..."}`. Statuses: 400 malformed/empty upload; 401 service authentication; 408 upload timeout; 413 size limit; 415 unsupported MIME/container; 422 fields/language/decode/duration; 429 busy with Retry-After; 503 model access/initialization/worker/decoder failure; 504 decode/model timeout. No transcript, audio, token, filename, signed Hub URL or raw inference exception is logged. Responses have `Cache-Control: no-store`.

Authentication and a streamed request-byte cap run before multipart parsing. Only one upload/decode/inference job is admitted per API instance; excess requests receive 429 rather than accumulate in memory. The API event loop remains available for health checks while a persistent spawned process performs inference. Separate cold-load and inference deadlines kill/reset that process when exceeded, releasing its model allocations; a subsequent request reloads from disk. Initialization failures have a retry cooldown to avoid repeatedly contacting Hugging Face. Multipart temporary files are closed after use; bounded PCM remains in memory only for the job. FFmpeg gets pipe-only protocols and explicit container demuxers.

For later rollout, `ENABLED_LANGUAGES=yo,en-NG,ha,ig` enables registered Hausa/Igbo checkpoints without a loader rewrite. Validate access and actual language quality first; the initial browser/proxy contract should continue exposing only the two tested language choices.

## Automated and manual tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Python tests do not import/download model weights. They use injected inference, real FFmpeg WAV/WebM/OGG decoding (a dev-only bundled FFmpeg), process timeout/crash/recovery checks, upload bounds, authentication, language validation, cache reuse, safe error mapping for Hugging Face 401/403 and model initialization failure, and concurrency/upload-timeout checks. The website integration now also has separate Express tests and Playwright microphone/browser tests; see the [integration guide](../docs/speech-to-text.md).

With accepted model access, the full dependencies installed, server running and a short spoken recording, run the real checkpoint check separately:

```sh
python -m pip install httpx==0.28.1
python scripts/manual_integration.py path/to/yoruba.wav --language yo
python scripts/manual_integration.py path/to/english.webm --language en-NG
```

Run each twice to observe warm reuse. Review transcript accuracy against the recording. Also test a recording longer than 30 seconds to confirm chunk continuity. This script reads the service token from `.env`, never command-line arguments, and prints the transcript only for this explicit manual check. Allow the initial weight download time; adjust `MODEL_LOAD_TIMEOUT_SECONDS` if your connection is slow.

## Docker and independent deployment

Build from `ai-service` (Docker is not required for local tests):

```sh
docker build -f Dockerfile.speech -t lightbridge-asr .
docker run --rm --env-file .env -e HF_HOME=/data/huggingface -p 127.0.0.1:8000:8000 -v lightbridge-hf:/data/huggingface lightbridge-asr
```

The image is CPU-first, runs as non-root and does not contain tokens or predownloaded weights. The mounted cache persists model files. For CUDA-capable hosts with the NVIDIA container runtime, build using `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128`, run with `--gpus all`, and set `ASR_DEVICE=cuda`; verify compatibility with the actual driver first. GPU execution is not tested on this local machine.

Keep port 8000 on a private network accessible only by Express. Use TLS between hosts and a secret manager for tokens. Add infrastructure request/concurrency limits in addition to Express's per-user rate limit; match proxy timeouts to upload + decode + cold-load + inference budgets or prewarm before admission. Containers and API workers can scale independently of static hosting; allocate one model worker per instance and persistent model storage.

Cloudflare Pages/ordinary Workers host the static React apps, **not PyTorch inference**. See the [root README](../README.md) for frontend/admin Cloudflare configuration. Docker builds, real checkpoint inference, GPU deployment and paid resources require separate validation; no cloud deployment has been performed.
