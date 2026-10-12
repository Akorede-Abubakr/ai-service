# RunPod GPU Pod deployment

This is the RunPod GPU deployment of the existing Transformers chat implementation.
It loads the gated `NCAIR1/N-ATLaS` repository directly with
`AutoTokenizer`/`AutoModelForCausalLM`, 4-bit bitsandbytes NF4 with double
quantization, float16 compute, `device_map="auto"`, the tokenizer's chat
template, and `model.generate()`. It does not convert or mirror model weights.

The earlier Kaggle live validation is recorded in `../docs/kaggle-chat.md`.
No notebook file was present in this checkout; this deployment reuses the
checked-in loader in `app/chat.py` and FastAPI routes in `app/main.py`.

## Build the RunPod image

From the repository root, build from the `ai-service` directory and push this
image to a registry RunPod can pull. Log into the registry interactively; do not embed registry credentials,
`HF_TOKEN`, or the service token in image layers or command arguments.

```sh
cd ai-service
docker build -f Dockerfile.runpod -t <registry>/lightbridge-natlas:runpod .
docker push <registry>/lightbridge-natlas:runpod
```

The image pins Python 3.12, PyTorch 2.9.1 CUDA 12.6, Transformers 4.57.6,
Accelerate 1.11.0, and bitsandbytes 0.48.2 through the existing requirements
files. It launches one Uvicorn worker on `0.0.0.0:10000`.

## Create the Pod

In RunPod, create a GPU Pod with:

| Setting | Value |
|---|---|
| GPU | NVIDIA RTX A5000, 24 GB VRAM; if unavailable, select a CUDA GPU with at least 24 GB VRAM |
| Container image | The image pushed above |
| Container disk | 30 GB or more |
| Network volume | 40 GB or more, attached in the same data center |
| Volume mount path | `/workspace` |
| Exposed HTTP port | `10000/http` |
| Container start command override | Blank; use the image command |
| Replicas / workers | One Pod / one Uvicorn worker |

RunPod's exposed-port URL has the form
`https://<POD_ID>-10000.proxy.runpod.net`. The network volume is needed because
model files are cached below `/workspace/huggingface` and the container's
writable layer is not the durable model cache. [RunPod Pod/port reference](https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods),
[RunPod network-volume behavior](https://docs.runpod.io/pods/troubleshooting/zero-gpus).

## Pod environment variables

Set these in the Pod's environment settings. Use secret fields for both tokens.

```dotenv
PORT=10000
HF_HOME=/workspace/huggingface
HF_TOKEN=<Hugging Face read token with accepted NCAIR1/N-ATLaS access>
AI_SERVICE_TOKEN=<random secret of at least 32 characters>
ASR_DEVICE=auto
ENABLED_LANGUAGES=yo,en-NG,ha,ig
NATLAS_ENABLED=true
NATLAS_BACKEND=transformers
NATLAS_MODEL_ID=NCAIR1/N-ATLaS
NATLAS_REVISION=main
NATLAS_MAX_INPUT_TOKENS=4096
NATLAS_MAX_NEW_TOKENS=512
NATLAS_TIMEOUT_SECONDS=120
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_PROGRESS_BARS=1
```

`AI_SERVICE_TOKEN` is the existing FastAPI Bearer credential. `HF_TOKEN` stays
on the Pod and needs accepted read access to `NCAIR1/N-ATLaS` and all four
official speech repositories: `NCAIR1/Yoruba-ASR`,
`NCAIR1/NigerianAccentedEnglish`, `NCAIR1/Hausa-ASR`, and `NCAIR1/Igbo-ASR`.
The authenticated API serves the official speech models at `/transcribe`; they
load on first use and keep one language model at a time. `ASR_DEVICE=auto`
uses the Pod GPU when available. The image starts with:

```sh
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 10000 --workers 1 --no-access-log
```

Wait for `GET /health/chat` to report HTTP 200 and `status: ready`. That means
weights loaded; it is not an inference test. The backend API stays
`POST /chat` with `{ "language": "en|yo|ha|ig", "messages": [...] }` and returns
`reply`, `language`, `model`, and `processing_time_ms` under the existing
LightBridge response envelope. `POST /transcribe` accepts the existing
authenticated multipart `audio` and `language` fields (`yo`, `en-NG`, `ha`,
`ig`) and returns the model-specific transcript. FFmpeg and
PyTorch/Transformers are already installed by `Dockerfile.runpod`.

## Backend environment

Set on the Node backend only after the Pod has loaded the model and passed the
live smoke test:

```dotenv
AI_SERVICE_URL=https://<POD_ID>-10000.proxy.runpod.net
AI_SERVICE_TOKEN=<same AI_SERVICE_TOKEN configured on the Pod>
AI_CHAT_API_FORMAT=native
AI_CHAT_TIMEOUT_MS=135000
```

Set `AI_ASR_SERVICE_URL` to this same RunPod root URL, and set
`AI_ASR_SERVICE_TOKEN` to the same Bearer secret, or leave both blank to use
the existing `AI_SERVICE_URL` and `AI_SERVICE_TOKEN` fallback. The backend's
transcription timeout remains `AI_SERVICE_TIMEOUT_MS` (default 780000 ms).
Keep these values server-side; never use `VITE_*` for credentials.

## Live verification

From a machine that can reach both the Pod proxy and the existing backend,
configure these variables in its process environment and run:

```sh
AI_SERVICE_URL=https://<POD_ID>-10000.proxy.runpod.net \
AI_SERVICE_TOKEN=<Pod service token> \
LIGHTBRIDGE_URL=<backend base URL> \
LIGHTBRIDGE_USER_TOKEN=<JWT for a user authorized to the test book> \
BOOK_ID=<indexed book ID> \
BOOK_SMOKE_QUESTION=<question answerable from that book> \
BOOK_SMOKE_EXPECTED=<distinctive phrase expected in the answer> \
BOOK_SMOKE_LANGUAGE=en \
node backend/scripts/smoke-natlas-live.cjs
```

The script makes genuine English, Yoruba, Hausa, and Igbo `/chat` requests,
checks model/language/reply fields and reports response times. It then creates
an authenticated LightBridge book conversation and requires a completed,
cited answer containing the expected book fact. A health response or mocked
test is not counted as successful inference. Review the four replies and book
citations manually before giving NAIC a live URL.
