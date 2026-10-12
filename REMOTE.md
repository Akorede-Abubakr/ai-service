# Official N-ATLaS: Render Free API + separate CPU inference

**2026-10-12: real N-ATLaS inference did not succeed. Nothing has been deployed.** No accessible official hosted inference endpoint was found. Render Free remains the HTTP API host, never the model host.

## Actual Hugging Face verification

Using the existing HF_TOKEN, without printing it or downloading weights:

| Check | Observed result |
| --- | --- |
| Official Hub model API with expand=inferenceProviderMapping | HTTP 200, empty mapping {} |
| Official gated config.json | HTTP 200: this token has model-file access |
| https://router.huggingface.co/v1/models | HTTP 200; NCAIR1/N-ATLaS absent |
| Actual POST https://router.huggingface.co/v1/chat/completions | HTTP 403: token lacks Inference Providers permission |
| Generated response | None; inference_passed=false |

The [official model page](https://huggingface.co/NCAIR1/N-ATLaS) independently states that no Inference Provider deploys this model. The request uses the [documented HF chat endpoint](https://huggingface.co/docs/inference-providers/tasks/chat-completion) with Bearer HF_TOKEN, model NCAIR1/N-ATLaS, a real user prompt, max_tokens=32 and stream=false. [Provider discovery documentation](https://huggingface.co/docs/inference-providers/hub-api).

These are two separate blockers: the current token cannot call Inference Providers, and no provider is listed for this model. Updating token permissions alone does not create a hosted N-ATLaS instance. Gated download permission is not hosted inference access. A paid dedicated endpoint would require provisioning; none was created or assumed to exist. No reference-project API, third-party credentials, substitute model or mock completion is used.

Reproduce locally (from ai-service; install requirements-remote.txt first):

```sh
python scripts/probe_official_inference.py
# Equivalent:
python scripts/smoke_remote.py --probe
```

These make a real HF generation request and exit nonzero on failure. Successful model-file access or health is never counted as inference. The probe may be rerun if official provider availability changes; it is not an automatically selected runtime provider.

## CPU requirements: estimates, not measured benchmarks

Preserve the existing official-model conversion and llama.cpp implementation. Source is pinned to NCAIR1/N-ATLaS revision e294476928aca9030e924ca27bb8e085e8581273, converted locally to Q4_K_M with the existing pinned converter. This is a quantization of the official model, not a substituted checkpoint. Do not use an unverified community GGUF.

| Phase | Practical resource budget |
| --- | --- |
| Serving, lowest sensible trial | 2 modern x86-64 CPU cores, 8 GiB RAM, one request at a time, 2048 context; tight and unverified |
| Serving, recommended configuration | 4 CPU cores, 16 GB RAM, 2048 context, 128 output tokens |
| Preparation with the unchanged pipeline | 4 CPU cores, 32 GB RAM, 80 GB persistent model/work disk; allow extra space for Docker image/build cache |
| Final artifact alone | Approximately 5 GB Q4_K_M plus manifest/reports; provision at least 10 GB if transferring only a verified final artifact to a separate server |
| Current same-disk workflow | Keep the full 80 GB disk through prepare and serve; it retains source/intermediates |

Reasoning: the [official 8B architecture](https://huggingface.co/NCAIR1/N-ATLaS) needs about 16 GB for BF16 weights alone. Q4_K_M is roughly 5 GB, plus runtime buffers, OS and about 256 MiB FP16 KV cache at 2048 tokens (32 layers, 8 KV heads, head dimension 128). Approximately 6-8 GiB working memory is a planning estimate; 8 GiB is a trial floor, not a guaranteed minimum. Memory mapping does not eliminate resident weight memory. Use 16 GB for a practical deployment; no latency promise is made. [llama.cpp quantization](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize).

The unchanged preparation job enforces at least 28 GiB available memory and 40 GiB free disk; 32 GB/80 GB is its operational budget. These checks are lower bounds, not peak guarantees. Do not attempt preparation on an 8/16 GB host. Conversion and real CPU generation remain unverified here.

## Exact CPU host configuration

This is a **separate paid or already-owned Linux machine**, never the Render Free API instance. Keep the existing Dockerfile, requirements, scripts/serve.py, preparation and CPU inference code.

If using a second Render service for inference: Docker, root ai-service for a parent repo or . for this standalone repo, Dockerfile ./Dockerfile, context ., default image command, one instance, Auto-Deploy Off. Preparation uses 4c-32g, disk 80 GB at /data, health /live. After actual preparation smoke succeeds, switch NATLAS_SETUP_MODE=serve, health /health, then use 4c-16g and retain the same disk. A Linux VPS can run the same image and /data mount; expose it through HTTPS and keep its Bearer authentication.

Use the existing render.env.example on this CPU host:

```dotenv
AI_SERVICE_TOKEN=<unique CPU-host secret, at least 32 characters>
HF_TOKEN=<your token with access to the official gated model>
HF_HOME=/data/huggingface
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_PROGRESS_BARS=1
NATLAS_SETUP_MODE=prepare
NATLAS_ENABLED=true
NATLAS_BACKEND=gguf
NATLAS_MANAGED_ARTIFACT=true
NATLAS_ARTIFACT_DIR=/data/natlas/model
NATLAS_PREPARE_WORK_DIR=/data/natlas/work
NATLAS_PREPARE_TIMEOUT_SECONDS=21600
HF_LOCAL_FILES_ONLY=true
NATLAS_CONTEXT_SIZE=2048
NATLAS_MAX_INPUT_TOKENS=1920
NATLAS_MAX_NEW_TOKENS=128
NATLAS_THREADS=4
NATLAS_BATCH_SIZE=128
NATLAS_TIMEOUT_SECONDS=120
REQUEST_TIMEOUT_SECONDS=30
PORT=10000
```

After preparation, set NATLAS_SETUP_MODE=serve. Read the actual manifest hash from prepared.json/the smoke report and verify the real CPU endpoint using scripts/smoke_cpu.py --url https://YOUR-CPU-HOST (with the CPU token, NATLAS_MANAGED_ARTIFACT=false and NATLAS_MANIFEST_SHA256 set on the smoke client). Review all four generated replies. Obtain this evidence before configuring/deploying the free gateway. No CPU host URL or artifact hash is currently available.

## Exact Render Free API configuration

| Render setting | Value |
| --- | --- |
| Service type / runtime | Web Service / Docker |
| Plan | Free, 512 MB |
| Root | ai-service in a parent repo; . for the standalone ai-service repo |
| Dockerfile / context | ./Dockerfile.remote / . |
| Docker command | Leave blank: python scripts/serve_remote.py |
| Workers / instances | 1 / 1 |
| Disk | None |
| Health path | /live (process liveness only) |
| Auto-Deploy | Off |

[Render compute specifications](https://render.com/docs/compute-plans). Free services [idle after 15 minutes and have no persistent disk or shell](https://render.com/docs/free). No weights or model libraries are installed in this image.

Use render-free.env.example, replacing placeholders only with values from your verified CPU host:

```dotenv
AI_SERVICE_TOKEN=<gateway secret matching Express, at least 32 characters>
NATLAS_BACKEND=remote
NATLAS_ENABLED=true
NATLAS_REMOTE_URL=https://YOUR-VERIFIED-CPU-HOST
NATLAS_REMOTE_TOKEN=<AI_SERVICE_TOKEN from that CPU host>
NATLAS_MANIFEST_SHA256=<64-character SHA256 from its verified manifest>
NATLAS_MAX_NEW_TOKENS=128
NATLAS_TIMEOUT_SECONDS=130
REQUEST_TIMEOUT_SECONDS=30
HF_LOCAL_FILES_ONLY=true
PORT=10000
```

The CPU host controls token/context budgets; NATLAS_MAX_NEW_TOKENS on the gateway does not override that host. HF_TOKEN is unnecessary on Render Free: it belongs on the preparation host or local official-provider probe. Keep the two service tokens distinct. No existing .env secret was replaced.

No usable endpoint is invented or prefilled. With URL/key/hash absent, /chat returns 503 identifying remote_endpoint_missing, remote_token_missing or remote_artifact_pin_missing. The runtime sends only this repository's native protocol: GET /health/chat must identify ready gguf-cpu, official model and pinned manifest; POST /chat receives the unchanged language/messages request. Those routes exist in app/main.py. A remote gateway cannot be used as its own inference host because the runtime check requires gguf-cpu.

Model/hash verification trusts your HTTPS CPU service and its existing artifact validation; it is not independent remote attestation. Only a valid actual reply sets gateway readiness. Errors, timeouts and oversized/malformed replies never become successful inference. No retries or fallback providers. Local model loading is disabled regardless of stale CPU variables on the gateway.

## API and verification

FastAPI and the existing Bearer guard, /chat schema and en/yo/ha/ig language support are preserved. /health and /health/chat return 503 until actual generation succeeds and reset on restart or failure. /live is process liveness only. /transcribe remains authenticated but returns 503 remote_speech_unavailable; existing local speech code remains separately deployable.

After CPU-host verification, configure the remote values locally and run:

```sh
python scripts/smoke_remote.py
```

This exercises real FastAPI routes and real HTTP inference: four language replies plus a history follow-up. It emits inference_passed=true only after all five real replies and readiness pass. Review language quality. Missing dependencies exit 1. After any eventually authorized gateway deployment, run the same test locally with --url https://YOUR-GATEWAY.onrender.com before connecting Express.

Express uses AI_SERVICE_URL=https://YOUR-GATEWAY.onrender.com, AI_SERVICE_TOKEN=<gateway secret>, AI_CHAT_API_FORMAT=native, AI_CHAT_TIMEOUT_MS=135000 and AI_MAX_CONCURRENT_REQUESTS=1. The gateway timeout is 130 seconds and CPU generation 120 seconds. Free cold-start delay is additional; wake the service before a timed smoke run.

Validation: 80 Python tests passed, including existing CPU/speech tests. The real remote smoke run in the lightweight environment exited 1 with 503 remote_endpoint_missing and inference_passed=false. These results verify safe failure, not inference. Docker build and Render/Linux memory measurements were not performed.

**Current outcome: no generated N-ATLaS response, no verified CPU artifact/host, no deployment.** The HTTP adapter and fail-closed behavior are implementation readiness only, not successful inference.
