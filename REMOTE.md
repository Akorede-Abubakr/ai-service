# Free-tier remote mode

**Blocked, not deployed:** the [requested Space](https://huggingface.co/spaces/zylvex/natlas-zylcode-bridge) exposes an authenticated API, but the owner-issued key is missing. On 2026-10-12, actual requests returned 200 for `/healthz` and 401 for both `/v1/models` and `/v1/chat/completions`. The public health reports no loaded ASR models. No successful N-ATLaS inference is claimed.

Use `Dockerfile.remote` and `requirements-remote.txt` for a single HTTP worker with no model libraries, downloads, preparation or persistent disk. The existing Dockerfile and CPU implementation remain the optional paid mode. `create_remote_app` forces remote mode; the generic factory supports `NATLAS_BACKEND=remote`.

Supply `AI_SERVICE_TOKEN` for your existing incoming API auth and **`NATLAS_REMOTE_API_KEY`** for the Space owner's private `NATLAS_API_KEY`. Neither your Hugging Face token nor your incoming service token substitutes for that key. Keep credentials in the environment or `.env`; see `render-free.env.example`.

The adapter uses the exact [inspected source contract](https://huggingface.co/spaces/zylvex/natlas-zylcode-bridge/blob/a2566d91fede85bf769daacdfb55d1678eafba2a/app.py): Bearer-authenticated JSON `POST /v1/chat/completions`, fixed `NCAIR1/N-ATLaS`, system prompt plus history, `max_tokens`, `temperature`, `repetition_penalty`, `stream=false`. It verifies health model identity and completion structure/usage. This trusts the provider; it cannot independently attest remote weights.

From this repository directory, with the remote dependencies installed:

```sh
python scripts/smoke_remote.py --probe
python scripts/smoke_remote.py
```

The public probe always exits nonzero and never counts as inference verification. The second command exercises the real FastAPI routes and actual upstream HTTPS. It checks incoming authentication, four actual generated language replies and a follow-up. Missing keys/errors exit 1 with `inference_passed=false`. Only all successful calls produce `inference_passed=true`. Review the printed replies before deploying; unit test doubles are not inference evidence.

`/chat`, authentication and `en/yo/ha/ig` are preserved. `/health` and `/health/chat` stay 503 until an actual completion succeeds, reset on restart/failure, and report the missing dependency. `/live` only indicates process liveness. `/transcribe` retains authentication but returns 503 `remote_speech_unavailable`; it never loads a local model. Local speech remains available in the original separately resourced deployment.

**Only after a successful reviewed smoke test:** Render Docker Web Service, Free 512 MB, Dockerfile `./Dockerfile.remote`, context `.`, one worker, no disk, auto-deploy off, environment from `render-free.env.example`. For this standalone Git repository use root `.`; for a parent repository use root `ai-service`. The image command binds `0.0.0.0:$PORT` (default 10000). `/live` is the process health check because readiness needs a first real inference after every cold start. A green deployment is not proof of inference. Run `python scripts/smoke_remote.py --url https://YOUR-SERVICE.onrender.com` from your local machine before connecting Express.

No deployment is performed by these scripts. Docker/Linux peak memory and authorized upstream inference remain unverified. The [full workspace guide](../docs/render-free.md) records investigation, resource constraints and manual rollout details. Render Free has [512 MB](https://render.com/docs/compute-plans), [idle spin-down, ephemeral storage and no shell](https://render.com/docs/free).
