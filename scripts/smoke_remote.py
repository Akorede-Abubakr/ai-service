"""Opt-in real remote inference through FastAPI. No fixtures or mock fallback.

Default: exercise the local app in-process with actual HTTPS to the Space.
--url: exercise an already running instance. --probe: public access diagnostic
only, always exits nonzero because it cannot establish authenticated inference.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.config import Settings
from app.main import create_app
from app.remote import SPACE_URL

PROMPTS = {
    "en": "In one short sentence, explain why reading is useful.",
    "yo": "Ní gbólóhùn kúkúrú kan, ṣàlàyé ìdí tí kíkà ìwé fi ṣe pàtàkì.",
    "ha": "A cikin gajeriyar jimla ɗaya, bayyana amfanin karatu.",
    "ig": "N'otu ahịrịokwu dị mkpirikpi, kọwaa uru ịgụ akwụkwọ bara.",
}


def emit(**values):
    print(json.dumps(values, ensure_ascii=True), flush=True)


def service_url(value):
    parsed = urlsplit(value)
    local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if (not parsed.hostname or parsed.scheme != "https" and not (local and parsed.scheme == "http")
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("Use an HTTPS service root URL, or localhost HTTP, without credentials")
    return value.rstrip("/")


async def check(client, settings):
    headers = {"Authorization": "Bearer " + settings.ai_service_token.get_secret_value()}
    first = {"language": "en", "messages": [{"role": "user", "content": PROMPTS["en"]}]}
    unauthenticated = await client.post("/chat", json=first)
    if unauthenticated.status_code != 401:
        raise ValueError("Service bearer authentication check failed")
    health = await client.get("/health/chat")
    if health.json().get("runtime") != "remote":
        raise ValueError("Expected remote runtime")
    for language, prompt in PROMPTS.items():
        started = time.monotonic()
        response = await client.post("/chat", headers=headers, json={
            "language": language, "messages": [{"role": "user", "content": prompt}],
        })
        if response.status_code != 200:
            # Only output the local service's safe error code, never raw provider errors.
            emit(language=language, status=response.status_code,
                 code=response.json().get("error", {}).get("code", "inference_failed"))
            raise ValueError("Real remote inference failed")
        data = response.json()
        if (data.get("model") != settings.natlas_model_id or data.get("language") != language
                or not isinstance(data.get("reply"), str) or not data["reply"].strip()):
            raise ValueError("Invalid inference response")
        emit(language=language, reply=data["reply"], seconds=round(time.monotonic() - started, 2))
    # Exercise history forwarding with an actual follow-up, not just isolated turns.
    followup = await client.post("/chat", headers=headers, json={
        "language": "ig", "messages": [
            {"role": "user", "content": PROMPTS["ig"]},
            {"role": "assistant", "content": data["reply"]},
            {"role": "user", "content": "Biko nye otu ihe atụ."},
        ],
    })
    followup_data = followup.json()
    if (followup.status_code != 200 or followup_data.get("model") != settings.natlas_model_id
            or followup_data.get("language") != "ig"
            or not isinstance(followup_data.get("reply"), str) or not followup_data["reply"].strip()):
        raise ValueError("Real follow-up inference failed")
    emit(language="ig", followup=True, reply=followup_data["reply"])
    health = await client.get("/health/chat")
    if health.status_code != 200 or health.json().get("inference_verified") is not True:
        raise ValueError("Remote readiness did not reflect successful inference")
    emit(inference_passed=True, model=settings.natlas_model_id, provider=SPACE_URL,
         languages=list(PROMPTS), requests=5,
         note="Actual responses above require human language-quality review.")


async def run(args):
    if args.probe:
        async with httpx.AsyncClient(base_url=SPACE_URL, timeout=30, follow_redirects=False, trust_env=False) as client:
            health = await client.get("/healthz")
            emit(probe="health", status=health.status_code,
                 model=health.json().get("model"), asr_models_loaded=health.json().get("asr_models_loaded"))
            models = await client.get("/v1/models")
            emit(probe="unauthenticated_models", status=models.status_code)
            chat = await client.post("/v1/chat/completions", json={
                "model": "NCAIR1/N-ATLaS", "messages": [{"role": "user", "content": "Say hello in English."}],
                "max_tokens": 16, "stream": False,
            })
            emit(probe="unauthenticated_chat", status=chat.status_code)
        raise ValueError("Public access probe is not authenticated inference verification")
    settings = Settings(natlas_backend="remote", natlas_enabled=True, natlas_max_new_tokens=128)
    if args.url:
        async with httpx.AsyncClient(base_url=service_url(args.url), timeout=135,
                                     follow_redirects=False, trust_env=False) as client:
            await check(client, settings)
    else:
        app = create_app(settings=settings)
        async with app.router.lifespan_context(app):
            # Only the local HTTP transport is in-process. The engine contacts the
            # actual Space over HTTPS; no test engine/transport is injected there.
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://local-app") as client:
                await check(client, settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--url", help="Existing service URL; this script does not deploy")
    group.add_argument("--probe", action="store_true", help="Diagnose public Space access, never report inference success")
    try:
        asyncio.run(run(parser.parse_args()))
    except Exception as exc:
        # No exception messages: third-party failures may contain secrets/URLs.
        emit(inference_passed=False, error_type=type(exc).__name__,
             action="Check NATLAS_REMOTE_API_KEY and the Space's availability. No deployment is authorized by this result.")
        sys.exit(1)
