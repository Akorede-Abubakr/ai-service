"""Opt-in REAL inference. No mocks, no skipped success, no alternative model."""
import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.artifact import MODEL_ID, COMMIT_FILE
from app.chat import ChatEngine, ChatRequest
from app.config import Settings

PROMPTS = {
    "en": "In one short sentence, explain why reading is useful.",
    "yo": "Ní gbólóhùn kúkúrú kan, ṣàlàyé ìdí tí kíkà ìwé fi ṣe pàtàkì.",
    "ha": "A cikin gajeriyar jimla ɗaya, bayyana amfanin karatu.",
    "ig": "N'otu ahịrịokwu dị mkpirikpi, kọwaa uru ịgụ akwụkwọ bara.",
}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def remote(url, path, token=None, payload=None, timeout=135):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url + path, headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None)
    # Do not forward bearer tokens through redirects or environment proxies.
    with urllib.request.build_opener(NoRedirect(), urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


async def run(args):
    settings = Settings(natlas_backend="gguf", natlas_enabled=True)
    if settings.natlas_managed_artifact and not settings.natlas_manifest_sha256:
        record = json.loads((settings.natlas_artifact_dir / COMMIT_FILE).read_text(encoding="utf-8"))
        settings.natlas_manifest_sha256 = record["manifest_sha256"]
    if not settings.natlas_manifest_sha256:
        raise ValueError("Set the independently recorded NATLAS_MANIFEST_SHA256")
    engine = None
    if args.url:
        parsed = urllib.parse.urlsplit(args.url)
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
        if parsed.scheme != "https" and not (local and parsed.scheme == "http"):
            raise ValueError("Use HTTPS or localhost HTTP")
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Use a service root URL without credentials")
        args.url = args.url.rstrip("/")
        health = await asyncio.to_thread(remote, args.url, "/health")
        if (health.get("status"), health.get("model"), health.get("runtime"), health.get("artifact_sha256")) != (
                "ready", MODEL_ID, "gguf-cpu", settings.natlas_manifest_sha256):
            raise ValueError("Expected CPU model/artifact is not ready")
        try:
            await asyncio.to_thread(remote, args.url, "/chat", None,
                                    {"language": "en", "messages": [{"role": "user", "content": "Hello"}]})
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
        else:
            raise ValueError("Missing bearer token was accepted")
    else:
        engine = ChatEngine(settings)
        started = time.monotonic()
        await engine.start()
        if engine.state != "ready":
            await engine.close()
            raise RuntimeError("Actual CPU model failed to load")
        print(json.dumps({"load_seconds": round(time.monotonic() - started, 2)}))
    try:
        for language, prompt in PROMPTS.items():
            payload = {"language": language, "messages": [{"role": "user", "content": prompt}]}
            started = time.monotonic()
            if engine:
                reply = await engine.generate(ChatRequest(**payload))
            else:
                response = await asyncio.to_thread(remote, args.url, "/chat",
                    settings.ai_service_token.get_secret_value(), payload, settings.natlas_timeout_seconds + 15)
                if response.get("language") != language or response.get("model") != MODEL_ID:
                    raise ValueError("Unexpected model/language")
                reply = response.get("reply")
            if not isinstance(reply, str) or not reply.strip() or not any(c.isalpha() for c in reply) or len(reply) > 8000:
                raise ValueError("No substantive generated text")
            # These are fixed, nonprivate test prompts. Print actual text for human review.
            print(json.dumps({"language": language, "reply": reply,
                              "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=True))
        print(json.dumps({"inference_passed": True, "model": MODEL_ID,
                          "artifact_sha256": settings.natlas_manifest_sha256,
                          "language_quality": "requires human review of the generated replies"}))
    finally:
        if engine:
            await engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Actual running CPU service; omit to load the real local GGUF")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        print(json.dumps({"inference_passed": False, "error_type": type(exc).__name__}), file=sys.stderr)
        sys.exit(1)
