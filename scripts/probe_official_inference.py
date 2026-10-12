"""Real HF discovery and generation probe. No downloads, fallback or mocks."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from app.config import Settings

MODEL = "NCAIR1/N-ATLaS"


def run():
    settings = Settings()
    token = settings.hf_token.get_secret_value() if settings.hf_token else ""
    headers = {"Authorization": "Bearer " + token} if token else {}
    with httpx.Client(timeout=60, follow_redirects=False, trust_env=False) as client:
        response = client.get("https://huggingface.co/api/models/" + MODEL,
                              params={"expand": "inferenceProviderMapping"}, headers=headers)
        mapping = response.json().get("inferenceProviderMapping") if response.status_code == 200 else None
        print(json.dumps({"check": "provider_mapping", "status": response.status_code, "mapping": mapping}), flush=True)
        # Metadata only, never weight files. Validate the user's gated-repo access.
        response = client.get("https://huggingface.co/" + MODEL + "/resolve/main/config.json", headers=headers)
        print(json.dumps({"check": "official_config_access", "status": response.status_code,
                          "redirect": response.is_redirect}), flush=True)
        response = client.get("https://router.huggingface.co/v1/models", headers=headers)
        entries = response.json().get("data", []) if response.status_code == 200 else []
        print(json.dumps({"check": "router_catalog", "status": response.status_code,
                          "official_model_listed": any(x.get("id") == MODEL for x in entries)}), flush=True)
        response = client.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json={
            "model": MODEL, "messages": [{"role": "user", "content": "Say hello in English in one short sentence."}],
            "max_tokens": 32, "stream": False,
        })
        data = response.json()
        # Only this explicit probe prints provider diagnostics; redact token first.
        diagnostic = json.dumps(data, ensure_ascii=True)
        if token:
            diagnostic = diagnostic.replace(token, "[REDACTED]")
        print(json.dumps({"check": "actual_generation", "authenticated": bool(token),
                          "status": response.status_code, "response": json.loads(diagnostic)}), flush=True)
        reply = data.get("choices", [{}])[0].get("message", {}).get("content")
        passed = response.status_code == 200 and isinstance(reply, str) and bool(reply.strip()) and data.get("model") == MODEL
        print(json.dumps({"inference_passed": passed, "model": MODEL}), flush=True)
        return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        print(json.dumps({"inference_passed": False, "error_type": type(exc).__name__}), flush=True)
        sys.exit(1)
