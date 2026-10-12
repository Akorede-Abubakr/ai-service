"""Explicit opt-in test against the real local API and official checkpoint."""
import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.registry import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--language", choices=["yo", "en-NG"], required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    settings = Settings()
    mime = {".wav": "audio/wav", ".webm": "audio/webm", ".ogg": "audio/ogg"}.get(args.audio.suffix.lower())
    if mime is None:
        parser.error("Use a .wav, .webm or .ogg recording")
    with args.audio.open("rb") as recording, httpx.Client(
        timeout=settings.model_load_timeout_seconds + settings.inference_timeout_seconds + 60
    ) as client:
        response = client.post(args.url.rstrip("/") + "/transcribe",
            headers={"Authorization": "Bearer " + settings.ai_service_token.get_secret_value()},
            data={"language": args.language}, files={"audio": (args.audio.name, recording, mime)})
    result = response.json()
    if response.status_code != 200:
        print(f"FAIL HTTP {response.status_code}: {result.get('error', {}).get('code', 'unknown')}")
        return 1
    assert result["language"] == args.language
    assert result["model"] == MODELS[args.language].model_id
    assert result["model_revision"] == MODELS[args.language].revision
    assert isinstance(result["transcript"], str) and result["transcript"].strip()
    print(f"PASS {result['model']} ({result['processing_time_ms']} ms)")
    print(result["transcript"])
    print("Compare with the recording manually for language accuracy, spelling and Yoruba diacritics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
