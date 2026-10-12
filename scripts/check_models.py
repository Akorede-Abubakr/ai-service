"""Verify model access; optionally download/load weights before serving requests."""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.errors import model_error
from app.registry import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-weights", action="store_true",
                        help="Download and load real models outside the API request deadline.")
    parser.add_argument("--language", choices=list(MODELS), help="Check only this enabled language.")
    args = parser.parse_args()
    if args.download_weights:
        # Apply before importing the Hub/Transformers libraries. Slow initial
        # downloads should not compete with the API's cold-load deadline.
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        logging.disable(logging.CRITICAL)  # Never print signed download URLs.
    from huggingface_hub import hf_hub_download

    settings = Settings()
    if args.language and args.language not in settings.languages:
        parser.error("The requested language is not enabled in service settings.")
    failed = False
    for language in (args.language,) if args.language else settings.languages:
        spec = MODELS[language]
        try:
            for filename in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "generation_config.json"):
                hf_hub_download(
                    spec.model_id, filename, revision=spec.revision,
                    token=settings.hf_token.get_secret_value() if settings.hf_token else False,
                    cache_dir=str(Path(settings.hf_home) / "hub"),
                    local_files_only=settings.hf_local_files_only,
                )
            print(f"OK {language}: {spec.model_id} @ {spec.revision}", flush=True)
            if args.download_weights:
                from app.inference import load_pipeline

                print(f"Loading real weights for {language}; the first download may take several minutes.", flush=True)
                model = load_pipeline(spec, settings)
                del model
                print(f"READY {language}: weights downloaded and model loaded successfully", flush=True)
        except Exception as exc:
            print(f"FAIL {language}: {model_error(exc).code}")
            # Exception class names aid diagnosis without exposing messages,
            # signed Hub URLs, request headers, or the user's token.
            causes = []
            current = exc
            seen = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                causes.append(type(current).__name__)
                current = current.__cause__ or current.__context__
            print("Failure types: " + " -> ".join(causes), flush=True)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
