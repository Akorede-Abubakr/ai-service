"""Start the existing app factory with Windows-compatible subprocess support."""
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run the existing N-ATLaS service locally (one worker, no reload).")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["NATLAS_ENABLED"] = "false"
    from app.config import Settings

    settings = Settings()
    binary = shutil.which(settings.ffmpeg_binary)
    if not binary:
        try:
            # Optional development fallback, already in requirements-dev.txt.
            # Production continues using the FFmpeg installed in the Docker image.
            import imageio_ffmpeg

            binary = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            parser.error("Install FFmpeg or set FFMPEG_BINARY to its full path. For local tests, install requirements-dev.txt.")
    os.environ["FFMPEG_BINARY"] = binary
    import uvicorn

    # --reload/multiple workers select SelectorEventLoop on Windows, which
    # cannot run the service's asyncio FFmpeg subprocess. Keep the normal
    # single-worker ProactorEventLoop; no changes to app.main are needed.
    uvicorn.run("app.main:create_app", factory=True, host="127.0.0.1", port=args.port,
                workers=1, reload=False, access_log=False)


if __name__ == "__main__":
    main()
