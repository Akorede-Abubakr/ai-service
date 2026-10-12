"""Supervise a finite preparation job on the disk-owning Render instance."""
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from .artifact import acquire_artifact, COMMIT_FILE, MODEL_ID
from .chat import ChatEngine

logger = logging.getLogger("lightbridge.preparation")


def check_smoke_report(directory):
    record = json.loads((directory / COMMIT_FILE).read_text(encoding="utf-8"))
    result = json.loads((directory / "smoke-report.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    if (result.get("inference_passed") is not True or result.get("model") != MODEL_ID
            or result.get("artifact_sha256") != record["manifest_sha256"]):
        raise ValueError("Missing or mismatched genuine inference report")


class RenderChatEngine(ChatEngine):
    def __init__(self, settings):
        super().__init__(settings)
        self.process = None

    async def run_job(self, script, *arguments, stdout=None):
        environment = dict(os.environ)
        environment.update(
            NATLAS_BACKEND="gguf", NATLAS_ENABLED="true", NATLAS_MANAGED_ARTIFACT="true",
            NATLAS_ARTIFACT_DIR=str(self.settings.natlas_artifact_dir),
            HF_LOCAL_FILES_ONLY="true", HF_HUB_DISABLE_PROGRESS_BARS="1",
        )
        # Only the downloader needs HF_TOKEN; the smoke subprocess is offline.
        if script != "convert_model.py":
            environment.pop("HF_TOKEN", None)
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / script), *arguments,
            env=environment, stdout=stdout or asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
        try:
            code = await asyncio.wait_for(self.process.wait(), self.settings.natlas_prepare_timeout_seconds)
            if code:
                raise RuntimeError("Preparation subprocess failed")
        finally:
            if self.process.returncode is None:
                # Render is Linux. Stop converter grandchildren too on cancellation.
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), 10)
                except TimeoutError:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    await self.process.wait()
            self.process = None

    async def start(self):
        if self.settings.natlas_setup_mode == "serve":
            # No preparation/download fallback on the smaller inference instance.
            try:
                check_smoke_report(self.settings.natlas_artifact_dir)
            except Exception:
                self.state = "unavailable"
                logger.error("code=verified_preparation_required")
                return
            await super().start()
            return
        self.state = "preparing"
        try:
            await self.run_job("convert_model.py", "--work-dir", str(self.settings.natlas_prepare_work_dir),
                               "--output-dir", str(self.settings.natlas_artifact_dir))
            await asyncio.to_thread(acquire_artifact, self.settings)
            self.state = "verifying"
            report = self.settings.natlas_artifact_dir / "smoke-report.jsonl"
            with report.open("wb") as output:
                await self.run_job("smoke_cpu.py", stdout=output)
            # Child exit status alone is not enough: require its explicit final result.
            check_smoke_report(self.settings.natlas_artifact_dir)
            self.state = "prepared"
            logger.warning("code=preparation_verified mode_switch_required=serve")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state = "preparation_failed"
            logger.error("code=preparation_failed error_type=%s", type(exc).__name__)
