"""Preparation control-flow tests. These do not constitute model inference."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.artifact import MODEL_FILE, MANIFEST_FILE, COMMIT_FILE, MODEL_ID, SOURCE_REVISION, LLAMA_CPP_REVISION, sha256
from app.main import create_app
from app.preparation import RenderChatEngine, check_smoke_report
from scripts.convert_model import reuse_prepared, prepare, atomic_json, check_resources, PreparationError


def committed(directory):
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / MODEL_FILE
    model.write_bytes(b"GGUFtest-fixture-not-real-weights")
    manifest = directory / MANIFEST_FILE
    atomic_json(manifest, dict(model_id=MODEL_ID, source_revision=SOURCE_REVISION,
        llama_cpp_revision=LLAMA_CPP_REVISION, quantization="Q4_K_M", filename=MODEL_FILE,
        sha256=sha256(model), size_bytes=model.stat().st_size))
    atomic_json(directory / COMMIT_FILE, {"manifest_sha256": sha256(manifest)})
    return sha256(manifest)


def test_restart_reuses_without_credentials_or_heavy_imports(tmp_path, monkeypatch):
    import scripts.convert_model as converter
    output, work = tmp_path / "model", tmp_path / "work"
    work.mkdir()
    committed(output)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    check = Mock(side_effect=AssertionError("Must not reprepare"))
    monkeypatch.setattr(converter, "check_resources", check)
    prepare(SimpleNamespace(work_dir=work, output_dir=output))
    check.assert_not_called()
    assert json.loads((work / "status.json").read_text())["stage"] == "reused"


def test_incomplete_and_corrupt_artifacts_are_never_reused(tmp_path):
    output = tmp_path / "model"
    committed(output)
    (output / MODEL_FILE).write_bytes(b"GGUFcorruption")
    with pytest.raises(ValueError, match="checksum"):
        reuse_prepared(output)
    (output / COMMIT_FILE).unlink()
    assert reuse_prepared(output) is False


def test_model_ready_requires_successful_matching_smoke_report(tmp_path):
    digest = committed(tmp_path)
    report = tmp_path / "smoke-report.jsonl"
    with pytest.raises(FileNotFoundError):
        check_smoke_report(tmp_path)
    report.write_text(json.dumps({"inference_passed": True, "model": MODEL_ID,
                                  "artifact_sha256": "0" * 64}) + "\n")
    with pytest.raises(ValueError):
        check_smoke_report(tmp_path)
    report.write_text(json.dumps({"inference_passed": True, "model": MODEL_ID,
                                  "artifact_sha256": digest}) + "\n")
    check_smoke_report(tmp_path)


@pytest.mark.asyncio
async def test_serve_never_prepares_missing_model(settings):
    settings.natlas_setup_mode = "serve"
    settings.natlas_artifact_dir = Path(settings.hf_home)
    engine = RenderChatEngine(settings)
    engine.run_job = AsyncMock(side_effect=AssertionError("Must not prepare in serve mode"))
    await engine.start()
    assert engine.state == "unavailable"
    engine.run_job.assert_not_called()


@pytest.mark.asyncio
async def test_failed_preparation_never_becomes_ready(settings):
    settings.natlas_setup_mode = "prepare"
    engine = RenderChatEngine(settings)
    engine.run_job = AsyncMock(side_effect=RuntimeError("private error must not be logged"))
    await engine.start()
    assert engine.state == "preparation_failed"


def test_preparation_liveness_is_separate_from_model_readiness(settings, headers):
    settings.natlas_backend = "gguf"
    settings.natlas_setup_mode = "prepare"
    engine = RenderChatEngine(settings)
    engine.state = "preparing"
    engine.start = AsyncMock()
    with TestClient(create_app(settings, chat_engine=engine, render_only=True)) as client:
        assert client.get("/live").json()["model_state"] == "preparing"
        assert client.get("/live").status_code == 200
        assert client.get("/health").status_code == 503
        assert client.post("/chat", json={}).status_code == 401
        request = {"language": "en", "messages": [{"role": "user", "content": "Hello"}]}
        assert client.post("/chat", headers=headers, json=request).status_code == 503
        assert client.post("/transcribe", headers={**headers, "Content-Type": "multipart/form-data"}).status_code == 404


@pytest.mark.asyncio
async def test_successful_job_requires_conversion_then_smoke_and_stays_unavailable_for_chat(settings, tmp_path):
    from app.chat import ChatRequest
    from app.errors import ServiceError
    settings.natlas_setup_mode = "prepare"
    settings.natlas_managed_artifact = True
    settings.natlas_artifact_dir = tmp_path / "model"
    engine = RenderChatEngine(settings)
    calls = []

    async def job(script, *args, stdout=None):
        calls.append(script)
        if script == "convert_model.py":
            committed(settings.natlas_artifact_dir)
        else:
            result = {"inference_passed": True, "model": MODEL_ID,
                      "artifact_sha256": settings.natlas_manifest_sha256}
            stdout.write((json.dumps(result) + "\n").encode())

    engine.run_job = job  # Unit-test double only; never used by runtime.
    await engine.start()
    assert calls == ["convert_model.py", "smoke_cpu.py"]
    assert engine.state == "prepared"
    with pytest.raises(ServiceError) as failure:
        await engine.generate(ChatRequest(language="en", messages=[{"role": "user", "content": "Hello"}]))
    assert failure.value.status == 503


def test_resource_checks_reject_small_conversion_disk_and_ram(tmp_path, monkeypatch):
    import scripts.convert_model as converter
    monkeypatch.setattr(converter.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024))
    with pytest.raises(PreparationError, match="insufficient_disk"):
        check_resources(tmp_path)
    monkeypatch.setattr(converter.shutil, "disk_usage", lambda _: SimpleNamespace(free=80 * 1024**3))
    monkeypatch.setattr(converter.os, "sysconf", lambda key: 4096 if key == "SC_PAGE_SIZE" else 1024, raising=False)
    with pytest.raises(PreparationError, match="insufficient_memory"):
        check_resources(tmp_path)
