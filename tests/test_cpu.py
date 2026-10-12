"""Unit checks only. scripts/smoke_cpu.py is the separate real inference gate."""
import asyncio
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from fastapi.testclient import TestClient

from app.artifact import MODEL_ID, SOURCE_REVISION, LLAMA_CPP_REVISION, MODEL_FILE, MANIFEST_FILE, sha256, acquire_artifact
from app.chat import ChatEngine, ChatRequest
from app.cpu import CpuModel
from app.errors import ServiceError
from app.main import create_app


def artifact(settings):
    settings.natlas_artifact_dir = Path(settings.hf_home)
    model = settings.natlas_artifact_dir / MODEL_FILE
    model.write_bytes(b"GGUFtest-only-not-a-real-model")
    manifest = settings.natlas_artifact_dir / MANIFEST_FILE
    data = dict(model_id=MODEL_ID, source_revision=SOURCE_REVISION,
                llama_cpp_revision=LLAMA_CPP_REVISION, quantization="Q4_K_M",
                filename=MODEL_FILE, sha256=sha256(model), size_bytes=model.stat().st_size)
    manifest.write_text(json.dumps(data))
    settings.natlas_manifest_sha256 = sha256(manifest)
    return model, manifest, data


def test_artifact_rejects_corruption_and_different_model(settings):
    model, manifest, data = artifact(settings)
    assert acquire_artifact(settings) == model
    model.write_bytes(b"GGUFcorrupt")
    with pytest.raises(ValueError, match="checksum"):
        acquire_artifact(settings)
    data["model_id"] = "another/model"
    manifest.write_text(json.dumps(data))
    settings.natlas_manifest_sha256 = sha256(manifest)
    with pytest.raises(ValueError, match="provenance"):
        acquire_artifact(settings)


def test_artifact_rejects_untrusted_manifest_and_never_fetches_offline(settings):
    _, manifest, _ = artifact(settings)
    settings.natlas_manifest_sha256 = "0" * 64
    with pytest.raises(ValueError, match="manifest checksum"):
        acquire_artifact(settings)
    manifest.unlink()
    settings.hf_local_files_only = True
    with pytest.raises(ValueError, match="Missing local artifact"):
        acquire_artifact(settings)


def test_render_requires_real_generation_verified_manifest(settings):
    _, manifest, data = artifact(settings)
    settings.natlas_require_verified_artifact = True
    with pytest.raises(ValueError, match="genuine N-ATLaS generation smoke"):
        acquire_artifact(settings)
    data = __import__("json").loads(manifest.read_text())
    data["generation_verified"] = True
    manifest.write_text(__import__("json").dumps(data))
    settings.natlas_manifest_sha256 = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    assert acquire_artifact(settings).is_file()


def test_cpu_generation_preserves_language_budget_and_cancellation(settings, monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(StoppingCriteriaList=list))
    cpu = CpuModel.__new__(CpuModel)
    cpu.settings = settings
    cpu.template = ImmutableSandboxedEnvironment().from_string(
        "{{ bos_token }}{{ date_string }}{% for m in messages %}{{ m.content }}{% endfor %}")
    cpu.llm = Mock()
    cpu.llm.tokenize.return_value = [1, 2, 3]
    cpu.llm.create_completion.return_value = {"choices": [{"text": "Unit test text"}]}
    request = ChatRequest(language="yo", messages=[{"role": "user", "content": "Ẹ káàárọ̀"}])
    stop = threading.Event()
    assert cpu.generate(request, stop) == "Unit test text"
    encoded = cpu.llm.tokenize.call_args.args[0].decode()
    assert "Yoruba" in encoded and "Ẹ káàárọ̀" in encoded
    assert cpu.llm.tokenize.call_args.kwargs == {"add_bos": False, "special": True}
    options = cpu.llm.create_completion.call_args.kwargs
    assert options["prompt"] == [1, 2, 3] and options["max_tokens"] == settings.natlas_max_new_tokens
    assert not options["stopping_criteria"][0]()
    stop.set()
    assert options["stopping_criteria"][0]()
    with pytest.raises(ServiceError) as failure:
        cpu.generate(request, stop)
    assert failure.value.code == "inference_timeout"
    stop.clear()
    settings.natlas_max_input_tokens = 2
    with pytest.raises(ServiceError) as failure:
        cpu.generate(request, stop)
    assert failure.value.code == "context_too_long"


def test_cpu_load_uses_no_gpu_and_requires_template(settings, monkeypatch):
    import sys
    from types import SimpleNamespace
    import app.cpu
    settings.natlas_max_input_tokens = 1000
    settings.natlas_max_new_tokens = 128
    llm = Mock(metadata={"general.architecture": "llama", "general.file_type": "15",
                         "tokenizer.chat_template": "{{ bos_token }}"})
    loader = Mock(return_value=llm)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=loader))
    monkeypatch.setattr(app.cpu, "acquire_artifact", lambda _: "test-only.gguf")
    CpuModel(settings)
    assert loader.call_args.kwargs["n_gpu_layers"] == 0
    assert loader.call_args.kwargs["offload_kqv"] is False
    assert loader.call_args.kwargs["n_ctx"] == 2048
    llm.metadata.pop("tokenizer.chat_template")
    with pytest.raises(ValueError, match="template missing"):
        CpuModel(settings)


def test_cpu_binds_before_loading_and_does_not_claim_ready(settings, headers):
    settings.natlas_backend = "gguf"

    class LoadingEngine:
        state = "loading"
        async def start(self):
            await asyncio.Event().wait()
        async def close(self):
            pass
        async def generate(self, request):
            raise ServiceError(503, "chat_unavailable", "Loading")

    with TestClient(create_app(settings, chat_engine=LoadingEngine())) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["runtime"] == "gguf-cpu"
        assert client.post("/chat", json={}).status_code == 401
        assert client.post("/chat", headers=headers, json={"language": "en", "messages": [
            {"role": "user", "content": "Hello"}]}).status_code == 503


@pytest.mark.asyncio
async def test_cpu_shutdown_releases_model_after_worker_stops(settings):
    settings.natlas_backend = "gguf"
    engine = ChatEngine(settings)
    engine.executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    closed = threading.Event()
    engine.model = Mock()
    engine.model.llm.close.side_effect = closed.set
    engine.active = asyncio.get_running_loop().run_in_executor(engine.executor, release.wait, 3)
    try:
        await engine.close()
        await engine.close()  # Shutdown is idempotent.
        assert not closed.is_set()
        release.set()
        assert await asyncio.to_thread(closed.wait, 3)
    finally:
        release.set()
