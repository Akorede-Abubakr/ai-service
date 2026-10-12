"""Integrity/provenance checks for our own conversion of the official model."""
import hashlib
import json
from pathlib import Path

MODEL_ID = "NCAIR1/N-ATLaS"
SOURCE_REVISION = "e294476928aca9030e924ca27bb8e085e8581273"
LLAMA_CPP_REVISION = "4227c9be4268ac844921b90f31595f81236bd317"
MODEL_FILE = "natlas-q4_k_m.gguf"
MANIFEST_FILE = "natlas-manifest.json"
COMMIT_FILE = "prepared.json"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_manifest(path, expected_digest):
    if not expected_digest or sha256(path) != expected_digest:
        raise ValueError("Artifact manifest checksum mismatch or missing")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"model_id": MODEL_ID, "source_revision": SOURCE_REVISION,
                "llama_cpp_revision": LLAMA_CPP_REVISION,
                "quantization": "Q4_K_M", "filename": MODEL_FILE}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Artifact provenance mismatch")
    return manifest


def acquire_artifact(settings):
    directory = settings.natlas_artifact_dir
    directory.mkdir(parents=True, exist_ok=True)

    def obtain(filename):
        path = directory / filename
        if not path.is_file():
            if settings.hf_local_files_only or not settings.natlas_gguf_repo or not settings.natlas_gguf_revision:
                raise ValueError("Missing local artifact or pinned artifact repository")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=settings.natlas_gguf_repo, revision=settings.natlas_gguf_revision,
                filename=filename, local_dir=directory,
                token=settings.hf_token.get_secret_value() if settings.hf_token else False,
            )
        return path

    if settings.natlas_managed_artifact:
        # This record is published last by our pinned conversion job on this disk.
        # Disk integrity is the trust boundary; never fetch/adopt a remote marker.
        record = json.loads((directory / COMMIT_FILE).read_text(encoding="utf-8"))
        digest = record["manifest_sha256"]
        if settings.natlas_manifest_sha256 and settings.natlas_manifest_sha256 != digest:
            raise ValueError("Pinned manifest differs from prepared model")
        manifest = validate_manifest(directory / MANIFEST_FILE, digest)
        model_path = directory / MODEL_FILE
        settings.natlas_manifest_sha256 = digest
    else:
        manifest = validate_manifest(obtain(MANIFEST_FILE), settings.natlas_manifest_sha256)
        model_path = obtain(MODEL_FILE)
    if getattr(settings, "natlas_require_verified_artifact", False) and manifest.get("generation_verified") is not True:
        raise ValueError("Artifact has not passed genuine N-ATLaS generation smoke testing")
    if model_path.stat().st_size != manifest["size_bytes"] or sha256(model_path) != manifest["sha256"]:
        raise ValueError("GGUF checksum mismatch; replace the invalid artifact")
    with model_path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError("Invalid GGUF file")
    return model_path
