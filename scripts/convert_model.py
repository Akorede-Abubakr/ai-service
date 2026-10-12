"""Convert exactly the pinned official N-ATLaS snapshot; no substitute model."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.artifact import MODEL_ID, SOURCE_REVISION, LLAMA_CPP_REVISION, MODEL_FILE, MANIFEST_FILE, COMMIT_FILE, sha256, acquire_artifact


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def stage(work, name, **details):
    atomic_json(work / "status.json", {"stage": name, **details})
    print(json.dumps({"stage": name, **details}), flush=True)


class PreparationError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def reuse_prepared(output):
    if not (output / COMMIT_FILE).exists():
        return False
    acquire_artifact(SimpleNamespace(natlas_artifact_dir=output, natlas_managed_artifact=True,
                                    natlas_manifest_sha256="", natlas_require_verified_artifact=False))
    return True


def check_resources(work):
    if shutil.disk_usage(work).free < 40 * 1024**3:
        raise PreparationError("insufficient_disk_40_gib_free_required")
    limits = [os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")]
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(path).read_text().strip()
            if value.isdigit():
                limits.append(int(value))
        except FileNotFoundError:
            pass
    if min(limits) < 28 * 1024**3:
        raise PreparationError("insufficient_memory_32_gb_instance_required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("/data/natlas/work"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/natlas/model"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Linux advisory lock is released by the OS even after an interrupted deploy.
    import fcntl
    with (args.work_dir / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare(args)
        except Exception as exc:
            previous = args.work_dir / "status.json"
            failed_stage = json.loads(previous.read_text())["stage"] if previous.exists() else "starting"
            stage(args.work_dir, "failed", failed_stage=failed_stage, error_type=type(exc).__name__,
                  code=getattr(exc, "code", "preparation_error"))
            raise


def prepare(args):
    if reuse_prepared(args.output_dir):
        stage(args.work_dir, "reused")
        return
    stage(args.work_dir, "resource_check")
    check_resources(args.work_dir)
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from llama_cpp import Llama

    if not os.environ.get("HF_TOKEN"):
        raise PreparationError("hf_token_required")
    revision = subprocess.check_output(["git", "-C", "/opt/llama.cpp", "rev-parse", "HEAD"], text=True).strip()
    if revision != LLAMA_CPP_REVISION:
        raise ValueError("Unexpected converter revision")
    output = args.output_dir / MODEL_FILE
    manifest_path = args.output_dir / MANIFEST_FILE
    stage(args.work_dir, "downloading")
    snapshot = Path(snapshot_download(
        repo_id=MODEL_ID, revision=SOURCE_REVISION, token=os.environ["HF_TOKEN"],
        local_dir=args.work_dir / "source",
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.md", "LICENSE*"],
    ))
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("architectures") != ["LlamaForCausalLM"] or config.get("vocab_size") != 128256:
        raise ValueError("Unexpected model architecture/tokenizer")
    # Child converter needs only local files; never pass the token to subprocesses.
    child_env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "AI_SERVICE_TOKEN")}
    child_env["HF_HUB_OFFLINE"] = "1"
    intermediate = args.work_dir / "natlas-f16.gguf"
    checkpoint = args.work_dir / "f16.json"
    stage(args.work_dir, "converting")
    if not (intermediate.exists() and checkpoint.exists()
            and json.loads(checkpoint.read_text()).get("source_revision") == SOURCE_REVISION
            and json.loads(checkpoint.read_text()).get("converter") == LLAMA_CPP_REVISION
            and json.loads(checkpoint.read_text())["sha256"] == sha256(intermediate)):
        partial = args.work_dir / "natlas-f16.gguf.partial"
        subprocess.run([sys.executable, "/opt/llama.cpp/convert_hf_to_gguf.py", str(snapshot),
                        "--outfile", str(partial), "--outtype", "f16"], check=True, env=child_env)
        partial.replace(intermediate)
        atomic_json(checkpoint, {"sha256": sha256(intermediate),
                                 "source_revision": SOURCE_REVISION, "converter": LLAMA_CPP_REVISION})
    pending = args.output_dir / (MODEL_FILE + ".partial")
    stage(args.work_dir, "quantizing")
    subprocess.run(["/opt/llama.cpp/build/bin/llama-quantize", str(intermediate), str(pending), "Q4_K_M"],
                   check=True, env=child_env)
    stage(args.work_dir, "tokenizer_check")
    # Check the actual converted tokenizer against the original on all API languages.
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    llm = Llama(model_path=str(pending), vocab_only=True, n_gpu_layers=0, verbose=False)
    try:
        if llm.metadata.get("tokenizer.chat_template") != tokenizer.chat_template:
            raise ValueError("Chat template changed during conversion")
        for text in ("Hello, how are you?", "Ẹ káàárọ̀. Báwo ni?", "Sannu. Yaya kake?", "Ndewo. Kedu ka ị mere?"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, date_string="12 Oct 2026")
            if tokenizer.encode(prompt, add_special_tokens=False) != llm.tokenize(prompt.encode(), add_bos=False, special=True):
                raise ValueError("GGUF and source tokenizer IDs differ")
    finally:
        llm.close()
    pending.replace(output)
    manifest = {
        "model_id": MODEL_ID, "source_revision": SOURCE_REVISION,
        "llama_cpp_revision": LLAMA_CPP_REVISION, "llama_cpp_python": "0.3.16",
        "quantization": "Q4_K_M", "filename": MODEL_FILE,
        "sha256": sha256(output), "size_bytes": output.stat().st_size,
        "tokenizer_checks": ["en", "yo", "ha", "ig"],
        "generation_verified": False,
    }
    atomic_json(manifest_path, manifest)
    # Publish last. An interrupted partial conversion never looks completed.
    atomic_json(args.output_dir / COMMIT_FILE, {"manifest_sha256": sha256(manifest_path)})
    stage(args.work_dir, "converted")
    print("NATLAS_MANIFEST_SHA256=" + sha256(manifest_path))
    print("Conversion/tokenizer checks finished. Run smoke_cpu.py to verify genuine generation.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # SDK exceptions may contain signed URLs: never dump their text/tracebacks.
        print("Conversion failed: " + type(exc).__name__, file=sys.stderr)
        sys.exit(1)
