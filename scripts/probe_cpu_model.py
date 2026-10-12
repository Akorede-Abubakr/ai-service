"""Read only small upstream metadata; never download weights or print credentials."""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

REVISION = "e294476928aca9030e924ca27bb8e085e8581273"


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing")
    for name in ("config.json", "tokenizer_config.json"):
        request = urllib.request.Request(
            f"https://huggingface.co/NCAIR1/N-ATLaS/resolve/{REVISION}/{name}",
            headers={"Authorization": "Bearer " + token},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        fields = ("architectures", "model_type", "hidden_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "vocab_size",
                  "max_position_embeddings", "rope_scaling", "bos_token", "eos_token", "chat_template")
        print(json.dumps({k: data[k] for k in fields if k in data}, ensure_ascii=True))
    if "--check-converter" in sys.argv:
        commit = "4227c9be4268ac844921b90f31595f81236bd317"
        base = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{commit}/"
        for filename in ("requirements/requirements-convert_hf_to_gguf.txt", "convert_hf_to_gguf.py"):
            with urllib.request.urlopen(base + filename, timeout=30) as response:
                source = response.read().decode()
            if filename.endswith(".txt"):
                print(json.dumps({"converter_requirements": source}))
            else:
                print(json.dumps({"converter_llama_supported": '"LlamaForCausalLM"' in source,
                                  "converter_llama3_tokenizer": '"llama-bpe"' in source,
                                  "imports": re.findall(r"^(?:import .+|from .+ import .+)$", source, re.M)}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__,
                          "http_status": getattr(exc, "code", None)}))
        sys.exit(1)
