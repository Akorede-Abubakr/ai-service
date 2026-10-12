"""CPU-only llama.cpp inference using the converted model's own chat template."""
import time
from datetime import datetime, timezone

from jinja2.sandbox import ImmutableSandboxedEnvironment

from .artifact import acquire_artifact
from .chat import LANGUAGES
from .errors import ServiceError


class CpuModel:
    def __init__(self, settings):
        from llama_cpp import Llama

        self.settings = settings
        if settings.natlas_max_input_tokens + settings.natlas_max_new_tokens > settings.natlas_context_size:
            raise ValueError("Input and output budgets exceed GGUF context")
        path = acquire_artifact(settings)
        self.llm = Llama(
            model_path=str(path), n_gpu_layers=0, offload_kqv=False,
            n_ctx=settings.natlas_context_size, n_threads=settings.natlas_threads,
            n_threads_batch=settings.natlas_threads, n_batch=settings.natlas_batch_size,
            n_ubatch=settings.natlas_batch_size, use_mmap=True, verbose=False,
        )
        metadata = self.llm.metadata
        if metadata.get("general.architecture") != "llama" or int(metadata.get("general.file_type", -1)) != 15:
            raise ValueError("Expected Llama Q4_K_M artifact")
        template = metadata.get("tokenizer.chat_template")
        if not template:
            raise ValueError("Model chat template missing; refusing a generic fallback")
        self.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template)

    def generate(self, request, stop):
        from llama_cpp import StoppingCriteriaList

        deadline = time.monotonic() + self.settings.natlas_timeout_seconds
        messages = [{"role": "system", "content": (
            "You are LightBridge AI, a helpful assistant on MUSLIM GUARDIAN, powered by N-ATLaS. "
            f"Respond in {LANGUAGES[request.language]}. Be clear and honest about uncertainty. "
            "Do not invent religious citations or claim access to the user's books or account."
        )}] + [m.model_dump() for m in request.messages]

        def raise_exception(message):
            raise ValueError("Invalid model chat template input")

        prompt = self.template.render(
            messages=messages, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
            add_generation_prompt=True, date_string=datetime.now(timezone.utc).strftime("%d %b %Y"),
            raise_exception=raise_exception,
        )
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) > self.settings.natlas_max_input_tokens:
            raise ServiceError(422, "context_too_long", "This conversation is too long. Start a new chat or shorten your message.")
        result = self.llm.create_completion(
            prompt=tokens, max_tokens=self.settings.natlas_max_new_tokens,
            temperature=0.1, repeat_penalty=1.12,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"],
            stopping_criteria=StoppingCriteriaList([lambda *_: stop.is_set() or time.monotonic() >= deadline]),
            echo=False,
        )
        if stop.is_set() or time.monotonic() >= deadline:
            raise ServiceError(504, "inference_timeout", "The response took too long. Please retry.")
        reply = result["choices"][0]["text"].strip()
        if not reply or len(reply) > 8000:
            raise ServiceError(503, "inference_failed", "The model could not produce a usable response. Please retry.")
        return reply
