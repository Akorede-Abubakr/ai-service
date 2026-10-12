"""N-ATLaS chat; model libraries are imported only on explicit opt-in."""
import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ServiceError

LANGUAGES = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}
logger = logging.getLogger("lightbridge.chat")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Empty message")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: Literal["en", "yo", "ha", "ig"]
    messages: list[ChatMessage] = Field(min_length=1, max_length=21)

    @model_validator(mode="after")
    def valid_history(self):
        if len(self.messages) % 2 != 1 or any(
            m.role != ("user" if i % 2 == 0 else "assistant")
            for i, m in enumerate(self.messages)
        ) or sum(len(m.content) for m in self.messages) > 24000:
            raise ValueError("Send alternating user/assistant history ending with user, at most 24000 characters")
        return self


class ChatEngine:
    def __init__(self, settings):
        self.settings = settings
        self.state = "disabled" if not settings.natlas_enabled else "not_loaded"
        self.model = self.tokenizer = None
        self.executor = None
        self.active = None
        self.stop = threading.Event()

    async def start(self):
        # Called by lifespan, never by /chat. A failed load requires restart.
        if self.state != "not_loaded":
            return
        self.state = "loading"
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="natlas")
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor, self._load)
            self.state = "ready"
        except Exception as exc:
            self.state = "unavailable"
            logger.error("code=natlas_initialization_failed error_type=%s", type(exc).__name__)

    def _load(self):
        if self.settings.natlas_backend == "gguf":
            from .cpu import CpuModel
            self.model = CpuModel(self.settings)
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("N-ATLaS requires a CUDA GPU; CPU loading is disabled")
        options = dict(
            token=self.settings.hf_token.get_secret_value() if self.settings.hf_token else None,
            cache_dir=self.settings.hf_home, revision=self.settings.natlas_revision,
            local_files_only=self.settings.hf_local_files_only, trust_remote_code=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.natlas_model_id, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.natlas_model_id, **options, torch_dtype=torch.float16,
            device_map="auto", quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            ),
        )
        if any(str(device) in ("cpu", "disk") for device in self.model.hf_device_map.values()):
            self.model = None
            raise RuntimeError("Insufficient GPU memory")
        self.model.eval()

    def _generate(self, request):
        if self.settings.natlas_backend == "gguf":
            return self.model.generate(request, self.stop)
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        stop = self.stop
        deadline = time.monotonic() + self.settings.natlas_timeout_seconds

        class StopOnDeadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set() or time.monotonic() >= deadline

        messages = [{"role": "system", "content": (
            "You are LightBridge AI, a helpful assistant on MUSLIM GUARDIAN, powered by N-ATLaS. "
            f"Respond in {LANGUAGES[request.language]}. Be clear and honest about uncertainty. "
            "Do not invent religious citations or claim access to the user's books or account."
        )}] + [m.model_dump() for m in request.messages]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            date_string=datetime.now(timezone.utc).strftime("%d %b %Y"),
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        length = inputs["input_ids"].shape[-1]
        if length > self.settings.natlas_max_input_tokens:
            raise ServiceError(422, "context_too_long", "This conversation is too long. Start a new chat or shorten your message.")
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        terminators = [self.tokenizer.eos_token_id]
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != self.tokenizer.unk_token_id:
            terminators.append(eot)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.settings.natlas_max_new_tokens,
                do_sample=True, temperature=0.1, repetition_penalty=1.12, use_cache=True,
                eos_token_id=list(set(terminators)), pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnDeadline()]),
            )
        if stop.is_set() or time.monotonic() >= deadline:
            raise ServiceError(504, "inference_timeout", "The response took too long. Please retry.")
        reply = self.tokenizer.decode(outputs[0][length:], skip_special_tokens=True).strip()
        if not reply or len(reply) > 8000:
            raise ServiceError(503, "inference_failed", "The model could not produce a usable response. Please retry.")
        return reply

    async def generate(self, request):
        if self.state != "ready":
            raise ServiceError(503, "chat_unavailable", "LightBridge AI is unavailable. The inference service must be enabled and ready.")
        if self.active is not None and not self.active.done():
            raise ServiceError(429, "service_busy", "LightBridge AI is busy. Please retry shortly.")
        self.stop.clear()
        self.active = asyncio.get_running_loop().run_in_executor(self.executor, self._generate, request)
        # Retrieve late exceptions after HTTP timeout/cancellation, without logging prompts.
        self.active.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        try:
            return await asyncio.wait_for(asyncio.shield(self.active), self.settings.natlas_timeout_seconds)
        except (TimeoutError, asyncio.CancelledError) as exc:
            self.stop.set()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise ServiceError(504, "inference_timeout", "The response took too long. Please retry.") from None
        except ServiceError:
            raise
        except Exception:
            raise ServiceError(503, "inference_failed", "The model could not complete this response. Please retry.") from None

    async def close(self):
        self.stop.set()
        if self.executor:
            executor, self.executor = self.executor, None
            if self.settings.natlas_backend == "gguf":
                # Serialize destruction after any still-running load/generation.
                executor.submit(self._close_cpu)
            executor.shutdown(wait=False, cancel_futures=self.settings.natlas_backend != "gguf")

    def _close_cpu(self):
        if self.model is not None:
            self.model.llm.close()
            self.model = None
