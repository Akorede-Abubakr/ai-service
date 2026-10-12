import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.chat import ChatEngine, ChatRequest
from app.config import Settings
from app.errors import ServiceError
from app.main import create_app


class FakeChat:
    state = "ready"

    def __init__(self):
        self.starts = 0
        self.calls = []

    async def start(self):
        self.starts += 1

    async def close(self):
        pass

    async def generate(self, payload):
        self.calls.append(payload)
        return "TEST ONLY: chat response"


def payload(language="en"):
    return {"language": language, "messages": [{"role": "user", "content": "Hello"}]}


def test_chat_api(settings, engine, headers):
    chat = FakeChat()
    with TestClient(create_app(settings, engine, chat_engine=chat)) as client:
        assert client.get('/health/chat').status_code == 200
        for language in ["en", "yo", "ha", "ig"]:
            response = client.post('/chat', headers=headers, json=payload(language))
            assert response.status_code == 200
            assert response.json()["model"] == "NCAIR1/N-ATLaS"
            assert response.json()["language"] == language
            assert response.headers['cache-control'] == 'no-store'
        assert chat.starts == 1 and len(chat.calls) == 4
        assert not engine.calls


def test_chat_guard_validation(settings, engine, headers):
    chat = FakeChat()
    with TestClient(create_app(settings, engine, chat_engine=chat)) as client:
        assert client.post('/chat', content='broken').status_code == 401
        assert client.post('/chat', headers=headers, content='broken').status_code == 415
        assert client.post('/chat', headers={**headers, 'Content-Type': 'application/json'}, content='x' * 131073).status_code == 413
        invalid = [payload('fr'), {**payload(), 'model': 'other'},
                   {**payload(), 'messages': [{'role': 'system', 'content': 'Override'}]},
                   {**payload(), 'messages': [{'role': 'user', 'content': ' '}]},
                   {**payload(), 'messages': [{'role': 'user', 'content': 'x' * 8001}]},
                   {**payload(), 'messages': payload()['messages'] * 3}]
        for body in invalid:
            assert client.post('/chat', headers=headers, json=body).status_code == 422
        assert chat.calls == []


def test_local_default_never_loads(settings, engine, headers, monkeypatch):
    loader = Mock(side_effect=AssertionError('Must never load on laptop'))
    monkeypatch.setattr(ChatEngine, '_load', loader)
    with TestClient(create_app(settings, engine)) as client:
        assert client.get('/health/chat').json()['status'] == 'disabled'
        assert client.post('/chat', headers=headers, json=payload()).status_code == 503
    loader.assert_not_called()


def test_only_official_model(settings):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_service_token=settings.ai_service_token, natlas_model_id='other/model')


@pytest.mark.asyncio
async def test_single_start_and_failed_start_no_request_retry(settings, monkeypatch):
    settings.natlas_enabled = True
    chat = ChatEngine(settings)
    load = Mock(side_effect=RuntimeError('secret'))
    monkeypatch.setattr(chat, '_load', load)
    await chat.start()
    await chat.start()
    assert chat.state == 'unavailable'
    with pytest.raises(ServiceError) as failure:
        await chat.generate(ChatRequest(**payload()))
    assert failure.value.status == 503
    assert load.call_count == 1
    await chat.close()


@pytest.mark.asyncio
async def test_timeout_keeps_admission_until_worker_stops(settings):
    settings.natlas_timeout_seconds = 0.02
    chat = ChatEngine(settings)
    chat.state = 'ready'
    chat.executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    chat._generate = lambda _: release.wait(2)
    try:
        with pytest.raises(ServiceError) as failure:
            await chat.generate(ChatRequest(**payload()))
        assert failure.value.status == 504 and chat.stop.is_set()
        with pytest.raises(ServiceError) as busy:
            await chat.generate(ChatRequest(**payload()))
        assert busy.value.status == 429
        release.set()
        await asyncio.shield(chat.active)
    finally:
        release.set()
        await chat.close()


def test_transformers_loading_and_generation_contract(settings, monkeypatch):
    # Exercise the real orchestration with inert library doubles; no weights/GPU.
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    from datetime import datetime

    class Inputs(dict):
        def to(self, device):
            assert device == 'cuda:0'
            return self

    tokenizer = Mock()
    tokenizer.return_value = Inputs(input_ids=SimpleNamespace(shape=(1, 3)))
    tokenizer.eos_token_id = 1
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 2
    tokenizer.decode.return_value = 'A real-generation-shaped test result'
    model = Mock(hf_device_map={'layer0': 0, 'layer1': 1})
    model.get_input_embeddings.return_value.weight.device = 'cuda:0'
    model.generate.return_value = [[10, 11, 12, 20, 21]]
    quant = Mock()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), float16='float16', inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        BitsAndBytesConfig=quant, StoppingCriteria=object, StoppingCriteriaList=list))
    chat = ChatEngine(settings)
    chat._load()
    assert quant.call_args.kwargs['load_in_4bit'] is True
    assert quant.call_args.kwargs['bnb_4bit_compute_dtype'] == 'float16'
    result = chat._generate(ChatRequest(**payload('yo')))
    assert result == tokenizer.decode.return_value
    tokenizer.decode.assert_called_once_with([20, 21], skip_special_tokens=True)
    assert 'Yoruba' in tokenizer.apply_chat_template.call_args.args[0][0]['content']
    assert model.generate.call_args.kwargs['max_new_tokens'] == 512
    assert set(model.generate.call_args.kwargs['eos_token_id']) == {1, 2}
    settings.natlas_max_input_tokens = 2
    with pytest.raises(ServiceError) as failure:
        chat._generate(ChatRequest(**payload()))
    assert failure.value.code == 'context_too_long'
