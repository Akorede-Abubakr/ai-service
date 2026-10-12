from pathlib import Path

from app.config import Settings


def test_default_env_file_is_service_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert Settings.model_config["env_file"] == Path(__file__).resolve().parents[1] / ".env"


def test_dotenv_loads_secrets_and_environment_takes_priority(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("AI_SERVICE_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AI_SERVICE_TOKEN=test-service-token-at-least-32-characters\nHF_TOKEN=test-hf-token\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file)
    assert settings.hf_token.get_secret_value() == "test-hf-token"
    assert "test-hf-token" not in repr(settings)
    monkeypatch.setenv("HF_TOKEN", "environment-token")
    assert Settings(_env_file=env_file).hf_token.get_secret_value() == "environment-token"
