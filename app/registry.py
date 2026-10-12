from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str


# Official Hub metadata verified 2026-10-09. No user-supplied model paths/code.
MODELS = {
    "yo": ModelSpec("NCAIR1/Yoruba-ASR", "d1ae7b8b79c2ccd547d8761effe5057433f3fc7f"),
    "en-NG": ModelSpec("NCAIR1/NigerianAccentedEnglish", "3c52c6e6c9ec508014a7b9db6a42b503b8930dff"),
    "ha": ModelSpec("NCAIR1/Hausa-ASR", "e635b9eda29060c6114c8f4d8b2d903f5c83a44a"),
    "ig": ModelSpec("NCAIR1/Igbo-ASR", "180732299d5cba3dc8b289260ac84b7838bb3954"),
}
