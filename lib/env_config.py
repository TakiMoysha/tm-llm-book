import os
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    OPENAI_URL: str = field(default_factory=lambda: os.getenv("OPENAI_URL"))
    LLM_MODEL: str = field(default_factory=lambda: os.getenv("LLM_MODEL"))
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL"))
    EMBEDDING_MODEL_DIMENSIONS: int = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_DIMENSIONS"))

    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"), repr=False)
    VAULT_PATH: str = field(default_factory=lambda: os.getenv("OBSIDIAN_VAULT_PATH"))

    QDRANT_URL: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ":memory:"))

    S3_API_URL: str = field(default_factory=lambda: os.getenv("S3_API_URL"))
    S3_ACCESS_KEY: str = field(default_factory=lambda: os.getenv("S3_ACCESS_KEY"), repr=False)
    S3_SECRET_KEY: str = field(default_factory=lambda: os.getenv("S3_SECRET_KEY"), repr=False)
    S3_SECURE: bool = field(default_factory=lambda: os.getenv("S3_SECURE") == "true")

    def __post_init__(self):
        bad_values = [k for k, v in self.__dict__.items() if v is None]

        if bad_values:
            raise ValueError(f"Missing environment variable, check {bad_values}")


def test_env_config_loading():
    config = EnvConfig()
    print(config)
