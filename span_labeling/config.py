from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
from pathlib import Path
from typing import Optional
import yaml

PROJECT_ROOT = Path(__file__).absolute().parent.parent.as_posix()


class ModelSettings(BaseSettings):
    mode: str = "vllm"

    base_url: str | None = None
    api_key: str | None = "testkey"
    organization: str | None = None

    system_prompt: str | None = "You are a precise span labeling model."

    max_completion_tokens: int | None = 4096
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    enable_thinking: bool | None = False
    example_n: int | None = 10000

    @model_validator(mode="after")
    def set_derived_values(self):
        if self.base_url is None:
            if self.mode == "vllm":
                self.base_url = "http://localhost:8057/v1"
            elif self.mode == "openai":
                self.base_url = "https://api.openai.com/v1"
            elif self.mode == "ollama":
                self.base_url = "http://localhost:11434/v1"
            elif self.mode == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1"

        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",  # Load from .env file
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    experiment_name: str = "span_labeling_experiment"
    model: ModelSettings = Field(default_factory=ModelSettings)

    # Class-level storage
    _instance: Optional["Settings"] = None

    @classmethod
    def from_yaml(cls, yaml_file: Path) -> "Settings":
        with open(yaml_file, "r", encoding="utf-8") as f:
            instance = cls(**yaml.safe_load(f))
            cls._instance = instance
            return instance

    @classmethod
    def get(cls) -> "Settings":
        """Get the global settings instance"""
        if cls._instance is None:
            raise RuntimeError(
                "Settings not initialized. Call Settings.from_yaml() first"
            )
        return cls._instance


Settings.from_yaml(Path(PROJECT_ROOT) / "span_labeling/config.yaml")


if __name__ == "__main__":
    settings = Settings.get()
    print(settings.model)
