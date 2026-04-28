from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
from typing import Optional
import yaml

PROJECT_ROOT = Path(__file__).absolute().parent.parent.as_posix()
CODE_ROOT = Path(PROJECT_ROOT) / "span_labeling"
CONFIGS_DIR = Path(CODE_ROOT) / "configs"


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=Path(PROJECT_ROOT) / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    api_key: str | None = Field(default=None, exclude=True)
    base_url: str | None = Field(default="https://api.openai.com/v1", exclude=True)
    organization: str | None = Field(default=None, exclude=True)


class OpenRouterSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    api_key: str | None = Field(default=None, exclude=True)
    base_url: str | None = Field(default="https://openrouter.ai/api/v1", exclude=True)


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=Path(PROJECT_ROOT) / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    api_key: str = Field(default="ollama", exclude=True)
    port: int = Field(default=11434, exclude=True)

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"


class VllmSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=Path(PROJECT_ROOT) / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(default="testkey", exclude=True)
    port: int = Field(default=8057, exclude=True)

    model: str = "mistralai/Mistral-Small-24B-Instruct-2501"
    # seed: int = 42
    tensor_parallel_size: int = 2
    max_num_seqs: int = 8
    gpu_mem_util: float = 0.85
    max_model_len: int = 16384
    max_batched_tokens: int = 4096
    dtype: str = "bfloat16"
    logits_processor: str | None = None
    reasoning_parser: str | None = None
    quantization: str | None = None
    chat_template: str | None = None
    attention_backend: str | None = "FLASH_ATTN"
    batch_invariant: bool = True

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"


PROVIDER_MAP = {
    "openai": OpenAISettings,
    "openrouter": OpenRouterSettings,
    "ollama": OllamaSettings,
    "vllm": VllmSettings,
}


class ProviderSettings(BaseSettings):
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    vllm: VllmSettings = Field(default_factory=VllmSettings)

    def get(
        self, mode: str
    ) -> OpenAISettings | OpenRouterSettings | OllamaSettings | VllmSettings:
        provider = getattr(self, mode, None)
        if provider is None:
            raise ValueError(f"Unknown mode: {mode}")
        return provider


class ModelConfig(BaseModel):
    name: str
    mode: str = "vllm"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_completion_tokens: int | None = 16384
    enable_thinking: bool = False
    system_prompt: str | None = (
        "You are a precise span labeling model. Always provide answers in the required format. Output the final answer after 'Output:'."
    )

    # resolved by validator, do not set in yaml
    api_key: str | None = Field(default=None, exclude=True)
    base_url: str | None = Field(default=None, exclude=True)
    organization: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def resolve_provider(self):
        provider_cls = PROVIDER_MAP.get(self.mode)
        if provider_cls is None:
            raise ValueError(f"Unknown mode: {self.mode}")
        provider = provider_cls()
        if self.api_key is None:
            self.api_key = provider.api_key
            if self.api_key is None:
                raise ValueError(f"API key not set for provider {self.mode}")
        if self.base_url is None:
            self.base_url = provider.base_url
        if hasattr(provider, "organization") and self.organization is None:
            self.organization = provider.organization
        return self

    def set_port(self, port: int) -> None:
        """Replace the port in base_url, preserving host and path."""
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(self.base_url)
        self.base_url = urlunparse(parsed._replace(netloc=f"{parsed.hostname}:{port}"))


class DatasetConfig(BaseModel):
    type: str
    path: str
    name: str | None = None

    @model_validator(mode="after")
    def set_name_from_path(self):
        if self.name is None:
            self.name = Path(self.path).stem
        return self


class MethodConfig(BaseModel):
    model_config = SettingsConfigDict(extra="allow")
    type: str
    name: str
    use_structured_outputs: bool = False
    constrained: bool = False
    enrich_prompt: bool = False


class ExperimentSettings(BaseModel):
    name: str = "span_labeling_experiment"
    output_dir: str = "results"
    models: list[ModelConfig] = Field(min_length=1)
    datasets: list[DatasetConfig] = Field(default_factory=list)
    dataset_groups: list[str] = Field(
        default_factory=list
    )  # e.g. ["ner_all", "wmt_news"]
    methods: list[MethodConfig] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=lambda: [42])

    @model_validator(mode="after")
    def validate_models(self):
        modes = {m.mode for m in self.models}
        if len(modes) > 1:
            raise ValueError(f"All models must use the same mode, got: {modes}")

        if self.models[0].mode == "vllm":
            names = {m.name for m in self.models}
            if len(names) > 1:
                raise ValueError(
                    f"Multiple vllm models would require multiple servers: {names}"
                )

        return self

    @model_validator(mode="after")
    def expand_dataset_groups(self):
        from span_labeling.datasets.dataset_groups import DATASET_GROUPS

        print(f"Expanding dataset groups: {self.dataset_groups}")
        for group in self.dataset_groups:
            if group not in DATASET_GROUPS:
                raise ValueError(f"Unknown dataset group: {group}")
            self.datasets += DATASET_GROUPS[group]
        return self

    @property
    def mode(self) -> str:
        return self.models[0].mode


class SlurmSettings(BaseModel):
    job_name: str = "exp_span_labeling"
    partition: str = "gpu-ms"
    num_gpus: int = 2
    constraint: str = "gpuram40G|gpuram48G"
    time: str = "48:00:00"
    cpus: int = 4
    mem: str = "48G"


class EnvSettings(BaseModel):
    cuda_home: str = "/opt/cuda/12.3"
    hf_home: str = "/lnet/troja/work/people/<your_username>/.cache/huggingface"
    hf_hub_offline: int = 1


class ProjectSettings(BaseModel):
    dir: str = "/home/<your_username>/personal_work_ms/span_labeling"
    run_script: str = "span_labeling/run.py"
    max_concurrent_requests: int = 8
    skip_experiment_if_exists: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(PROJECT_ROOT) / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    config_path: str | None = None

    experiment: ExperimentSettings
    slurm: SlurmSettings = Field(default_factory=SlurmSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    env: EnvSettings = Field(default_factory=EnvSettings)
    project: ProjectSettings = Field(default_factory=ProjectSettings)

    model: ModelConfig | None = None
    dataset: DatasetConfig | None = None
    method: MethodConfig | None = None
    seed: int | None = None

    _instance: Optional["Settings"] = None

    @classmethod
    def from_yaml(cls, yaml_file: Path | str) -> "Settings":
        yaml_file = Path(yaml_file)
        with open(yaml_file, "r", encoding="utf-8") as f:
            instance = cls(**yaml.safe_load(f))
            cls._instance = instance
            cls._instance.config_path = yaml_file.as_posix()
            return instance

    @classmethod
    def get(cls) -> "Settings":
        if cls._instance is None:
            raise RuntimeError(
                "Settings not initialized. Call Settings.from_yaml() first"
            )
        return cls._instance


# %%
if __name__ == "__main__":
    from pprint import pprint

    settings = Settings.from_yaml("<config_path>")
    pprint(settings.model_dump())
