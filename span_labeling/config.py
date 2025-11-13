from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).absolute().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_cached: Dict[str, Any] | None = None


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    global _cached
    if _cached is not None:
        return _cached

    # Allow override via env var
    cfg_path = Path(os.getenv("SPAN_LABELING_CONFIG", path or DEFAULT_CONFIG_PATH))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        _cached = yaml.safe_load(f) or {}

    return _cached


def get_ollama_base_url() -> str:
    cfg = load_config()
    return (
        cfg.get("llm", {})
        .get("ollama", {})
        .get("base_url", "http://localhost:11434/v1")
    )


def get_ollama_model() -> str:
    cfg = load_config()
    return cfg.get("llm", {}).get("ollama", {}).get("model", "llama3.1")


def get_hf_api_url() -> str:
    cfg = load_config()
    return cfg.get("llm", {}).get("hf_api", {}).get("api_url", "http://localhost:8000")


def get_hf_model() -> str:
    cfg = load_config()
    return (
        cfg.get("llm", {})
        .get("hf_api", {})
        .get("model", "microsoft/Phi-4-mini-instruct")
    )


def get_vllm_api_url() -> str:
    cfg = load_config()
    return cfg.get("llm", {}).get("vllm", {}).get("api_url", "http://localhost:8000")


def get_vllm_model() -> str:
    cfg = load_config()
    return (
        cfg.get("llm", {}).get("vllm", {}).get("model", "microsoft/Phi-4-mini-instruct")
    )


def get_system_message() -> str:
    cfg = load_config()
    return cfg.get("llm", {}).get("system_message", "You are a helpful assistant.")


def get_hard_matching() -> bool:
    cfg = load_config()
    return cfg.get("evaluation", {}).get("hard_matching", True)


def get_constrained_xml_debug() -> bool:
    cfg = load_config()
    return cfg.get("methods", {}).get("constrained_xml", {}).get("debug", False)
