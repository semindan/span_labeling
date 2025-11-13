from abc import abstractmethod
from typing import Dict, List

import requests

from span_labeling.base import SpanLabelerBase
from span_labeling.config import (
    get_ollama_base_url,
    get_ollama_model,
    get_system_message,
)
from span_labeling.prompt_utils import build_prompt


class SpanLabeler(SpanLabelerBase):
    name: str = "base"

    def __init__(self, model_name: str | None = None):
        # Default to central config if not provided
        self.model_name = model_name or get_ollama_model()
        # Remove /v1 suffix if present (OpenAI compatibility endpoint)
        self.base_url = get_ollama_base_url().replace("/v1", "")

    def call_api(self, prompt: str) -> str:
        """Call Ollama via native API"""
        try:
            # Native Ollama API format with explicit deterministic parameters
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": get_system_message(),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    # Core sampling parameters - deterministic greedy decoding
                    "temperature": 0.0,
                    "num_predict": 1024,  # max_tokens equivalent in Ollama
                    "top_k": -1,  # Disable top-k filtering (consider all tokens)
                    "top_p": 1.0,  # Disable top-p/nucleus sampling
                    "min_p": 0.0,  # Disable min-p filtering
                    # Repetition/frequency penalties - disabled for determinism
                    "repeat_last_n": 0,  # Context for repetition penalty (Ollama default)
                    "repeat_penalty": 1.0,  # No repetition penalty
                    "presence_penalty": 0.0,  # No presence penalty
                    "frequency_penalty": 0.0,  # No frequency penalty
                    # Mirostat - disabled
                    "mirostat": 0,  # Disable mirostat
                    "mirostat_tau": 5.0,  # Not used when mirostat=0
                    "mirostat_eta": 0.1,  # Not used when mirostat=0
                    # Other parameters
                    "seed": 42,  # Set seed for reproducibility
                    "tfs_z": 1.0,  # Disable tail-free sampling
                    "typical_p": 1.0,  # Disable locally typical sampling
                },
            }

            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=300
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("message", {}).get("content", "")

            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[-1]

            return content

        except Exception as e:
            print(f"Error: {e}")
            return ""

    def format_prompt(self, entry: dict) -> str:
        return build_prompt(self.name, entry["key"], entry)

    @abstractmethod
    def parse_response(self, entry: dict) -> List[Dict]:
        pass

    def run(self, entry: dict) -> str:
        return self.call_api(entry["prompt"])

    def predict(self, entry: dict) -> Dict:
        """Main method"""
        entry["prompt"] = self.format_prompt(entry)
        entry["response"] = self.run(entry)

        try:
            spans = self.parse_response(entry)
            entry["output"] = {
                "success": True,
                "spans": spans,
                "raw_response": entry["response"],
            }
        except Exception as e:
            entry["output"] = {
                "success": False,
                "spans": [],
                "raw_response": entry["response"],
                "error": str(e),
            }
        return entry
