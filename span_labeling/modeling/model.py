from abc import ABC, abstractmethod
from typing import Dict
import re
import time
import json
from pydantic import BaseModel


class Model(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def resolve_parameters(self, entry: dict) -> dict:
        """Resolve model-specific parameters"""
        pass

    @abstractmethod
    def call_api(self, entry: dict) -> tuple[str, int | None]:
        """Call the model API. Returns (response_text, completion_tokens)."""
        pass

    @abstractmethod
    async def async_call_api(self, entry: dict) -> tuple[str, int | None]:
        """Asynchronously call the model API. Returns (response_text, completion_tokens)."""
        pass

    def _sanitize_for_json(self, value):
        """Best-effort conversion to JSON-serializable primitives."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, type) and issubclass(value, BaseModel):
            return self._sanitize_for_json(value.model_json_schema())

        if isinstance(value, BaseModel):
            return self._sanitize_for_json(value.model_dump())

        if isinstance(value, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._sanitize_for_json(v) for v in value]

        return str(value)

    def _sanitize_entry(self, entry: dict) -> dict:
        return {k: self._sanitize_for_json(v) for k, v in entry.items()}

    def postprocess_response(self, entry: dict, response) -> Dict:
        if hasattr(response.choices[0].message, "parsed"):
            entry["response"] = json.dumps(
                response.choices[0].message.parsed.model_dump()["spans"]
            )
        else:
            entry["response"] = response.choices[0].message.content

        entry["completion_tokens"] = response.usage.completion_tokens

        if "<think>" in entry["response"] and "</think>" in entry["response"]:
            pattern = r"<think>(.*)<\/think>(.*)"
            matched = re.match(pattern, response, re.DOTALL)
            entry["reasoning"] = matched.group(1)
            entry["response"] = matched.group(2)
        elif (
            hasattr(response.choices[0].message, "reasoning")
            and response.choices[0].message.reasoning is not None
        ):
            entry["reasoning"] = response.choices[0].message.reasoning
        elif (
            hasattr(response.choices[0].message, "reasoning_content")
            and response.choices[0].message.reasoning_content is not None
        ):
            entry["reasoning"] = response.choices[0].message.reasoning_content

        return entry

    def predict(self, entry: dict) -> Dict:
        try:
            t0 = time.perf_counter()
            response = self.call_api(entry)
            entry["latency"] = time.perf_counter() - t0
            entry = self.postprocess_response(entry, response)
        except Exception as e:
            entry["response"] = ""
            entry["error"] = str(e)
        return self._sanitize_entry(entry)

    async def async_predict(self, entry: dict) -> Dict:
        try:
            t0 = time.perf_counter()
            response = await self.async_call_api(entry)
            entry["latency"] = time.perf_counter() - t0
            entry = self.postprocess_response(entry, response)
        except Exception as e:
            entry["response"] = ""
            entry["error"] = str(e)
        return self._sanitize_entry(entry)
