from abc import abstractmethod
from typing import List, Dict
from span_labeling.base import SpanLabelerBase
from span_labeling.prompt_utils import build_prompt
from span_labeling.config import Settings

settings = Settings.get()


class SpanLabeler(SpanLabelerBase):
    key: str = "base"

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def format_prompt(self, entry: dict) -> str:
        return build_prompt(self.key, entry["key"], entry)

    @abstractmethod
    def parse_response(self, entry: dict) -> List[Dict]:
        pass

    def predict(self, entry: dict) -> Dict:
        """Main synchronous method (remains for compatibility/sequential use)"""
        entry["prompt"] = self.format_prompt(entry)
        entry = self.model.predict(entry)

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

    async def async_predict(self, entry: dict) -> Dict:
        """Main asynchronous method for concurrent execution"""
        # CPU-bound formatting is fast and fine to run synchronously
        entry["prompt"] = self.format_prompt(entry)
        if self.config.method.use_structured_outputs:
            entry["response_format"] = self.get_json_schema(
                entry["key"], self.config.model.mode
            )
        entry = await self.model.async_predict(entry)

        try:
            # CPU-bound parsing is also fast and synchronous
            spans = self.parse_response(entry)
            entry["output"] = {
                "success": True,
                "spans": spans,
                "raw_response": entry["response"],
            }
            if entry.get("reasoning"):
                entry["output"]["reasoning"] = entry["reasoning"]
        except Exception as e:
            entry["output"] = {
                "success": False,
                "spans": [],
                "raw_response": entry["response"],
                "error": str(e),
            }
            if entry.get("reasoning"):
                entry["output"]["reasoning"] = entry["reasoning"]

        return entry
