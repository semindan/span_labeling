from abc import abstractmethod
from openai import OpenAI
from typing import List, Dict
from span_labeling.base import SpanLabelerBase
from span_labeling.prompt_utils import build_prompt


class SpanLabeler(SpanLabelerBase):
    name: str = "base"

    def __init__(self, model_name="hermes3:8b"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def call_api(self, prompt: str) -> str:
        """Call Ollama via OpenAI client"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise span labeling system. Follow the output format exactly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=8192,
            )
            response = response.choices[0].message.content

            if "<think>" in response and "</think>" in response:
                response = response.split("</think>")[-1]

            return response

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
