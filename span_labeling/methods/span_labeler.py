from abc import abstractmethod
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict
from span_labeling.base import SpanLabelerBase
from span_labeling.prompt_utils import build_prompt
from span_labeling.config import Settings
import asyncio

settings = Settings.get()


class SpanLabeler(SpanLabelerBase):
    name: str = "base"

    def __init__(self, model_name="hermes3:8b"):
        self.model_name = model_name
        self.system_prompt = settings.model.system_prompt
        self.client = OpenAI(
            base_url=settings.model.base_url,
            api_key=settings.model.api_key,
        )

        self.async_client = AsyncOpenAI(
            base_url=settings.model.base_url,
            api_key=settings.model.api_key,
        )

    async def call_api_async(self, prompt: str) -> str:
        """Async version for batching"""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=settings.model.max_completion_tokens,
                chat_template_kwargs={"enable_thinking": False},
            )
            response = response.choices[0].message.content
            if "<think>" in response and "</think>" in response:
                response = response.split("</think>")[-1]
            return response
        except Exception as e:
            print(f"Error: {e}")
            return ""

    async def predict_async(self, entry: dict) -> Dict:
        """Async predict for batching"""
        entry["prompt"] = self.format_prompt(entry)
        entry["response"] = await self.call_api_async(entry["prompt"])
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

    async def predict_batch(
        self, entries: List[dict], batch_size: int = 32
    ) -> List[Dict]:
        results = []

        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            tasks = [self.predict_async(entry) for entry in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            print(
                f"Processed {min(i + batch_size, len(entries))}/{len(entries)} entries"
            )

        return results

    def call_api(self, prompt: str) -> str:
        """Call Model via OpenAI client"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=settings.model.max_completion_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
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
