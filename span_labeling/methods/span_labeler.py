from abc import abstractmethod
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict
from span_labeling.base import SpanLabelerBase
from span_labeling.prompt_utils import build_prompt
from span_labeling.config import Settings
import re
import json

settings = Settings.get()


class SpanLabeler(SpanLabelerBase):
    key: str = "base"

    def __init__(
        self,
        model_name="hermes3:8b",
        use_structured_outputs=False,
        enable_thinking=False,
        constrained=False,
        **kwargs,
    ):
        self.model_name = model_name
        self.use_structured_outputs = use_structured_outputs
        self.enable_thinking = enable_thinking
        self.constrained = constrained
        self.system_prompt = settings.model.system_prompt
        self.client = OpenAI(
            base_url=settings.model.base_url,
            organization=settings.model.organization,
            api_key=settings.model.api_key,
            timeout=6000,
        )
        self.async_client = AsyncOpenAI(
            base_url=settings.model.base_url,
            organization=settings.model.organization,
            api_key=settings.model.api_key,
            timeout=6000,
        )
        # Store additional API parameters
        self.api_kwargs = kwargs

    def call_api(self, entry: dict) -> str:
        """Call Model via OpenAI client"""
        try:
            prompt = entry["prompt"]
            # Build base parameters
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }

            if self.constrained:
                extra_body["vllm_xargs"] = {
                    "input_text": entry["text"],
                    "constrained_key": "text",
                    "allowed_labels": entry["allowed_labels"],
                    "label_key": "label",
                }

            params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": settings.model.max_completion_tokens,
                # "extra_body": extra_body
                "reasoning": "none",
            }

            # Add structured outputs schema if enabled
            if self.use_structured_outputs and hasattr(
                self.__class__, "get_json_schema"
            ):
                json_schema = self.__class__.get_json_schema(entry["key"])

                # Check if using OpenAI models (gpt-*, o1-*, etc.)
                if self.model_name.startswith(("gpt-", "o1-", "openai/")):
                    # Use OpenAI's response_format
                    params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "spans_output",
                            "strict": True,
                            "schema": json_schema,
                        },
                    }
                else:
                    # Use vllm's guided_json
                    extra_body["guided_json"] = json_schema

            # params.update(self.api_kwargs)

            response = self.client.chat.completions.create(**params)

            # Try to get parsed content first (OpenAI structured outputs)
            if (
                hasattr(response.choices[0].message, "parsed")
                and response.choices[0].message.parsed is not None
            ):
                return response.choices[0].message.parsed

            response = response.choices[0].message.content

            return response

        except Exception as e:
            print(f"Error: {e}")
            return ""

    def format_prompt(self, entry: dict) -> str:
        return build_prompt(self.key, entry["key"], entry)

    @abstractmethod
    def parse_response(self, entry: dict) -> List[Dict]:
        pass

    def run(self, entry: dict) -> str:
        return self.call_api(entry)

    def predict(self, entry: dict) -> Dict:
        """Main synchronous method (remains for compatibility/sequential use)"""
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

    async def async_call_api(self, entry: dict) -> str:
        """Asynchronously call Model via AsyncOpenAI client"""
        try:
            prompt = entry["prompt"]
            # input_text = entry["text"]

            # Build base parameters
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            }

            if self.constrained:
                extra_body["vllm_xargs"] = {
                    "input_text": entry["text"],
                    "constrained_key": "text",
                    "allowed_labels": entry["allowed_labels"],
                    "label_key": "label",
                }

            params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                # "max_completion_tokens": settings.model.max_completion_tokens,
                # "max_tokens": settings.model.max_completion_tokens,
                # "extra_body": extra_body,
            }

            if settings.model.mode == "vllm":
                params["extra_body"] = extra_body
                params["max_tokens"] = settings.model.max_completion_tokens
            elif settings.model.mode == "openai":
                params["max_completion_tokens"] = settings.model.max_completion_tokens
                params["reasoning_effort"] = "minimal"

            # Add structured outputs schema if enabled
            if self.use_structured_outputs and hasattr(
                self.__class__, "get_json_schema"
            ):
                json_schema = self.__class__.get_json_schema(entry["key"])

                # Check if using OpenAI models (gpt-*, o1-*, etc.)
                if (
                    self.model_name.startswith(("gpt-", "o1-", "openai/"))
                    or settings.model.mode == "openai"
                ):
                    openai_json_schema = self.__class__.get_openai_json_schema(
                        entry["key"]
                    )
                    # Use OpenAI's response_format
                    params["response_format"] = openai_json_schema

                    response = await self.async_client.chat.completions.parse(**params)

                    if hasattr(response.choices[0].message, "parsed"):
                        resp = json.dumps(
                            response.choices[0].message.parsed.model_dump()["spans"]
                        )
                        return resp

                else:
                    # Use vllm's guided_json
                    extra_body["guided_json"] = json_schema

            # params.update(self.api_kwargs)

            # Use self.async_client and await the API call
            response = await self.async_client.chat.completions.create(**params)

            # # Try to get parsed content first (OpenAI structured outputs)
            # if hasattr(response.choices[0].message, 'parsed') and response.choices[0].message.parsed is not None:
            #     return response.choices[0].message.parsed

            response = response.choices[0].message.content

            return response

        except Exception as e:
            # return ""
            raise ValueError(f"Error occurred while calling API: {e}")

    async def async_predict(self, entry: dict) -> Dict:
        """Main asynchronous method for concurrent execution"""
        # CPU-bound formatting is fast and fine to run synchronously
        entry["prompt"] = self.format_prompt(entry)

        # Await the I/O-bound API call
        try:
            response = await self.async_call_api(entry)

            if "<think>" in response and "</think>" in response:
                pattern = r"<think>(.*)<\/think>(.*)"
                matched = re.match(pattern, response, re.DOTALL)
                entry["reasoning"] = matched.group(1)
                entry["response"] = matched.group(2)
            else:
                entry["response"] = response
        except Exception as e:
            entry["response"] = ""
            entry["error"] = str(e)

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
