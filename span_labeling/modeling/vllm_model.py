from openai import OpenAI, AsyncOpenAI
from span_labeling.modeling.model import Model


class VLLMModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI(
            base_url=config.model.base_url,
            organization=config.model.organization,
            api_key=config.model.api_key,
            timeout=6000,
        )
        self.async_client = AsyncOpenAI(
            base_url=config.model.base_url,
            organization=config.model.organization,
            api_key=config.model.api_key,
            timeout=6000,
        )

    def resolve_parameters(self, entry: dict) -> dict:
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": self.config.model.enable_thinking
            },
        }

        if self.config.method.constrained:
            extra_body["vllm_xargs"] = {
                "input_text": entry["text"],
                "constrained_key": "text",
                "allowed_labels": entry["allowed_labels"],
                "label_key": "label",
            }

        params = {
            "extra_body": extra_body,
            "max_tokens": self.config.model.max_completion_tokens,
            "seed": self.config.seed,
        }

        if self.config.model.temperature is not None:
            params["temperature"] = self.config.model.temperature
        if self.config.model.top_p is not None:
            params["top_p"] = self.config.model.top_p
        if self.config.model.top_k is not None:
            params["extra_body"]["top_k"] = self.config.model.top_k

        # Add structured outputs schema if enabled
        if (
            self.config.method.use_structured_outputs
            and entry.get("response_format") is not None
        ):
            extra_body["structured_outputs"] = {"json": entry["response_format"]}

        return params

    def call_api(self, entry: dict) -> str:
        try:
            params = self.resolve_parameters(entry)
            params["model"] = self.config.model.name
            params["messages"] = [
                {"role": "system", "content": self.config.model.system_prompt},
                {"role": "user", "content": entry["prompt"]},
            ]

            response = self.client.chat.completions.create(**params)
            return response

        except Exception as e:
            print(f"Error during vLLM API call: {e}")
            return ""

    async def async_call_api(self, entry: dict) -> str:
        try:
            params = self.resolve_parameters(entry)
            params["model"] = self.config.model.name
            params["messages"] = [
                {"role": "system", "content": self.config.model.system_prompt},
                {"role": "user", "content": entry["prompt"]},
            ]

            response = await self.async_client.chat.completions.create(**params)
            return response

        except Exception as e:
            print(f"Error during async vLLM API call: {e}")
            return ""
