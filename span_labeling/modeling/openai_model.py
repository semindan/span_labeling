from openai import OpenAI, AsyncOpenAI
from span_labeling.modeling.model import Model


class OpenAIModel(Model):
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
        params = {
            "max_completion_tokens": self.config.model.max_completion_tokens,
            "reasoning_effort": "minimal",
            "seed": self.config.seed,
        }

        if self.config.model.temperature is not None:
            params["temperature"] = self.config.model.temperature
        if self.config.model.top_p is not None:
            params["top_p"] = self.config.model.top_p
        if self.config.model.top_k is not None:
            params["top_k"] = self.config.model.top_k

        if (
            hasattr(self.config.method, "use_structured_outputs")
            and self.config.method.use_structured_outputs
            and entry.get("response_format") is not None
        ):
            params["response_format"] = entry["response_format"]

        return params

    def call_api(self, entry: dict) -> str:
        try:
            params = self.resolve_parameters(entry)
            params["model"] = self.config.model.name
            params["messages"] = [
                {"role": "system", "content": self.config.model.system_prompt},
                {"role": "user", "content": entry["prompt"]},
            ]

            if (
                hasattr(self.config.method, "use_structured_outputs")
                and self.config.method.use_structured_outputs
            ):
                response = self.client.chat.completions.parse(**params)
                return response
            else:
                response = self.client.chat.completions.create(**params)
                return response

        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return ""

    async def async_call_api(self, entry: dict) -> str:
        try:
            params = self.resolve_parameters(entry)
            params["model"] = self.config.model.name
            params["messages"] = [
                {"role": "system", "content": self.config.model.system_prompt},
                {"role": "user", "content": entry["prompt"]},
            ]

            if self.config.method.use_structured_outputs:
                response = await self.async_client.chat.completions.parse(**params)
                return response
            else:
                response = await self.async_client.chat.completions.create(**params)
                return response

        except Exception as e:
            print(f"Error during async OpenAI API call: {e}")
            return ""
