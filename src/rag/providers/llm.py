from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

class LLMProvider(Protocol):
    def chat(self, messages: list[dict], **kwargs) -> LLMResult: ...

class LocalLLMProvider:
    def chat(self, messages: list[dict], **kwargs) -> LLMResult:
        return LLMResult("", 0, 0)

class AzureOpenAILLMProvider:
    def __init__(self, endpoint: str, deployment: str, api_key: str | None = None):
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError("Install the azure extra to use Azure providers") from exc
        kwargs = {"azure_endpoint": endpoint, "api_version": "2024-02-15-preview"}
        if api_key:
            kwargs["api_key"] = api_key
        else:
            kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        self.client, self.deployment = AzureOpenAI(**kwargs), deployment

    def chat(self, messages: list[dict], **kwargs) -> LLMResult:
        result = self.client.chat.completions.create(model=self.deployment, messages=messages,
                                                     temperature=0, **kwargs)
        usage = result.usage
        return LLMResult(result.choices[0].message.content or "", usage.prompt_tokens,
                         usage.completion_tokens)
