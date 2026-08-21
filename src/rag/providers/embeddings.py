from __future__ import annotations

import hashlib
import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbeddingProvider:
    """Stable feature-hash embeddings; no model or network access."""
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.lower().split():
                idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions
                vector[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            out.append([x / norm for x in vector])
        return out


class AzureOpenAIEmbeddingProvider:
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

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [x.embedding for x in self.client.embeddings.create(input=texts, model=self.deployment).data]
