"""LLM adapters. Each returns schema-conforming JSON for a (system, prompt) pair."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from neuralstate.providers.base import LLMRefusal, ProviderError

log = logging.getLogger(__name__)

# A malformed JSON body is worth one retry; anything else is surfaced to the job runner,
# which has its own backoff.
_json_retry = retry(
    retry=retry_if_exception_type(json.JSONDecodeError),
    stop=stop_after_attempt(2),
    wait=wait_exponential(min=1, max=4),
    reraise=True,
)


class AnthropicLLM:
    name = "anthropic"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str | None = None, effort: str = "medium", max_tokens: int = 16000,
                 fallbacks: bool = True, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        self.client = client
        self.model = model or "claude-opus-5-5"
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks

    def request_kwargs(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        if self.fallbacks:
            kwargs["betas"] = [self.FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    @_json_retry
    async def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        kwargs = self.request_kwargs(system=system, prompt=prompt, schema=schema)
        # `fallbacks` is a beta parameter, so it is only accepted on the beta namespace.
        messages = self.client.beta.messages if self.fallbacks else self.client.messages
        response = await messages.create(**kwargs)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(f"model declined: {getattr(details, 'category', None)}")
        if response.stop_reason == "max_tokens":
            raise ProviderError("response truncated at max_tokens; raise NS_LLM_MAX_TOKENS")
        text = "".join(b.text for b in response.content if b.type == "text")
        return json.loads(text)


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str | None = None, base_url: str | None = None, max_tokens: int = 16000,
                 client: Any = None):
        if client is None:
            import openai

            client = openai.AsyncOpenAI(base_url=base_url)
        self.client = client
        self.model = model or "gpt-4.1-mini"
        self.max_tokens = max_tokens

    @_json_retry
    async def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema, "strict": True},
            },
        )
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMRefusal(choice.message.refusal)
        return json.loads(choice.message.content)


class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str | None = None, url: str = "http://localhost:11434", timeout: float = 300.0):
        self.model = model or "llama3.1"
        self.url = url.rstrip("/")
        self.timeout = timeout

    @_json_retry
    async def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.url}/api/chat", json={
                "model": self.model,
                "stream": False,
                "format": schema,
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            })
        if r.status_code >= 400:
            raise ProviderError(f"ollama {r.status_code}: {r.text[:300]}")
        return json.loads(r.json()["message"]["content"])
