from types import SimpleNamespace

import numpy as np
import pytest

from neuralstate.providers.base import LLMRefusal, ProviderError, check_dims
from neuralstate.providers.embed import HashEmbedder
from neuralstate.providers.llm import AnthropicLLM

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
          "additionalProperties": False}


class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _client(response):
    msgs = FakeMessages(response)
    return SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs)), msgs


def _resp(text='{"ok": true}', stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, stop_details=None,
                           content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])


async def test_anthropic_request_shape_with_fallbacks():
    client, msgs = _client(_resp())
    llm = AnthropicLLM(client=client, effort="low")
    assert await llm.complete_json(system="sys", prompt="hi", schema=SCHEMA) == {"ok": True}
    kw = msgs.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in kw and "temperature" not in kw


async def test_anthropic_without_fallbacks_uses_stable_namespace():
    client, msgs = _client(_resp())
    client.beta = None  # would raise if touched
    await AnthropicLLM(client=client, fallbacks=False).complete_json(system="s", prompt="p", schema=SCHEMA)
    assert "betas" not in msgs.kwargs and "fallbacks" not in msgs.kwargs


async def test_anthropic_refusal_and_truncation_raise():
    client, _ = _client(_resp(stop="refusal"))
    with pytest.raises(LLMRefusal):
        await AnthropicLLM(client=client).complete_json(system="s", prompt="p", schema=SCHEMA)
    client, _ = _client(_resp(stop="max_tokens"))
    with pytest.raises(ProviderError):
        await AnthropicLLM(client=client).complete_json(system="s", prompt="p", schema=SCHEMA)


async def test_anthropic_retries_once_on_bad_json():
    calls = []

    class Flaky:
        async def create(self, **kw):
            calls.append(1)
            return _resp("not json" if len(calls) == 1 else '{"ok": false}')

    client = SimpleNamespace(beta=SimpleNamespace(messages=Flaky()))
    llm = AnthropicLLM(client=client)
    assert await llm.complete_json(system="s", prompt="p", schema=SCHEMA) == {"ok": False}
    assert len(calls) == 2


async def test_hash_embedder_is_unit_length_and_meaningful():
    e = HashEmbedder(dim=128)
    a, b, c = await e.embed(["deploy the storefront for acme", "deploy the storefront for globex",
                             "refund a duplicated invoice"])
    for v in (a, b, c):
        assert abs(np.linalg.norm(v) - 1) < 1e-9 and len(v) == 128
    assert np.dot(a, b) > np.dot(a, c)
    assert (await e.embed(["same"]))[0] == (await e.embed(["same"]))[0]


def test_check_dims_explains_mismatch():
    with pytest.raises(ProviderError, match="NS_EMBED_DIM"):
        check_dims([[0.0] * 10], 384, "x")


def test_registry_builds_offline_providers():
    from neuralstate.config import Settings
    from neuralstate.providers.registry import build_providers

    p = build_providers(Settings(_env_file=None, llm_provider="fake", embed_provider="fake"))
    assert p.reasoner.name == "heuristic" and p.embedder.name == "fake"
