"""Check optional reasoning configuration at the actual HTTPX request seam."""

import json
import unittest

import httpx

from test_completion_contract import CommandBridge, gemini, make_api
from ten_ai_base.struct import LLMRequest


class ReasoningEffortTests(unittest.IsolatedAsyncioTestCase):
    async def request_body(self, config_values=None, parameters=None):
        bodies = []

        async def handle(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"data: [DONE]\n\n",
            )

        api = make_api(CommandBridge(), None)
        api.config = gemini.GeminiLLM2Config.model_validate_json(
            json.dumps({"model": "synthetic-model", **(config_values or {})})
        )
        async with httpx.AsyncClient(
            base_url="https://synthetic.invalid",
            transport=httpx.MockTransport(handle),
        ) as client:
            api.http_client = client
            request = LLMRequest(
                request_id="synthetic",
                messages=[],
                streaming=True,
                parameters=parameters or {},
            )
            _ = [event async for event in api.get_chat_completions(request)]
        self.assertEqual(len(bodies), 1)
        return bodies[0]

    async def test_default_and_explicit_null_omit_reasoning_effort(self):
        for config in ({}, {"reasoning_effort": None}):
            with self.subTest(config=config):
                body = await self.request_body(config)
                self.assertNotIn("reasoning_effort", body)

    async def test_configured_none_is_sent_to_provider(self):
        body = await self.request_body({"reasoning_effort": "none"})
        self.assertEqual(body.get("reasoning_effort"), "none")

    async def test_request_parameter_overrides_configured_reasoning(self):
        body = await self.request_body(
            {"reasoning_effort": "none"},
            {"reasoning_effort": "high", "model": "blocked-model"},
        )
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(body["model"], "synthetic-model")
