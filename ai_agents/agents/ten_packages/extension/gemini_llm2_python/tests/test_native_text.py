"""Native text streaming through the real HTTPX/response lifetime seam."""

import asyncio
from contextlib import suppress
import json
import unittest
from unittest.mock import patch

import httpx

from test_completion_contract import CommandBridge, gemini, make_api, run_bridge
from ten_ai_base.struct import LLMMessageContent, LLMRequest
from ten_ai_base.types import LLMToolMetadata


class NativeStream(httpx.AsyncByteStream):
    def __init__(self):
        self.release = asyncio.Event()
        self.waiting = asyncio.Event()
        self.closed = False
        self.finish = "STOP"
        self.eof = False

    async def __aiter__(self):
        yield self.frame({"parts": [{"text": "private", "thought": True}]})
        yield self.frame({"parts": [{"text": "First sentence."}]})
        self.waiting.set()
        await self.release.wait()
        if not self.eof:
            yield self.frame({}, self.finish)
            # A terminal provider event must finish without waiting for EOF.
            await asyncio.Event().wait()

    @staticmethod
    def frame(content, finish=None):
        candidate = {"content": content}
        if finish:
            candidate["finishReason"] = finish
        body = {"responseId": "native-fixture", "candidates": [candidate]}
        return ("data: " + json.dumps(body) + "\n\n").encode()

    async def aclose(self):
        self.closed = True


class NativeTextTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stream = NativeStream()
        self.requests = []

        async def handle(request):
            self.requests.append(request)
            return httpx.Response(200, stream=self.stream)

        self.client = httpx.AsyncClient(
            base_url="https://synthetic.invalid/v1beta",
            transport=httpx.MockTransport(handle),
        )
        self.api = make_api(CommandBridge(), None)
        self.api.config = gemini.GeminiLLM2Config.model_validate_json(
            json.dumps(
                {
                    "api_key": "fixture-key",
                    "model": "gemini-2.5-flash-lite",
                    "native_text_api": True,
                    "reasoning_effort": "none",
                    "prompt": "Synthetic system instruction",
                }
            )
        )
        self.api.http_client = self.client
        self.request = LLMRequest(
            request_id="native",
            messages=[
                LLMMessageContent(role="user", content="Remember blue."),
                LLMMessageContent(role="assistant", content="Blue."),
                LLMMessageContent(role="user", content="Which color?"),
            ],
            streaming=True,
        )
        self.generator = self.api.get_chat_completions(self.request)
        self.pending = None

    async def asyncTearDown(self):
        self.stream.release.set()
        if self.pending and not self.pending.done():
            self.pending.cancel()
        if self.pending:
            with suppress(asyncio.CancelledError, Exception):
                await self.pending
        await self.generator.aclose()
        await self.client.aclose()

    async def first(self):
        self.pending = asyncio.create_task(anext(self.generator))
        return await asyncio.wait_for(asyncio.shield(self.pending), 0.5)

    async def test_native_wire_and_incremental_text_before_eof(self):
        first = await self.first()
        self.assertEqual(first.delta, "First sentence.")
        self.assertNotIn("private", first.content)
        request = self.requests[0]
        self.assertEqual(
            str(request.url),
            "https://synthetic.invalid/v1beta/models/"
            "gemini-2.5-flash-lite:streamGenerateContent?alt=sse",
        )
        self.assertEqual(request.headers["x-goog-api-key"], "fixture-key")
        self.assertNotIn("authorization", request.headers)
        body = json.loads(request.content)
        self.assertEqual(
            [c["role"] for c in body["contents"]], ["user", "model", "user"]
        )
        self.assertEqual(
            body["generationConfig"]["thinkingConfig"], {"thinkingBudget": 0}
        )
        self.assertNotIn("fixture-key", request.content.decode())
        self.assertFalse(self.stream.closed)
        self.stream.release.set()
        done = await asyncio.wait_for(anext(self.generator), 0.5)
        self.assertEqual(done.content, "First sentence.")
        self.assertIsInstance(done.created, int)
        with self.assertRaises(StopAsyncIteration):
            await anext(self.generator)
        self.assertTrue(self.stream.closed)

    async def test_consumer_close_and_cancel_release_native_response(self):
        await self.first()
        await self.generator.aclose()
        self.assertTrue(self.stream.closed)
        self.stream = NativeStream()
        self.generator = self.api.get_chat_completions(self.request)
        await self.first()
        self.pending = asyncio.create_task(anext(self.generator))
        await asyncio.wait_for(self.stream.waiting.wait(), 0.5)
        self.pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await self.pending
        self.assertTrue(self.stream.closed)
        self.stream = NativeStream()
        self.stream.release.set()
        fresh = self.api.get_chat_completions(self.request)
        events = await asyncio.wait_for(self.collect(fresh), 0.5)
        self.assertEqual(len(events), 2)

    @staticmethod
    async def collect(generator):
        return [e async for e in generator]

    async def test_incomplete_or_blocked_stream_is_not_normal_completion(self):
        for premature, reason in ((True, "STOP"), (False, "SAFETY")):
            with self.subTest(premature=premature, reason=reason):
                self.stream = NativeStream()
                self.stream.eof, self.stream.finish = premature, reason
                self.stream.release.set()
                self.generator = self.api.get_chat_completions(self.request)
                await self.first()
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(anext(self.generator), 0.5)
                self.assertTrue(self.stream.closed)

    async def test_nonstream_native_response_and_parameter_override(self):
        await self.client.aclose()

        async def handle(request):
            self.requests.append(request)
            return httpx.Response(
                200,
                json={
                    "responseId": "full-fixture",
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "Blue."}]},
                            "finishReason": "STOP",
                        }
                    ],
                },
            )

        self.client = httpx.AsyncClient(
            base_url="https://synthetic.invalid/v1beta",
            transport=httpx.MockTransport(handle),
        )
        self.api.http_client = self.client
        self.request.streaming = False
        self.request.parameters = {"temperature": 0.2, "max_tokens": 123}
        self.generator = self.api.get_chat_completions(self.request)
        events = await self.collect(self.generator)
        self.assertEqual([e.content for e in events], ["Blue.", "Blue."])
        self.assertTrue(self.requests[0].url.path.endswith(":generateContent"))
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], 123)
        self.assertEqual(body["generationConfig"]["temperature"], 0.2)

    async def test_unsupported_parameters_fail_before_network(self):
        self.request.parameters = {"unknown_option": True}
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(anext(self.generator), 0.5)
        self.assertEqual(self.requests, [])

    async def test_tools_fail_before_network(self):
        self.request.tools = [
            LLMToolMetadata(
                name="fixture", description="fixture", parameters=[]
            )
        ]
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(anext(self.generator), 0.5)
        self.assertEqual(self.requests, [])

    async def test_native_completion_reaches_main_and_seals_tts(self):
        self.stream.release.set()
        with patch("test_completion_contract.make_api", return_value=self.api):
            bridge, callbacks = await run_bridge(None)
        self.assertEqual(bridge.rejected, [])
        self.assertEqual(bridge.errors, [])
        self.assertEqual(callbacks[-1], ("", "First sentence.", True))
        self.assertEqual(
            [event["text_input_end"] for event in bridge.tts_inputs],
            [False, True],
        )
        self.assertTrue(self.stream.closed)
