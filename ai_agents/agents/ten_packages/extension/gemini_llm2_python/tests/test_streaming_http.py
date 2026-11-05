"""Exercise incremental delivery and response cleanup through real HTTPX."""

import asyncio
from contextlib import suppress
import json
import unittest

import httpx

from test_completion_contract import CommandBridge, make_api
from ten_ai_base.struct import LLMRequest, LLMResponseMessageDone


class GatedStream(httpx.AsyncByteStream):
    def __init__(self, done_before_eof=False):
        self.first_sent = asyncio.Event()
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.done_before_eof = done_before_eof

    async def __aiter__(self):
        body = {
            "id": "synthetic-stream",
            "created": 1700000000,
            "choices": [{"delta": {"content": "First sentence."}}],
        }
        self.first_sent.set()
        yield ("data: " + json.dumps(body) + "\n\n").encode()
        if self.done_before_eof:
            yield b"data: [DONE]\n\n"
        self.waiting.set()
        await self.release.wait()
        if not self.done_before_eof:
            yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


class StreamingHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stream = GatedStream()
        self.requests = []

        async def handle(request):
            self.requests.append(request)
            return httpx.Response(200, stream=self.stream)

        self.client = httpx.AsyncClient(
            base_url="https://synthetic.invalid",
            transport=httpx.MockTransport(handle),
        )
        self.api = make_api(CommandBridge(), None)
        self.api.http_client = self.client
        self.generator = self.api.get_chat_completions(
            LLMRequest(request_id="synthetic", messages=[], streaming=True)
        )
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

    async def first_delta(self):
        self.pending = asyncio.create_task(anext(self.generator))
        await asyncio.wait_for(self.stream.first_sent.wait(), 1)
        return await asyncio.wait_for(asyncio.shield(self.pending), 0.5)

    async def test_first_delta_arrives_before_provider_eof(self):
        delta = await self.first_delta()
        self.assertEqual(delta.delta, "First sentence.")
        self.assertFalse(self.stream.release.is_set())
        self.assertFalse(self.stream.closed)
        self.stream.release.set()
        rest = [event async for event in self.generator]
        self.assertEqual(len(rest), 1)
        self.assertIsInstance(rest[0], LLMResponseMessageDone)
        self.assertEqual(rest[0].content, delta.content)
        self.assertTrue(self.stream.closed)
        self.assertTrue(json.loads(self.requests[0].content)["stream"])

    async def test_consumer_close_releases_response(self):
        await self.first_delta()
        await self.generator.aclose()
        self.assertTrue(self.stream.closed)

    async def test_abort_while_waiting_closes_response_and_next_request_works(
        self,
    ):
        await self.first_delta()
        self.pending = asyncio.create_task(anext(self.generator))
        await asyncio.wait_for(self.stream.waiting.wait(), 1)
        self.pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await self.pending
        self.assertTrue(self.stream.closed)
        self.stream = GatedStream()
        self.stream.release.set()
        fresh = self.api.get_chat_completions(
            LLMRequest(request_id="fresh", messages=[], streaming=True)
        )
        events = [event async for event in fresh]
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[-1], LLMResponseMessageDone)
        self.assertTrue(self.stream.closed)

    async def test_done_marker_does_not_wait_for_http_eof(self):
        self.stream.done_before_eof = True
        await self.first_delta()
        self.pending = asyncio.create_task(anext(self.generator))
        done = await asyncio.wait_for(asyncio.shield(self.pending), 0.5)
        self.assertIsInstance(done, LLMResponseMessageDone)
        await self.generator.aclose()
        self.assertTrue(self.stream.closed)
        self.assertFalse(self.stream.release.is_set())

    async def test_http_error_body_is_read_and_response_closed(self):
        self.stream.release.set()
        await self.client.aclose()

        async def handle(_request):
            return httpx.Response(503, stream=self.stream)

        self.client = httpx.AsyncClient(
            base_url="https://synthetic.invalid",
            transport=httpx.MockTransport(handle),
        )
        self.api.http_client = self.client
        with self.assertRaisesRegex(RuntimeError, "Gemini API error: 503"):
            await anext(self.generator)
        self.assertTrue(self.stream.closed)

    async def test_nonstream_http_response_still_emits_full_text_and_done(self):
        await self.client.aclose()
        body = {
            "id": "synthetic-full",
            "created": 1700000000,
            "choices": [{"message": {"content": "Complete answer."}}],
        }

        async def handle(_request):
            return httpx.Response(200, json=body)

        self.client = httpx.AsyncClient(
            base_url="https://synthetic.invalid",
            transport=httpx.MockTransport(handle),
        )
        self.api.http_client = self.client
        generator = self.api.get_chat_completions(
            LLMRequest(request_id="full", messages=[], streaming=False)
        )
        events = [event async for event in generator]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].delta, "Complete answer.")
        self.assertIsInstance(events[-1], LLMResponseMessageDone)
        self.assertEqual(events[-1].content, "Complete answer.")
