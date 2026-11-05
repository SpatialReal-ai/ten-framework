"""Close provider streams when the real TEN consumer exits or is aborted."""

# These boundary regressions deliberately invoke the real consumer and inspect
# its per-invocation ownership scope rather than a mirrored implementation.
# pylint: disable=protected-access

import asyncio
from contextlib import suppress
import sys
import types
import unittest

import httpx

from test_completion_contract import (
    CommandBridge,
    EXTENSION,
    load_module,
    make_api,
)
from test_streaming_http import GatedStream
from ten_ai_base.struct import LLMRequest
from ten_runtime.cmd import Cmd


PACKAGE_NAME = "lifecycle_test_gemini"
PACKAGE = types.ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(EXTENSION)]
sys.modules[PACKAGE_NAME] = PACKAGE
GEMINI_EXTENSION = load_module(
    PACKAGE_NAME + ".extension", EXTENSION / "extension.py"
)


class GatedResults(CommandBridge):
    def __init__(self):
        super().__init__()
        self.delta_entered = asyncio.Event()
        self.final_entered = asyncio.Event()
        self.delta_release = asyncio.Event()
        self.final_release = asyncio.Event()

    async def return_result(self, result):
        if result.is_final():
            self.final_entered.set()
            await self.final_release.wait()
        else:
            self.delta_entered.set()
            await self.delta_release.wait()
        return None


class StreamLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.streams = []
        self.tasks = []

        async def handle(_request):
            stream = GatedStream()
            self.streams.append(stream)
            return httpx.Response(200, stream=stream)

        self.client = httpx.AsyncClient(
            base_url="https://synthetic.invalid",
            transport=httpx.MockTransport(handle),
        )
        self.provider = GEMINI_EXTENSION.GeminiLLM2Extension(
            "synthetic-lifecycle"
        )
        api = make_api(CommandBridge(), None)
        api.http_client = self.client
        self.provider.client = api

    async def asyncTearDown(self):
        for task, bridge in self.tasks:
            bridge.delta_release.set()
            bridge.final_release.set()
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        for stream in self.streams:
            stream.release.set()
        await self.client.aclose()

    async def start_consumer(self, request_id="synthetic-request"):
        bridge = GatedResults()
        request = LLMRequest(request_id=request_id, messages=[], streaming=True)

        async def consume():
            await self.provider._run_stream(
                bridge, Cmd.create("chat_completion"), request
            )
            return self.provider._stream_scope.get()

        task = asyncio.create_task(consume())
        self.tasks.append((task, bridge))
        await asyncio.wait_for(bridge.delta_entered.wait(), 1)
        return task, bridge, self.streams[-1]

    async def abort_consumer(self, task, bridge):
        task.cancel()
        await asyncio.wait_for(bridge.final_entered.wait(), 1)
        # The registry base may still await its cancellation-final ACK. The
        # extension promises deterministic cleanup before task completion.
        bridge.final_release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)

    async def test_abort_during_return_result_closes_without_gc(self):
        task, bridge, stream = await self.start_consumer()
        await self.abort_consumer(task, bridge)
        self.assertTrue(stream.closed)

    async def test_same_request_id_concurrent_tasks_have_separate_owners(self):
        first, first_bridge, first_stream = await self.start_consumer("same")
        second, second_bridge, second_stream = await self.start_consumer("same")
        await self.abort_consumer(first, first_bridge)
        self.assertTrue(first_stream.closed)
        self.assertFalse(second_stream.closed)
        self.assertFalse(second.done())
        await self.abort_consumer(second, second_bridge)
        self.assertTrue(second_stream.closed)

    async def test_normal_completion_closes_and_resets_scope(self):
        task, bridge, stream = await self.start_consumer()
        stream.release.set()
        bridge.delta_release.set()
        bridge.final_release.set()
        scope_after = await asyncio.wait_for(task, 1)
        self.assertTrue(stream.closed)
        self.assertIsNone(scope_after)

        # A later call on the same extension cannot inherit the closed scope.
        fresh, fresh_bridge, fresh_stream = await self.start_consumer()
        self.assertFalse(fresh_stream.closed)
        await self.abort_consumer(fresh, fresh_bridge)
        self.assertTrue(fresh_stream.closed)
