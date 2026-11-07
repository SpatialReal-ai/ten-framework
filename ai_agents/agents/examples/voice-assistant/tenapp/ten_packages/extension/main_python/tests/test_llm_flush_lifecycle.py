"""Real Agent/LLMExec/Main cancellation, with only TEN transport controlled."""

# pylint: disable=protected-access

import asyncio
import json
import unittest
from unittest.mock import patch

from ten_runtime import StatusCode
from test_interrupt_routing import (
    FakeLoc,
    FakeMessage,
    FakeTransport,
    helper,
    main_control,
)


class Message(FakeMessage):
    def get_name(self):
        return self.name

    def get_property_to_json(self, _key):
        return json.dumps(self.payload), None


class Result:
    def __init__(self, body):
        self.body = body

    def is_final(self):
        return False

    def get_status_code(self):
        return StatusCode.OK

    def get_property_to_json(self, _key):
        return json.dumps(self.body), None


def response(done=False):
    return {
        "response_id": "synthetic-response",
        "created": 1,
        "role": "assistant",
        "type": "message_content_done" if done else "message_content_delta",
        "delta": "Synthetic sentence.",
        "content": "Synthetic sentence.",
    }


async def settle():
    for _ in range(40):
        await asyncio.sleep(0)


class Transport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.requests = []
        self.streams = {}
        self.tts = []
        self.abort_error = False
        self.abort_started = asyncio.Event()
        self.abort_release = asyncio.Event()
        self.abort_release.set()
        self.cancel_started = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.cancel_release.set()
        self.late_done_on_cancel = False
        self.block_tts_send = False
        self.tts_send_started = asyncio.Event()
        self.tts_send_cancelled = asyncio.Event()
        self.handler_cancel_release = asyncio.Event()
        self.handler_cancel_release.set()
        self.errors = []

    async def get_property_to_json(self, _key):
        return json.dumps({"greeting": ""}), None

    def log_error(self, _message):
        self.errors.append("error")

    def log_warn(self, _message):
        pass

    async def send_cmd(self, command):
        if command.name == "abort":
            self.abort_started.set()
            await self.abort_release.wait()
            if self.abort_error:
                raise RuntimeError("synthetic abort failure")
        return await super().send_cmd(command)

    async def send_cmd_ex(self, command):
        request = command.payload["request_id"]
        queue = self.streams[request] = asyncio.Queue()
        self.requests.append(request)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield Result(item), None
        except asyncio.CancelledError:
            self.cancel_started.set()
            await self.cancel_release.wait()
            if self.late_done_on_cancel:
                # Vendor callbacks may race abort or suppress task cancellation.
                yield Result(response(True)), None
            else:
                raise

    async def send_data(self, data):
        if data.name == "tts_text_input":
            if self.block_tts_send:
                self.tts_send_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.tts_send_cancelled.set()
                    await self.handler_cancel_release.wait()
                    raise
            self.tts.append(dict(data.payload))
        return await super().send_data(data)


class TestLLMFlushLifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        for name, value in (
            ("Cmd", Message),
            ("Data", Message),
            ("Loc", FakeLoc),
        ):
            patcher = patch.object(helper, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.env = Transport()
        self.controller = main_control.MainControlExtension("flush-test")
        await self.controller.on_init(self.env)

    async def asyncTearDown(self):
        self.env.abort_release.set()
        self.env.cancel_release.set()
        self.env.handler_cancel_release.set()
        await self.controller.on_stop(self.env)
        await settle()

    async def start_request(self):
        before = len(self.env.requests)
        await self.controller.agent.queue_llm_input("Synthetic user input")
        await settle()
        self.assertEqual(len(self.env.requests), before + 1)
        return self.env.requests[-1]

    async def delta(self, request):
        await self.env.streams[request].put(response())
        await settle()
        self.assertTrue(self.env.tts)

    async def finish(self, request):
        await self.env.streams[request].put(response(True))
        await self.env.streams[request].put(None)
        await settle()

    async def test_interrupt_never_resurrects_old_tts_with_cancelled_final(
        self,
    ):
        self.controller.turn_id = 5
        request = await self.start_request()
        await self.delta(request)
        self.assertEqual([t["text_input_end"] for t in self.env.tts], [False])
        await self.controller._interrupt()
        after_flush = len(self.env.tts)
        await settle()
        self.assertEqual(len(self.env.tts), after_flush)
        self.assertFalse(any(t["text_input_end"] for t in self.env.tts))
        self.controller.turn_id = 6
        following = await self.start_request()
        await self.delta(following)
        await self.finish(following)
        self.assertEqual(
            [t["request_id"] for t in self.env.tts if t["text_input_end"]],
            ["tts-request-6"],
        )
        self.assertIsNone(self.controller.agent.llm_exec.current_request_id)
        self.assertIsNone(self.controller.agent.llm_exec.current_text)
        self.assertEqual(self.env.errors, [])

    async def test_flush_waits_for_cancel_before_new_input_can_start(self):
        request = await self.start_request()
        await self.delta(request)
        self.env.cancel_release.clear()
        flush = asyncio.create_task(self.controller.agent.flush_llm())
        await settle()
        self.assertTrue(self.env.cancel_started.is_set())
        self.assertFalse(flush.done())
        following = asyncio.create_task(
            self.controller.agent.queue_llm_input("Next synthetic input")
        )
        await settle()
        self.assertEqual(len(self.env.requests), 1)
        self.env.cancel_release.set()
        await flush
        await following
        await settle()
        self.assertEqual(len(self.env.requests), 2)
        await self.delta(self.env.requests[-1])
        await self.finish(self.env.requests[-1])

    async def test_late_done_while_abort_is_pending_cannot_cross_flush(self):
        request = await self.start_request()
        await self.delta(request)
        self.env.abort_release.clear()
        self.env.late_done_on_cancel = True
        flush = asyncio.create_task(self.controller.agent.flush_llm())
        await settle()
        self.assertTrue(self.env.abort_started.is_set())
        # Old implementations clear request_id then await abort while continuing
        # to accept genuine provider Done events on that still-running request.
        await self.env.streams[request].put(response(True))
        await settle()
        self.assertFalse(any(t["text_input_end"] for t in self.env.tts))
        self.env.abort_release.set()
        await flush
        await settle()
        self.assertFalse(any(t["text_input_end"] for t in self.env.tts))

    async def test_fresh_input_waits_for_cancelled_main_handler_cleanup(self):
        request = await self.start_request()
        self.env.block_tts_send = True
        self.env.handler_cancel_release.clear()
        await self.env.streams[request].put(response())
        await settle()
        self.assertTrue(self.env.tts_send_started.is_set())
        flush = asyncio.create_task(self.controller.agent.flush_llm())
        await settle()
        self.assertTrue(self.env.tts_send_cancelled.is_set())
        self.assertFalse(flush.done())
        following = asyncio.create_task(
            self.controller.agent.queue_llm_input("Next input")
        )
        await settle()
        self.assertEqual(len(self.env.requests), 1)
        self.env.block_tts_send = False
        self.env.handler_cancel_release.set()
        await flush
        await following
        await settle()
        self.assertEqual(len(self.env.requests), 2)
        self.assertEqual(self.env.tts, [])
        await self.delta(self.env.requests[-1])
        await self.finish(self.env.requests[-1])
        self.assertEqual(
            [t["text_input_end"] for t in self.env.tts], [False, True]
        )

    async def test_abort_error_still_cleans_old_main_handler_before_new_input(
        self,
    ):
        request = await self.start_request()
        self.env.block_tts_send = True
        self.env.handler_cancel_release.clear()
        await self.env.streams[request].put(response())
        await settle()
        self.assertTrue(self.env.tts_send_started.is_set())
        self.env.abort_error = True
        flush = asyncio.create_task(self.controller.agent.flush_llm())
        await settle()
        try:
            self.assertTrue(self.env.tts_send_cancelled.is_set())
            self.assertFalse(flush.done())
            following = asyncio.create_task(
                self.controller.agent.queue_llm_input("Fresh after error")
            )
            await settle()
            self.assertEqual(len(self.env.requests), 1)
        finally:
            self.env.block_tts_send = False
            self.env.handler_cancel_release.set()
        with self.assertRaisesRegex(RuntimeError, "synthetic abort failure"):
            await flush
        self.env.abort_error = False
        await following
        await settle()
        self.assertEqual(self.env.tts, [])
        self.assertEqual(len(self.env.requests), 2)
        await self.delta(self.env.requests[-1])
        await self.finish(self.env.requests[-1])
        self.assertEqual(
            [t["text_input_end"] for t in self.env.tts], [False, True]
        )

    async def test_stop_cleans_all_workers_even_if_abort_transport_fails(self):
        request = await self.start_request()
        await self.delta(request)
        self.env.abort_error = True
        with self.assertRaisesRegex(RuntimeError, "synthetic abort failure"):
            await self.controller.on_stop(self.env)
        await settle()
        names = {
            "LLMExec._process_input_queue",
            "Agent._consume_asr",
            "Agent._consume_llm",
        }
        workers = [
            t
            for t in asyncio.all_tasks()
            if not t.done() and t.get_coro().__qualname__ in names
        ]
        self.assertEqual(workers, [])
        self.assertIsNone(self.controller.agent.llm_exec.current_task)
        self.assertIsNone(self.controller.agent.llm_exec.current_request_id)

    async def test_normal_done_still_finalizes_once_and_stop_owns_all_workers(
        self,
    ):
        request = await self.start_request()
        await self.delta(request)
        await self.finish(request)
        self.assertEqual(
            [t["text_input_end"] for t in self.env.tts], [False, True]
        )
        self.assertIsNone(self.controller.agent.llm_exec.current_request_id)
        await self.controller.on_stop(self.env)
        await settle()
        workers = [
            t
            for t in asyncio.all_tasks()
            if not t.done()
            and t.get_coro().__qualname__ == "LLMExec._process_input_queue"
        ]
        self.assertEqual(workers, [])
        await self.controller.agent.queue_llm_input(
            "Must not reopen after stop"
        )
        await settle()
        self.assertEqual(len(self.env.requests), 1)
