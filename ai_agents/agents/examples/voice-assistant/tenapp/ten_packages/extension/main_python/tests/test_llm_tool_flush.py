"""Real tool-result recursion across flush; only TEN transport is controlled."""

# pylint: disable=protected-access

import asyncio
import unittest
from unittest.mock import patch

from ten_ai_base.types import LLMToolMetadata
from test_llm_flush_lifecycle import (
    FakeLoc,
    Message,
    Result,
    Transport,
    helper,
    main_control,
    response,
    settle,
)


def tool_call():
    return {
        "response_id": "synthetic-tool-response",
        "created": 1,
        "role": "assistant",
        "type": "tool_call_content",
        "id": "synthetic-tool-item",
        "tool_call_id": "synthetic-call",
        "name": "synthetic_lookup",
        "arguments": {"value": 1},
    }


class ToolTransport(Transport):
    def __init__(self):
        super().__init__()
        self.llm_inputs = []
        self.tool_commands = []
        self.tool_started = asyncio.Event()
        self.tool_release = asyncio.Event()
        self.tool_release.set()
        self.tool_cancelled = asyncio.Event()
        self.suppress_tool_cancel = False

    # Return the existing generator unchanged, without an extra consumer layer.
    def send_cmd_ex(self, command):  # pylint: disable=invalid-overridden-method
        self.llm_inputs.append(command.payload)
        return super().send_cmd_ex(command)

    async def send_cmd(self, command):
        if command.name != "tool_call":
            return await super().send_cmd(command)
        self.tool_commands.append(command)
        self.tool_started.set()
        try:
            await self.tool_release.wait()
        except asyncio.CancelledError:
            self.tool_cancelled.set()
            if not self.suppress_tool_cancel:
                raise
            # Model a tool reply already in transport while flush invalidates A.
            await self.tool_release.wait()
        return (
            Result({"type": "llmresult", "content": "Synthetic tool result"}),
            None,
        )


class TestLLMToolFlush(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        for name, value in (
            ("Cmd", Message),
            ("Data", Message),
            ("Loc", FakeLoc),
        ):
            patcher = patch.object(helper, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.env = ToolTransport()
        self.controller = main_control.MainControlExtension("tool-flush-test")
        await self.controller.on_init(self.env)
        await self.controller.agent.register_llm_tool(
            LLMToolMetadata(
                name="synthetic_lookup",
                description="Return a synthetic fixture.",
                parameters=[],
            ),
            "synthetic_tool",
        )

    async def asyncTearDown(self):
        self.env.tool_release.set()
        self.env.abort_release.set()
        self.env.cancel_release.set()
        await self.controller.on_stop(self.env)
        await settle()

    async def start_request(self):
        count = len(self.env.requests)
        await self.controller.agent.queue_llm_input("Synthetic user input")
        await settle()
        self.assertEqual(len(self.env.requests), count + 1)
        return self.env.requests[-1]

    async def finish(self, request):
        await self.env.streams[request].put(response())
        await self.env.streams[request].put(response(True))
        await self.env.streams[request].put(None)
        await settle()

    async def test_tool_result_recurses_and_normal_following_request_finishes(
        self,
    ):
        self.controller.turn_id = 7
        outer = await self.start_request()
        await self.env.streams[outer].put(tool_call())
        await settle()
        self.assertTrue(self.env.tool_started.is_set())
        self.assertEqual(len(self.env.requests), 2)
        inner = self.env.requests[-1]
        self.assertNotEqual(inner, outer)
        self.assertEqual(
            self.controller.agent.llm_exec.current_request_id, inner
        )
        self.assertEqual(
            self.env.tool_commands[0].payload["name"], "synthetic_lookup"
        )
        self.assertEqual(
            self.env.tool_commands[0].payload["arguments"], {"value": 1}
        )
        tool_message = self.env.llm_inputs[-1]["messages"][-1]
        self.assertEqual(tool_message["type"], "function_call_output")
        self.assertEqual(tool_message["call_id"], "synthetic-call")
        self.assertEqual(tool_message["output"], "Synthetic tool result")
        await self.finish(inner)
        # The outer stream is still live. Nested completion must restore its
        # control identity, rather than leaving a stale inner request to abort.
        self.assertEqual(
            self.controller.agent.llm_exec.current_request_id, outer
        )
        await self.env.streams[outer].put(None)
        await settle()
        self.assertIsNone(self.controller.agent.llm_exec.current_request_id)
        self.assertEqual(
            [x["request_id"] for x in self.env.tts if x["text_input_end"]],
            ["tts-request-7"],
        )
        self.controller.turn_id = 8
        following = await self.start_request()
        await self.finish(following)
        self.assertEqual(
            [x["request_id"] for x in self.env.tts if x["text_input_end"]],
            ["tts-request-7", "tts-request-8"],
        )
        self.assertEqual(self.env.errors, [])

    async def test_tool_reply_after_flush_cannot_start_recursive_request(self):
        self.env.tool_release.clear()
        self.env.suppress_tool_cancel = True
        outer = await self.start_request()
        await self.env.streams[outer].put(tool_call())
        await settle()
        self.assertTrue(self.env.tool_started.is_set())
        self.env.abort_release.clear()
        flush = asyncio.create_task(self.controller.agent.flush_llm())
        await settle()
        self.assertFalse(flush.done())
        # Whether cancellation cleanup or abort ACK is awaited first, this
        # successful old tool result arrives after the flush barrier began.
        self.env.tool_release.set()
        await settle()
        self.assertEqual(len(self.env.requests), 1)
        self.assertEqual(self.env.tts, [])
        self.env.abort_release.set()
        await asyncio.wait_for(flush, timeout=1)
        await settle()
        self.controller.turn_id = 9
        following = await self.start_request()
        await self.finish(following)
        self.assertEqual(len(self.env.requests), 2)
        self.assertEqual(
            [x["request_id"] for x in self.env.tts if x["text_input_end"]],
            ["tts-request-9"],
        )
        self.assertIsNone(self.controller.agent.llm_exec.current_request_id)
        self.assertEqual(self.env.errors, [])


if __name__ == "__main__":
    unittest.main()
