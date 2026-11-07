#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#
"""Gemini -> LLM2 command stream -> real main_python/TTS completion regression.

Run with the TEN runtime and ten_ai_base Python interfaces on PYTHONPATH.
TEN_TEST_MAIN_ROOT optionally selects an installed main_python package; by
default the tracked voice-assistant example in this repository is used.
"""

from contextlib import asynccontextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

# Exercise real producer/consumer internals at the native command boundary.
# pylint: disable=protected-access

import ten_ai_base
from ten_ai_base.llm2 import AsyncLLM2BaseExtension
from ten_ai_base.struct import LLMMessageContent, LLMRequest


EXTENSION = Path(__file__).resolve().parents[1]
AGENTS = EXTENSION.parents[2]
MAIN_ROOT = Path(
    os.environ.get(
        "TEN_TEST_MAIN_ROOT",
        str(
            AGENTS
            / "examples/voice-assistant/tenapp/ten_packages/extension/main_python"
        ),
    )
)
BASE_ROOT = Path(ten_ai_base.__file__).resolve().parents[2]
RESULT_SCHEMA = next(
    command["result"]["property"]["properties"]
    for command in json.loads(
        (BASE_ROOT / "api/llm-interface.json").read_text(encoding="utf-8")
    )["cmd_in"]
    if command["name"] == "chat_completion"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load real implementation files without executing native addon registration.
for package_name, directory in (
    ("completion_test_main", MAIN_ROOT),
    ("completion_test_main.agent", MAIN_ROOT / "agent"),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package

gemini = load_module("completion_test_gemini", EXTENSION / "gemini.py")
llm_exec = importlib.import_module("completion_test_main.agent.llm_exec")
main_control = importlib.import_module("completion_test_main.extension")
events = importlib.import_module("completion_test_main.agent.events")


class SyntheticResponse:
    status_code = 200

    def __init__(self, chunks=None, body=None):
        self.chunks = chunks or []
        self.body = body or {}

    async def aiter_lines(self):
        for chunk in self.chunks:
            yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"

    async def aread(self):
        return json.dumps(self.body).encode()

    def json(self):
        return self.body


class SyntheticHTTPClient:
    def __init__(self, response):
        self.response = response

    @asynccontextmanager
    async def stream(self, *_args, **_kwargs):
        yield self.response


class SchemaRejection:
    """TEN return_result reports schema errors as return values, not raises."""


class CommandBridge:
    """Fake transport checks the installed TEN schema before delivering results."""

    def __init__(self):
        self.provider = None
        self.accepted = []
        self.rejected = []
        self.results = []
        self.tts_inputs = []
        self.errors = []

    def log_info(self, _message):
        pass

    def log_debug(self, _message):
        pass

    def log_error(self, _message):
        self.errors.append("producer_error")

    @staticmethod
    def invalid_fields(body):
        invalid = []
        for name, field in RESULT_SCHEMA.items():
            if name not in body:
                continue
            if field["type"] == "int64" and (
                not isinstance(body[name], int) or isinstance(body[name], bool)
            ):
                invalid.append(name)
            if field["type"] == "string" and not isinstance(body[name], str):
                invalid.append(name)
        return invalid

    async def return_result(self, result):
        payload, _ = result.get_property_to_json(None)
        body = json.loads(payload) if payload else {}
        invalid = self.invalid_fields(body)
        if invalid:
            self.rejected.append({"type": body.get("type"), "fields": invalid})
            return SchemaRejection()
        self.accepted.append(body)
        self.results.append(result)
        return None

    async def send_cmd_ex(self, command):
        payload, _ = command.get_property_to_json(None)
        request = LLMRequest.model_validate_json(payload)
        await AsyncLLM2BaseExtension._run_stream(
            self.provider, self, command, request
        )
        for result in self.results:
            yield result, None

    async def send_data(self, data):
        if data.get_name() == "tts_text_input":
            payload, _ = data.get_property_to_json(None)
            self.tts_inputs.append(json.loads(payload))
        return None


def make_api(bridge, response):
    api = object.__new__(gemini.GeminiChatAPI)
    api.ten_env = bridge
    api.config = types.SimpleNamespace(
        prompt="Synthetic fixture",
        model="synthetic-fixture",
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        reasoning_effort=None,
        is_black_list_params=lambda _key: False,
    )
    api.http_client = SyntheticHTTPClient(response)
    api._request_headers = {}
    api._request_params = {}
    return api


def chunk(content, **metadata):
    return {
        "id": "synthetic-response",
        "choices": [{"delta": {"content": content}}],
        **metadata,
    }


async def discard_transcript(*_args, **_kwargs):
    pass


async def run_bridge(response):
    bridge = CommandBridge()
    api = make_api(bridge, response)
    bridge.provider = types.SimpleNamespace(
        on_call_chat_completion=lambda _env, request: api.get_chat_completions(
            request
        )
    )
    controller = types.SimpleNamespace(
        ten_env=bridge,
        sentence_fragment="",
        turn_id=1,
        session_id="synthetic-session",
        _send_transcript=discard_transcript,
    )
    controller._current_metadata = types.MethodType(
        main_control.MainControlExtension._current_metadata, controller
    )
    controller._send_to_tts = types.MethodType(
        main_control.MainControlExtension._send_to_tts, controller
    )

    callbacks = []

    async def on_response(_env, delta, text, final):
        callbacks.append((delta, text, final))
        event = events.LLMResponseEvent(delta=delta, text=text, is_final=final)
        await main_control.MainControlExtension._on_llm_response(
            controller, event
        )

    executor = object.__new__(llm_exec.LLMExec)
    executor.ten_env = bridge
    executor.contexts = []
    executor.available_tools = []
    executor.current_request_id = None
    executor.current_text = None
    executor._request_generation = 0
    executor.tool_registry = {}
    executor.on_response = on_response
    executor.on_reasoning_response = None
    await executor._send_to_llm(
        bridge, LLMMessageContent(role="user", content="Synthetic prompt")
    )
    return bridge, callbacks


class TestCompletionContract(unittest.IsolatedAsyncioTestCase):
    async def test_stream_reaches_real_consumer_and_ends_tts(self):
        response = SyntheticResponse(
            chunks=[
                chunk("First sentence. ", created=1700000000),
                chunk("Second sentence.", created=1700000000),
            ]
        )
        bridge, callbacks = await run_bridge(response)
        self.assertEqual(bridge.rejected, [])
        self.assertEqual(bridge.errors, [])
        self.assertEqual(
            callbacks,
            [
                ("First sentence. ", "First sentence. ", False),
                (
                    "Second sentence.",
                    "First sentence. Second sentence.",
                    False,
                ),
                ("", "First sentence. Second sentence.", True),
            ],
        )
        self.assertEqual(
            [event["text_input_end"] for event in bridge.tts_inputs],
            [False, False, True],
        )
        done = bridge.accepted[-2]
        self.assertEqual(done["type"], "message_content_done")
        self.assertEqual(done["created"], 1700000000)
        self.assertEqual(done["response_id"], "synthetic-response")

    async def test_zero_provider_timestamp_is_preserved(self):
        bridge, callbacks = await run_bridge(
            SyntheticResponse(chunks=[chunk("Synthetic.", created=0)])
        )
        self.assertEqual(bridge.rejected, [])
        self.assertTrue(callbacks[-1][2])
        self.assertEqual(
            [event["created"] for event in bridge.accepted[:-1]], [0, 0]
        )

    async def test_missing_and_null_created_use_current_seconds(self):
        for metadata in ({}, {"created": None}):
            with self.subTest(metadata=metadata), patch(
                "time.time", return_value=1800000000.75
            ):
                bridge, callbacks = await run_bridge(
                    SyntheticResponse(chunks=[chunk("Synthetic.", **metadata)])
                )
                self.assertEqual(bridge.rejected, [])
                self.assertTrue(callbacks[-1][2])
                for event in bridge.accepted[:-1]:
                    self.assertEqual(event["created"], 1800000000)

    async def test_non_integer_created_uses_current_seconds(self):
        for created in (True, "1700000000", 1700000000.5):
            with self.subTest(created=created), patch(
                "time.time", return_value=1800000000.75
            ):
                bridge, callbacks = await run_bridge(
                    SyntheticResponse(
                        chunks=[chunk("Synthetic.", created=created)]
                    )
                )
                self.assertEqual(bridge.rejected, [])
                self.assertTrue(callbacks[-1][2])
                for event in bridge.accepted[:-1]:
                    self.assertEqual(event["created"], 1800000000)

    async def test_empty_stream_has_valid_done_without_tts_text(self):
        with patch("time.time", return_value=1800000000.75):
            bridge, callbacks = await run_bridge(SyntheticResponse())
        self.assertEqual(bridge.rejected, [])
        self.assertEqual(bridge.accepted[0]["type"], "message_content_done")
        self.assertEqual(bridge.accepted[0]["created"], 1800000000)
        self.assertEqual(bridge.accepted[0]["content"], "")
        self.assertEqual(callbacks, [])
        self.assertEqual(bridge.tts_inputs, [])

    async def test_nonstream_delta_and_done_have_complete_text_and_metadata(
        self,
    ):
        for metadata in ({"created": 1700000000}, {}, {"created": None}):
            with self.subTest(metadata=metadata), patch(
                "time.time", return_value=1800000000.75
            ):
                response = SyntheticResponse(
                    body={
                        "id": "synthetic-response",
                        "choices": [{"message": {"content": "Synthetic."}}],
                        **metadata,
                    }
                )
                bridge = CommandBridge()
                api = make_api(bridge, response)
                request = LLMRequest(
                    request_id="synthetic-request",
                    messages=[],
                    streaming=False,
                )
                output = [
                    event async for event in api.get_chat_completions(request)
                ]
                self.assertEqual(len(output), 2)
                self.assertEqual(output[0].delta, "Synthetic.")
                self.assertEqual(output[0].content, "Synthetic.")
                self.assertEqual(output[1].content, "Synthetic.")
                expected = metadata.get("created") or 1800000000
                for event in output:
                    self.assertEqual(event.created, expected)
                    self.assertEqual(
                        bridge.invalid_fields(
                            json.loads(event.model_dump_json())
                        ),
                        [],
                    )


if __name__ == "__main__":
    unittest.main()
