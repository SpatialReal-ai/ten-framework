"""Provider WS event -> actual ASR Data JSON -> main turn admission."""

# pylint: disable=protected-access

import importlib
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock

from test_interrupt_routing import main_control

MAIN = Path(__file__).resolve().parents[1]
ASR_ROOT = Path(
    os.environ.get("TEN_TEST_ASR_ROOT")
    or MAIN.parents[5] / "ten_packages/extension/deepgram_asr_python"
)
PACKAGE = types.ModuleType("boundary_test_asr")
PACKAGE.__path__ = [str(ASR_ROOT)]
sys.modules[PACKAGE.__name__] = PACKAGE
extension = importlib.import_module("boundary_test_asr.extension")
recognition = importlib.import_module("boundary_test_asr.recognition")


class NovaBoundaryBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.output = []
        self.env = MagicMock()

        async def send_data(data):
            payload, error = data.get_property_to_json(None)
            self.assertIsNone(error)
            self.output.append(json.loads(payload))

        self.env.send_data = send_data
        self.asr = extension.DeepgramASRExtension("boundary-test")
        self.asr.ten_env = self.env
        self.asr.config = extension.DeepgramASRConfig.model_validate(
            {"params": {"model": "nova-3", "language": "en-US"}}
        )
        self.asr.metadata = {"session_id": "100", "custom": "fixture"}
        self.asr.audio_timeline.add_user_audio(1000)
        self.asr.audio_timeline.add_silence_audio(200)
        self.asr.audio_timeline.add_user_audio(1000)
        self.asr.sent_user_audio_duration_ms_before_last_reset = 3000
        self.wire = recognition.DeepgramASRRecognition(
            api_key="fixture",
            audio_timeline=self.asr.audio_timeline,
            ten_env=self.env,
            config={"model": "nova-3"},
            callback=self.asr,
        )
        self.main = main_control.MainControlExtension("boundary-consumer")
        self.main.ten_env = self.env
        self.main.agent = types.SimpleNamespace(queue_llm_input=AsyncMock())
        self.main._interrupt = AsyncMock()
        self.main._send_transcript = AsyncMock()

    @staticmethod
    def result(endpoint=False):
        return {
            "type": "Results",
            "start": 0,
            "duration": 2.2,
            "is_final": True,
            "speech_final": endpoint,
            "channel": {
                "alternatives": [
                    {
                        "transcript": "Complete synthetic question.",
                        "words": [
                            {"word": "question", "start": 1.3, "end": 1.5}
                        ],
                    }
                ]
            },
        }

    async def deliver(self, body):
        before = len(self.output)
        await self.wire._handle_message(json.dumps(body))
        for data in self.output[before:]:
            await self.main._on_asr_result(
                main_control.ASRResultEvent(
                    text=data["text"],
                    final=data["final"],
                    metadata=data["metadata"],
                )
            )

    async def test_real_wire_gap_releases_missing_endpoint_without_fake_final(
        self,
    ):
        await self.deliver(self.result())
        self.main.agent.queue_llm_input.assert_not_awaited()
        gap = {"type": "UtteranceEnd", "last_word_end": 1.5, "channel": [0, 1]}
        await self.deliver(gap)
        self.assertEqual(len(self.output), 2)
        event = self.output[-1]
        self.assertEqual(event["text"], "")
        self.assertFalse(event["final"])
        self.assertEqual(event["start_ms"], 4300)
        self.assertEqual(event["metadata"]["session_id"], "100")
        self.assertEqual(
            event["metadata"]["asr_info"]["last_word_end_ms"], 4300
        )
        self.assertEqual(
            event["metadata"]["asr_info"]["message_type"], "UtteranceEnd"
        )
        self.main.agent.queue_llm_input.assert_awaited_once_with(
            "Complete synthetic question."
        )
        await self.deliver(gap)
        self.assertEqual(self.main.agent.queue_llm_input.await_count, 1)
        self.assertEqual(
            self.asr.metadata, {"session_id": "100", "custom": "fixture"}
        )

    async def test_normal_endpoint_and_invalid_provider_marker(self):
        await self.deliver(self.result(endpoint=True))
        await self.deliver({"type": "UtteranceEnd", "last_word_end": -1})
        self.assertEqual(len(self.output), 1)
        await self.deliver({"type": "UtteranceEnd", "last_word_end": 1.5})
        self.assertEqual(self.main.agent.queue_llm_input.await_count, 1)

    async def test_flux_does_not_accept_nova_boundary_events(self):
        self.wire.is_flux_model = True
        await self.deliver({"type": "UtteranceEnd", "last_word_end": 1.5})
        self.assertEqual(self.output, [])
