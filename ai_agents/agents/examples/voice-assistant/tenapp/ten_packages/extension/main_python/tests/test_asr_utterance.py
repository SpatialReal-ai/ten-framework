"""Offline Nova boundaries through the real main-controller handler/helpers."""

# pylint: disable=protected-access

import unittest
from unittest.mock import patch

from test_interrupt_routing import (
    FakeLoc,
    FakeMessage,
    FakeTransport,
    MainControlConfig,
    helper,
    main_control,
)

ASRResultEvent = main_control.ASRResultEvent
FIRST = "first stable part "  # 18 characters, a synthetic finalized segment.
SECOND = "second segment."  # 15 characters, the next segment of that utterance.


class Transport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.messages = []
        self.warnings = []

    async def send_data(self, data):
        self.messages.append((data.name, data.payload))
        return await super().send_data(data)

    def log_warn(self, message):
        self.warnings.append(message)


class Agent:
    def __init__(self, transport):
        self.transport = transport
        self.inputs = []
        self.stopped = False

    async def flush_llm(self):
        self.transport.events.append(("llm_flush",))

    async def queue_llm_input(self, text):
        self.inputs.append(text)
        self.transport.events.append(("llm_input",))

    async def stop(self):
        self.stopped = True


def nova(text, final=False, endpoint=False, session="100"):
    return ASRResultEvent(
        text=text,
        final=final,
        metadata={
            "session_id": session,
            "asr_info": {
                "vendor": "deepgram",
                "api": "listen_v1",
                "message_type": "Results",
                "is_final": final,
                "speech_final": endpoint,
            },
        },
    )


def timed_nova(text, word_end, final=False, endpoint=False):
    event = nova(text, final, endpoint)
    event.metadata["asr_info"]["last_word_end_ms"] = word_end
    return event


def utterance_end(word_end):
    event = timed_nova("", word_end)
    event.metadata["asr_info"]["message_type"] = "UtteranceEnd"
    return event


class TestASRUtterance(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.transport = Transport()
        self.controller = main_control.MainControlExtension("asr-test")
        self.controller.ten_env = self.transport
        self.controller.config = MainControlConfig(greeting="")
        self.controller.agent = Agent(self.transport)
        for name, value in (
            ("Cmd", FakeMessage),
            ("Data", FakeMessage),
            ("Loc", FakeLoc),
        ):
            patcher = patch.object(helper, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def send(self, *events):
        for event in events:
            await self.controller._on_asr_result(event)

    def flushes(self):
        return self.transport.events.count(("llm_flush",))

    async def test_missing_speech_final_uses_matching_provider_gap_once(self):
        await self.send(timed_nova(FIRST, 1000, final=True))
        self.assertEqual(self.controller.agent.inputs, [])
        await self.send(utterance_end(1000), utterance_end(1000))
        await self.send(nova("", True, True))
        self.assertEqual(self.controller.agent.inputs, [FIRST.strip()])
        self.assertEqual(self.controller.turn_id, 1)

    async def test_late_gap_does_not_cut_newer_interim_or_final_words(self):
        await self.send(
            timed_nova(FIRST, 1000, final=True),
            timed_nova("second", 1500),
            utterance_end(1000),
        )
        self.assertEqual(self.controller.agent.inputs, [])
        await self.send(
            timed_nova(SECOND, 2000, final=True), utterance_end(1000)
        )
        self.assertEqual(self.controller.agent.inputs, [])
        await self.send(utterance_end(2000))
        self.assertEqual(self.controller.agent.inputs, [FIRST + SECOND])

    async def test_invalid_or_unmatched_gap_never_submits_partial_text(self):
        await self.send(timed_nova(FIRST, 1000))
        await self.send(utterance_end(1000))
        self.assertEqual(self.controller.agent.inputs, [])
        await self.send(timed_nova(FIRST, 1000, final=True))
        for value in (None, -1, True, 999, 1001):
            await self.send(utterance_end(value))
        self.assertEqual(self.controller.agent.inputs, [])
        await self.send(utterance_end(1000))
        self.assertEqual(self.controller.agent.inputs, [FIRST.strip()])

    async def test_normal_endpoint_does_not_wait_for_gap_or_duplicate_it(self):
        await self.send(timed_nova(FIRST, 1000, final=True, endpoint=True))
        self.assertEqual(self.controller.agent.inputs, [FIRST.strip()])
        await self.send(utterance_end(1000))
        await self.send(
            timed_nova(SECOND, 2000, final=True), utterance_end(1000)
        )
        self.assertEqual(self.controller.agent.inputs, [FIRST.strip()])
        await self.send(utterance_end(2000))
        self.assertEqual(self.controller.agent.inputs, [FIRST.strip(), SECOND])

    async def test_segment_finals_make_one_complete_llm_input(self):
        self.assertEqual((len(FIRST), len(SECOND)), (18, 15))
        await self.send(nova(FIRST, True))
        self.assertEqual(self.controller.agent.inputs, [])
        await self.send(nova(SECOND, True, True))
        self.assertEqual(self.controller.agent.inputs, [FIRST + SECOND])
        self.assertEqual(self.controller.turn_id, 1)
        self.assertEqual(self.flushes(), 1)

    async def test_empty_endpoint_commits_prior_final_parts_once(self):
        await self.send(nova(FIRST, True), nova("", True, True))
        self.assertEqual(self.controller.agent.inputs, [FIRST.strip()])
        await self.send(nova("", True, True))
        self.assertEqual(self.controller.agent.inputs, [FIRST.strip()])
        self.assertEqual(self.controller.turn_id, 1)

    async def test_interims_replace_the_live_suffix_and_never_accumulate(self):
        await self.send(
            nova("obsolete interim"),
            nova(FIRST, True),
            nova("second"),
            nova("second obsolete interim"),
            nova(SECOND, True, True),
        )
        self.assertEqual(self.controller.agent.inputs, [FIRST + SECOND])
        messages = [
            v for name, v in self.transport.messages if name == "message"
        ]
        self.assertEqual(
            messages[-2]["text"], FIRST + "second obsolete interim"
        )
        self.assertFalse(messages[-2]["is_final"])
        self.assertTrue(messages[-1]["is_final"])
        self.assertEqual(self.flushes(), 1)

    async def test_two_utterances_increment_two_turns_without_old_prefix(self):
        await self.send(
            nova(FIRST, True),
            nova(SECOND, True, True),
            nova("new question", True, True),
        )
        self.assertEqual(
            self.controller.agent.inputs,
            [FIRST + SECOND, "new question"],
        )
        self.assertEqual(self.controller.turn_id, 2)
        self.assertEqual(self.flushes(), 2)

    async def test_identical_text_in_two_real_utterances_is_not_deduplicated(
        self,
    ):
        await self.send(
            nova("repeat question", True, True),
            nova("repeat question", True, True),
        )
        self.assertEqual(self.controller.agent.inputs, ["repeat question"] * 2)
        self.assertEqual(self.controller.turn_id, 2)
        self.assertEqual(self.flushes(), 2)

    async def test_sessions_do_not_merge_final_segments(self):
        await self.send(
            nova(FIRST, True, session="101"),
            nova("other speaker", True, True, "102"),
            nova(SECOND, True, True, "101"),
        )
        self.assertEqual(
            self.controller.agent.inputs, ["other speaker", FIRST + SECOND]
        )

    async def test_other_asr_and_flux_keep_their_final_contract(self):
        for info in (
            {},
            {"vendor": "other"},
            {"vendor": "deepgram", "turn_event": "EndOfTurn"},
        ):
            await self.send(
                ASRResultEvent(
                    text="legacy final",
                    final=True,
                    metadata={"session_id": "100", "asr_info": info},
                )
            )
        self.assertEqual(self.controller.agent.inputs, ["legacy final"] * 3)
        self.assertEqual(self.controller.turn_id, 3)

    async def test_barge_in_flushes_immediately_before_any_endpoint(self):
        self.controller.sentence_fragment = "pending assistant sentence"
        await self.send(nova("user starts speaking"))
        self.assertEqual(self.flushes(), 1)
        self.assertEqual(self.controller.sentence_fragment, "")
        self.assertEqual(self.controller.agent.inputs, [])
        self.assertIn(("data", "tts_flush", ["tts"]), self.transport.events)
        self.assertIn(("cmd", "flush", ["agora_rtc"]), self.transport.events)
        await self.send(
            nova("user keeps speaking"), nova("complete question", True, True)
        )
        self.assertEqual(self.flushes(), 1)
        self.assertEqual(self.controller.agent.inputs, ["complete question"])

    async def test_leave_and_stop_discard_unfinished_prefixes(self):
        await self.send(nova(FIRST, True))
        await self.controller._on_user_left(main_control.UserLeftEvent())
        await self.send(nova(SECOND, True, True))
        self.assertEqual(self.controller.agent.inputs, [SECOND])
        await self.send(nova(FIRST, True))
        await self.controller.on_stop(self.transport)
        self.assertTrue(self.controller.agent.stopped)
        await self.send(nova(SECOND, True, True))
        self.assertEqual(self.controller.agent.inputs, [SECOND])

    async def test_fragment_overflow_discards_whole_utterance_until_endpoint(
        self,
    ):
        for _ in range(257):
            await self.send(nova("synthetic stable segment", True))
        await self.send(nova("", True, True))
        self.assertEqual(self.controller.agent.inputs, [])
        self.assertTrue(self.transport.warnings)
        await self.send(nova("fresh complete question", True, True))
        self.assertEqual(
            self.controller.agent.inputs, ["fresh complete question"]
        )


if __name__ == "__main__":
    unittest.main()
