"""Offline regression for explicit interruption destinations.

Uses the real main controller and its real send helpers with a fake TEN
transport. Requires installed TEN interfaces, without an agent or provider.
"""

# pylint: disable=protected-access

import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from pydantic import ValidationError


MAIN = Path(__file__).resolve().parents[1]
for name, path in (
    ("interrupt_test_main", MAIN),
    ("interrupt_test_main.agent", MAIN / "agent"),
):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

main_control = importlib.import_module("interrupt_test_main.extension")
helper = importlib.import_module("interrupt_test_main.helper")
MainControlConfig = importlib.import_module(
    "interrupt_test_main.config"
).MainControlConfig


class FakeMessage:
    def __init__(self, name):
        self.name = name
        self.destinations = []
        self.payload = None

    @classmethod
    def create(cls, name):
        return cls(name)

    def set_dests(self, destinations):
        self.destinations = destinations

    def set_property_from_json(self, _key, value):
        self.payload = json.loads(value)


class FakeLoc:
    def __init__(self, _app, _graph, extension):
        self.extension = extension


class FakeTransport:
    def __init__(self):
        self.events = []

    def log_debug(self, *_args):
        pass

    def log_info(self, *_args):
        pass

    async def send_cmd(self, command):
        self.events.append(
            ("cmd", command.name, [d.extension for d in command.destinations])
        )
        return None, None

    async def send_data(self, data):
        self.events.append(
            ("data", data.name, [d.extension for d in data.destinations])
        )
        return None


async def interrupt(config):
    transport = FakeTransport()

    async def flush_llm():
        transport.events.append(("llm_flush",))

    controller = types.SimpleNamespace(
        config=config,
        sentence_fragment="Synthetic pending fragment",
        ten_env=transport,
        agent=types.SimpleNamespace(flush_llm=flush_llm),
    )
    with (
        patch.object(helper, "Cmd", FakeMessage),
        patch.object(helper, "Data", FakeMessage),
        patch.object(helper, "Loc", FakeLoc),
    ):
        await main_control.MainControlExtension._interrupt(controller)
    assert controller.sentence_fragment == ""
    return transport.events


class TestInterruptRouting(unittest.IsolatedAsyncioTestCase):
    async def test_default_keeps_existing_rtc_destination(self):
        self.assertEqual(
            await interrupt(MainControlConfig()),
            [
                ("llm_flush",),
                ("data", "tts_flush", ["tts"]),
                ("cmd", "flush", ["agora_rtc"]),
            ],
        )

    async def test_configured_targets_receive_explicit_flush_once(self):
        self.assertEqual(
            await interrupt(
                MainControlConfig(
                    flush_destinations=["avatar", "agora_rtc", "avatar"]
                )
            ),
            [
                ("llm_flush",),
                ("data", "tts_flush", ["tts"]),
                ("cmd", "flush", ["avatar"]),
                ("cmd", "flush", ["agora_rtc"]),
            ],
        )

    async def test_avatar_only_supports_graph_owned_forwarding(self):
        self.assertEqual(
            await interrupt(MainControlConfig(flush_destinations=["avatar"])),
            [
                ("llm_flush",),
                ("data", "tts_flush", ["tts"]),
                ("cmd", "flush", ["avatar"]),
            ],
        )

    def test_target_names_are_trimmed_and_order_is_preserved(self):
        config = MainControlConfig(
            flush_destinations=[" avatar ", "agora_rtc", "avatar"]
        )
        self.assertEqual(config.flush_destinations, ["avatar", "agora_rtc"])

    def test_empty_destinations_are_invalid(self):
        for destinations in ([], [""], [" \t"]):
            with self.subTest(destinations=destinations):
                with self.assertRaises(ValidationError):
                    MainControlConfig(flush_destinations=destinations)


if __name__ == "__main__":
    unittest.main()
