import asyncio
import json
import time
from typing import Literal
from dataclasses import dataclass, field

from .agent.decorators import agent_event_handler
from ten_runtime import (
    AsyncExtension,
    AsyncTenEnv,
    Cmd,
    Data,
)

from .agent.agent import Agent
from .agent.events import (
    ASRResultEvent,
    LLMResponseEvent,
    ToolRegisterEvent,
    UserJoinedEvent,
    UserLeftEvent,
)
from .helper import _send_cmd, _send_data, parse_sentences
from .config import MainControlConfig  # assume extracted from your base model

import uuid


# Bounds contain unfinished vendor segments, not a timeout or a turn detector.
NOVA_MAX_SESSIONS = 8
NOVA_MAX_SEGMENTS = 256
NOVA_MAX_CHARACTERS = 32768


@dataclass
class _NovaUtterance:
    parts: list[str] = field(default_factory=list)
    characters: int = 0
    interrupted: bool = False
    overflow: bool = False


class MainControlExtension(AsyncExtension):
    """
    The entry point of the agent module.
    Consumes semantic AgentEvents from the Agent class and drives the runtime behavior.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.ten_env: AsyncTenEnv = None
        self.agent: Agent = None
        self.config: MainControlConfig = None

        self.stopped: bool = False
        self._rtc_user_count: int = 0
        self.sentence_fragment: str = ""
        self.turn_id: int = 0
        self.session_id: str = "0"
        self._nova_utterances: dict[str, _NovaUtterance] = {}
        self._nova_session_bound_reported = False

    def _current_metadata(self) -> dict:
        return {"session_id": self.session_id, "turn_id": self.turn_id}

    async def on_init(self, ten_env: AsyncTenEnv):
        self.ten_env = ten_env

        # Load config from runtime properties
        config_json, _ = await ten_env.get_property_to_json(None)
        self.config = MainControlConfig.model_validate_json(config_json)

        self.agent = Agent(ten_env)

        # Now auto-register decorated methods
        for attr_name in dir(self):
            fn = getattr(self, attr_name)
            event_type = getattr(fn, "_agent_event_type", None)
            if event_type:
                self.agent.on(event_type, fn)

    # === Register handlers with decorators ===
    @agent_event_handler(UserJoinedEvent)
    async def _on_user_joined(self, event: UserJoinedEvent):
        self._rtc_user_count += 1
        if self._rtc_user_count == 1 and self.config and self.config.greeting:
            await self._send_to_tts(self.config.greeting, True)
            await self._send_transcript(
                "assistant", self.config.greeting, True, 100
            )

    @agent_event_handler(UserLeftEvent)
    async def _on_user_left(self, event: UserLeftEvent):
        self._rtc_user_count -= 1
        # UserLeftEvent has no speaker identity. Do not retain unfinished words
        # across membership changes or join them to a replacement speaker.
        self._nova_utterances.clear()
        self._nova_session_bound_reported = False

    @agent_event_handler(ToolRegisterEvent)
    async def _on_tool_register(self, event: ToolRegisterEvent):
        await self.agent.register_llm_tool(event.tool, event.source)

    @agent_event_handler(ASRResultEvent)
    async def _on_asr_result(self, event: ASRResultEvent):
        if self.stopped:
            return
        self.session_id = event.metadata.get("session_id", "100")
        stream_id = int(self.session_id)
        info = event.metadata.get("asr_info")
        if (
            isinstance(info, dict)
            and info.get("vendor") == "deepgram"
            and info.get("api") == "listen_v1"
        ):
            await self._on_nova_result(event, info, stream_id)
            return
        if not event.text:
            return
        if event.final or len(event.text) > 2:
            await self._interrupt()
        if event.final:
            self.turn_id += 1
            await self.agent.queue_llm_input(event.text)
        await self._send_transcript("user", event.text, event.final, stream_id)

    async def _on_nova_result(
        self, event: ASRResultEvent, info: dict, stream_id: int
    ) -> None:
        """Nova finalizes segments; only speech_final ends a user utterance."""
        session_id = self.session_id
        state = self._nova_utterances.get(session_id)
        if state is None:
            if len(self._nova_utterances) >= NOVA_MAX_SESSIONS:
                # Keep known sessions rather than evicting an unfinished prefix
                # and later submitting a truncated suffix as a complete turn.
                if not self._nova_session_bound_reported:
                    self.ten_env.log_warn(
                        "Nova utterance session limit reached"
                    )
                    self._nova_session_bound_reported = True
                return
            state = _NovaUtterance()
            self._nova_utterances[session_id] = state

        text = event.text.strip()
        endpoint = info.get("speech_final") is True
        if text and not state.interrupted and (event.final or len(text) > 2):
            state.interrupted = True
            await self._interrupt()
            if (
                self.stopped
                or self._nova_utterances.get(session_id) is not state
            ):
                return

        if not state.overflow:
            size = state.characters + len(text) + bool(state.parts and text)
            if size > NOVA_MAX_CHARACTERS or (
                text and event.final and len(state.parts) >= NOVA_MAX_SEGMENTS
            ):
                state.overflow = True
                state.parts.clear()
                state.characters = 0
                self.ten_env.log_warn(
                    "Nova utterance buffer limit reached; discarding until endpoint"
                )
            elif text and event.final:
                state.parts.append(text)
                state.characters = size

        if state.overflow:
            if endpoint:
                self._nova_utterances[session_id] = _NovaUtterance()
            return

        # Interim results replace the live suffix; they are never appended to
        # the stable segment buffer. An empty endpoint still commits that buffer.
        parts = state.parts + ([text] if text and not event.final else [])
        combined = " ".join(parts)
        if endpoint:
            self._nova_utterances[session_id] = _NovaUtterance()
            if not combined:
                return
            self.turn_id += 1
            await self.agent.queue_llm_input(combined)
        if combined:
            await self._send_transcript("user", combined, endpoint, stream_id)

    @agent_event_handler(LLMResponseEvent)
    async def _on_llm_response(self, event: LLMResponseEvent):
        if not event.is_final and event.type == "message":
            sentences, self.sentence_fragment = parse_sentences(
                self.sentence_fragment, event.delta
            )
            for s in sentences:
                await self._send_to_tts(s, False)

        if event.is_final and event.type == "message":
            remaining_text = self.sentence_fragment or ""
            self.sentence_fragment = ""
            await self._send_to_tts(remaining_text, True)

        await self._send_transcript(
            "assistant",
            event.text,
            event.is_final,
            100,
            data_type=("reasoning" if event.type == "reasoning" else "text"),
        )

    async def on_start(self, ten_env: AsyncTenEnv):
        ten_env.log_info("[MainControlExtension] on_start")

    async def on_stop(self, ten_env: AsyncTenEnv):
        ten_env.log_info("[MainControlExtension] on_stop")
        self.stopped = True
        self._nova_utterances.clear()
        self._nova_session_bound_reported = False
        await self.agent.stop()

    async def on_cmd(self, ten_env: AsyncTenEnv, cmd: Cmd):
        await self.agent.on_cmd(cmd)

    async def on_data(self, ten_env: AsyncTenEnv, data: Data):
        await self.agent.on_data(data)

    # === helpers ===
    async def _send_transcript(
        self,
        role: str,
        text: str,
        final: bool,
        stream_id: int,
        data_type: Literal["text", "reasoning"] = "text",
    ):
        """
        Sends the transcript (ASR or LLM output) to the message collector.
        """
        if data_type == "text":
            await _send_data(
                self.ten_env,
                "message",
                "message_collector",
                {
                    "data_type": "transcribe",
                    "role": role,
                    "text": text,
                    "text_ts": int(time.time() * 1000),
                    "is_final": final,
                    "stream_id": stream_id,
                },
            )
        elif data_type == "reasoning":
            await _send_data(
                self.ten_env,
                "message",
                "message_collector",
                {
                    "data_type": "raw",
                    "role": role,
                    "text": json.dumps(
                        {
                            "type": "reasoning",
                            "data": {
                                "text": text,
                            },
                        }
                    ),
                    "text_ts": int(time.time() * 1000),
                    "is_final": final,
                    "stream_id": stream_id,
                },
            )
        self.ten_env.log_info(
            f"[MainControlExtension] Sent transcript: {role}, final={final}, text={text}"
        )

    async def _send_to_tts(self, text: str, is_final: bool):
        """
        Sends a sentence to the TTS system.
        """
        request_id = f"tts-request-{self.turn_id}"
        await _send_data(
            self.ten_env,
            "tts_text_input",
            "tts",
            {
                "request_id": request_id,
                "text": text,
                "text_input_end": is_final,
                "metadata": self._current_metadata(),
            },
        )
        self.ten_env.log_info(
            f"[MainControlExtension] Sent to TTS: is_final={is_final}, text={text}"
        )

    async def _interrupt(self):
        """
        Interrupts ongoing LLM and TTS generation. Typically called when user speech is detected.
        """
        self.sentence_fragment = ""
        await self.agent.flush_llm()
        await _send_data(
            self.ten_env, "tts_flush", "tts", {"flush_id": str(uuid.uuid4())}
        )
        for destination in self.config.flush_destinations:
            await _send_cmd(self.ten_env, "flush", destination)
        self.ten_env.log_info("[MainControlExtension] Interrupt signal sent")
