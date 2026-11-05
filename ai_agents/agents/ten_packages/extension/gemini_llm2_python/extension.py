#
#
# Agora Real Time Engagement
# Created for Google Gemini Integration
# Copyright (c) 2024 Agora IO. All rights reserved.
#
#
import asyncio
from contextlib import AsyncExitStack
from contextvars import ContextVar
from typing import AsyncGenerator, Optional

from ten_ai_base.llm2 import AsyncLLM2BaseExtension
from ten_ai_base.struct import (
    LLMRequest,
    LLMRequestRetrievePrompt,
    LLMResponse,
    LLMResponseRetrievePrompt,
)
from ten_runtime.async_ten_env import AsyncTenEnv
from ten_runtime.cmd import Cmd

from .gemini import GeminiChatAPI, GeminiLLM2Config


class GeminiLLM2Extension(AsyncLLM2BaseExtension):
    def __init__(self, name: str):
        super().__init__(name)
        self.memory = []
        self.memory_cache = []
        self.config = None
        self.client = None
        self.sentence_fragment = ""
        self.tool_task_future: Optional[asyncio.Future] = None
        self.users_count = 0
        self.last_reasoning_ts = 0
        self._stream_scope: ContextVar[
            Optional[tuple[Optional[asyncio.Task], AsyncExitStack]]
        ] = ContextVar("gemini_response_stream_scope", default=None)

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_info("on_init")
        await super().on_init(ten_env)

    async def on_start(self, async_ten_env: AsyncTenEnv) -> None:
        async_ten_env.log_info("on_start")
        await super().on_start(async_ten_env)
        config_json, _ = await self.ten_env.get_property_to_json("")
        self.config = GeminiLLM2Config.model_validate_json(config_json)

        # Mandatory properties
        if not self.config.api_key:
            async_ten_env.log_info("API key is missing, exiting on_start")
            return

        # Create instance
        try:
            self.client = GeminiChatAPI(async_ten_env, self.config)
            async_ten_env.log_info(
                f"initialized with max_tokens: {self.config.max_tokens}, "
                f"model: {self.config.model}"
            )
        except Exception as err:
            async_ten_env.log_info(f"Failed to initialize GeminiChatAPI: {err}")

    async def on_stop(self, async_ten_env: AsyncTenEnv) -> None:
        async_ten_env.log_info("on_stop")
        await super().on_stop(async_ten_env)

    async def on_deinit(self, async_ten_env: AsyncTenEnv) -> None:
        async_ten_env.log_info("on_deinit")
        await super().on_deinit(async_ten_env)

    async def on_retrieve_prompt(
        self, async_ten_env: AsyncTenEnv, request: LLMRequestRetrievePrompt
    ) -> LLMResponseRetrievePrompt:
        """Retrieve the current prompt from config."""
        prompt = self.config.prompt if self.config else ""
        async_ten_env.log_info(
            f"Retrieved prompt for request_id: {request.request_id}"
        )
        return LLMResponseRetrievePrompt(prompt=prompt)

    def on_call_chat_completion(
        self, async_ten_env: AsyncTenEnv, request_input: LLMRequest
    ) -> AsyncGenerator[LLMResponse, None]:
        stream = self.client.get_chat_completions(request_input)
        scope = self._stream_scope.get()
        if scope is not None:
            owner, stack = scope
            if owner is asyncio.current_task():
                stack.push_async_callback(stream.aclose)
        return stream

    async def _run_stream(
        self, ten_env: AsyncTenEnv, cmd: Cmd, req: LLMRequest
    ) -> None:
        # The registry base awaits return_result after each yielded event.
        # Cancellation there does not enter the provider generator, so own its
        # closure here rather than waiting for async-generator garbage collection.
        async with AsyncExitStack() as stack:
            token = self._stream_scope.set((asyncio.current_task(), stack))
            try:
                await super()._run_stream(ten_env, cmd, req)
            finally:
                self._stream_scope.reset(token)
