#
#
# Agora Real Time Engagement
# Google Gemini LLM2 Integration
# Copyright (c) 2024 Agora IO. All rights reserved.
#
#
from dataclasses import dataclass, field
import json
import time
from typing import AsyncGenerator, List, Optional
from urllib.parse import quote
from pydantic import BaseModel
import httpx

from ten_ai_base.struct import (
    LLMMessageContent,
    LLMMessageFunctionCall,
    LLMMessageFunctionCallOutput,
    LLMRequest,
    LLMResponse,
    LLMResponseMessageDelta,
    LLMResponseMessageDone,
    LLMResponseToolCall,
    TextContent,
)
from ten_ai_base.types import LLMToolMetadata
from ten_runtime.async_ten_env import AsyncTenEnv


@dataclass
class GeminiLLM2Config(BaseModel):
    api_key: str = ""
    model: str = "gemini-3-pro-preview"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    reasoning_effort: Optional[str] = None
    native_text_api: bool = False
    prompt: str = "You are a helpful assistant."
    black_list_params: List[str] = field(
        default_factory=lambda: ["messages", "tools", "stream", "model"]
    )

    def is_black_list_params(self, key: str) -> bool:
        return key in self.black_list_params


def _created_timestamp(value: object) -> int:
    # TEN's command-result schema rejects null even though the Python model
    # allows it. Preserve provider integers and fill missing timestamps.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(time.time())


def _native_request(request: dict) -> dict:
    """Translate the supported text-only subset, rejecting silent omissions."""
    allowed = {
        "model",
        "messages",
        "stream",
        "temperature",
        "top_p",
        "max_tokens",
        "reasoning_effort",
    }
    if set(request) - allowed:
        raise ValueError("Native text API does not support these parameters")
    generation = {
        "temperature": request["temperature"],
        "topP": request["top_p"],
        "maxOutputTokens": request["max_tokens"],
    }
    effort = request.get("reasoning_effort")
    if effort is not None:
        if effort != "none":
            raise ValueError("Native text API supports reasoning_effort=none")
        generation["thinkingConfig"] = {"thinkingBudget": 0}
    system, contents = [], []
    for message in request["messages"]:
        if set(message) != {"role", "content"}:
            raise ValueError("Native text API does not support tool history")
        parts = [{"text": message["content"]}]
        role = message["role"]
        if role == "system":
            system.extend(parts)
        elif role in ("user", "assistant"):
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": parts,
                }
            )
        else:
            raise ValueError("Native text API requires text conversation roles")
    result = {"contents": contents, "generationConfig": generation}
    if system:
        result["systemInstruction"] = {"parts": system}
    return result


def _native_response(payload: dict, streaming: bool) -> dict:
    """Normalize native text events to the existing TEN response adapter."""
    if payload.get("error") or payload.get("promptFeedback", {}).get(
        "blockReason"
    ):
        raise RuntimeError("Native text API rejected the generation")
    candidates = payload.get("candidates") or []
    if not candidates:
        return {}
    candidate = candidates[0]
    reason = candidate.get("finishReason")
    if reason and reason not in ("STOP", "MAX_TOKENS"):
        raise RuntimeError(f"Native text generation ended with {reason}")
    text = ""
    for part in candidate.get("content", {}).get("parts", []):
        if part.get("thought"):
            continue
        if "text" not in part:
            raise RuntimeError("Native text API returned non-text content")
        text += part["text"]
    return {
        "id": payload.get("responseId") or "",
        "choices": [{"delta" if streaming else "message": {"content": text}}],
        "native_finished": bool(reason),
    }


class GeminiChatAPI:
    def __init__(self, ten_env: AsyncTenEnv, config: GeminiLLM2Config):
        self.config = config
        self.ten_env = ten_env
        ten_env.log_info(
            f"GeminiChatAPI initialized with model: {config.model} "
            f"(base_url={config.base_url})"
        )
        self.http_client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=30.0,
        )
        self._request_headers = {}
        self._request_params = {}
        if self.config.api_key:
            # OpenAI-compatible endpoint expects Bearer token in
            # Authorization header
            self._request_headers["Authorization"] = (
                f"Bearer {self.config.api_key}"
            )

    def _convert_tools_to_dict(self, tool: LLMToolMetadata):
        """Convert LLMToolMetadata to Gemini function definition format."""
        json_dict = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

        for param in tool.parameters:
            properties = json_dict["function"]["parameters"]["properties"]
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required = json_dict["function"]["parameters"]["required"]
                required.append(param.name)

        return json_dict

    def _parse_message_content(self, message: LLMMessageContent):
        """Parse message content into a plain text payload for Gemini."""
        content = message.content
        if isinstance(content, str):
            return content

        text_content = ""
        if isinstance(content, list):
            for part in content:
                if isinstance(part, TextContent):
                    text_content += part.text
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        text_content += part.get("text", "")
                    elif self.config.native_text_api:
                        raise ValueError("Native text API requires text input")
                elif self.config.native_text_api:
                    raise ValueError("Native text API requires text input")
        return text_content

    async def get_chat_completions(
        self, request_input: LLMRequest
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream chat completions from Gemini API."""
        try:
            # Build system prompt
            system_prompt = self.config.prompt

            # Convert messages
            parsed_messages = []
            tools = None

            for message in request_input.messages or []:
                if isinstance(message, LLMMessageContent):
                    role = message.role.lower()
                    content = self._parse_message_content(message)
                    if content:
                        parsed_messages.append(
                            {"role": role, "content": content}
                        )
                elif isinstance(message, LLMMessageFunctionCall):
                    parsed_messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": message.call_id,
                                    "type": "function",
                                    "function": {
                                        "name": message.name,
                                        "arguments": message.arguments,
                                    },
                                }
                            ],
                        }
                    )
                elif isinstance(message, LLMMessageFunctionCallOutput):
                    parsed_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": message.call_id,
                            "content": message.output,
                        }
                    )
                elif self.config.native_text_api:
                    raise ValueError("Native text API requires text messages")

            for tool in request_input.tools or []:
                if tools is None:
                    tools = []
                tools.append(self._convert_tools_to_dict(tool))

            # Build request - only include Gemini-compatible parameters
            req = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    *parsed_messages,
                ],
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
                "stream": request_input.streaming,
            }

            if self.config.reasoning_effort is not None:
                req["reasoning_effort"] = self.config.reasoning_effort

            # Only add tools if present
            if tools:
                req["tools"] = tools

            # Add additional parameters if they are not in the black list
            for key, value in (request_input.parameters or {}).items():
                if not self.config.is_black_list_params(key):
                    self.ten_env.log_debug(f"set gemini param: {key} = {value}")
                    req[key] = value

            path = "/chat/completions"
            headers, params = self._request_headers, self._request_params
            native = self.config.native_text_api
            if native:
                method = (
                    "streamGenerateContent"
                    if request_input.streaming
                    else "generateContent"
                )
                path = f"/models/{quote(self.config.model, safe='')}:{method}"
                req = _native_request(req)
                headers = {"x-goog-api-key": self.config.api_key}
                params = {"alt": "sse"} if request_input.streaming else {}

            self.ten_env.log_info(
                f"Requesting chat completions with model: {self.config.model}"
            )

            # Read SSE as it arrives; post() buffers the whole response body.
            # The context also closes the response on cancellation or aclose().
            async with self.http_client.stream(
                "POST",
                path,
                json=req,
                params=params,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    error_text = response.text
                    extra_hint = ""
                    if response.status_code == 404:
                        extra_hint = (
                            " (check that the model exists at the configured "
                            "base_url; Vertex users may need a project-specific "
                            "endpoint)"
                        )
                    error_msg = (
                        f"Gemini API error: {response.status_code} - "
                        f"{error_text}{extra_hint}"
                    )
                    self.ten_env.log_error(error_msg)
                    raise RuntimeError(
                        f"Gemini API error: {response.status_code} - {error_text}"
                    )

                # Handle streaming response
                if request_input.streaming:
                    full_content = ""
                    response_id = ""
                    created = int(time.time())
                    native_finished = False
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: [DONE]"):
                            break
                        if line.startswith("data: "):
                            try:
                                chunk_data = json.loads(line[6:])
                                if native:
                                    chunk_data = _native_response(
                                        chunk_data, streaming=True
                                    )
                                    native_finished = chunk_data.get(
                                        "native_finished", False
                                    )
                                if chunk_data.get("choices"):
                                    response_id = chunk_data.get("id") or ""
                                    created = _created_timestamp(
                                        chunk_data.get("created")
                                    )
                                    choice = chunk_data["choices"][0]
                                    delta = choice.get("delta", {})
                                    if delta.get("content"):
                                        content = delta["content"]
                                        full_content += content
                                        yield LLMResponseMessageDelta(
                                            response_id=response_id,
                                            role="assistant",
                                            content=full_content,
                                            delta=content,
                                            created=created,
                                        )
                                    # Check for tool calls
                                    if delta.get("tool_calls"):
                                        for tool_call in delta["tool_calls"]:
                                            function = tool_call.get(
                                                "function", {}
                                            )
                                            yield LLMResponseToolCall(
                                                response_id=chunk_data.get(
                                                    "id", ""
                                                ),
                                                call_id=tool_call.get("id", ""),
                                                name=function.get("name", ""),
                                                arguments=function.get(
                                                    "arguments", ""
                                                ),
                                            )
                                if native_finished:
                                    break
                            except json.JSONDecodeError:
                                if native:
                                    raise
                                self.ten_env.log_debug(
                                    f"Could not parse line: {line}"
                                )
                                continue

                    if native and not native_finished:
                        raise RuntimeError("Native text stream ended early")

                    # Send completion message
                    yield LLMResponseMessageDone(
                        response_id=response_id,
                        role="assistant",
                        content=full_content,
                        created=created,
                    )
                else:
                    # Non-streaming response
                    await response.aread()
                    response_data = response.json()
                    if native:
                        response_data = _native_response(
                            response_data, streaming=False
                        )
                        if not response_data.get("native_finished"):
                            raise RuntimeError(
                                "Native text response incomplete"
                            )
                    if response_data.get("choices"):
                        response_id = response_data.get("id") or ""
                        created = _created_timestamp(
                            response_data.get("created")
                        )
                        choice = response_data["choices"][0]
                        content = choice.get("message", {}).get("content") or ""
                        if content:
                            yield LLMResponseMessageDelta(
                                response_id=response_id,
                                role="assistant",
                                content=content,
                                delta=content,
                                created=created,
                            )
                        yield LLMResponseMessageDone(
                            response_id=response_id,
                            role="assistant",
                            content=content,
                            created=created,
                        )
        except Exception as e:
            self.ten_env.log_error(f"Error in get_chat_completions: {e}")
            raise RuntimeError(f"CreateChatCompletion failed, err: {e}") from e
