# Main Control Python Extension

A TEN Framework extension that serves as the central control logic for AI agent interactions, managing speech recognition, language model processing, and text-to-speech coordination.

## Overview

The `main_python` extension acts as the orchestrator for AI agent conversations, handling real-time speech processing, LLM interactions, and TTS output. It manages user session state and coordinates data flow between different components in the TEN Framework.

## Features

- **Real-time Speech Processing**: Handles ASR (Automatic Speech Recognition) results and manages streaming text
- **LLM Integration**: Coordinates with language models for natural language understanding and response generation
- **TTS Coordination**: Manages text-to-speech requests for audio output
- **Session Management**: Tracks user presence and manages conversation state
- **Streaming Support**: Handles both final and intermediate results for smooth user experience
- **Caption Generation**: Provides real-time captions for accessibility and logging

## API Interface

### Input Data

#### ASR Result
```json
{
  "text": "string",
  "final": "bool",
  "metadata": {
    "session_id": "string"
  }
}
```

#### LLM Result
```json
{
  "text": "string",
  "end_of_segment": "bool"
}
```

For Deepgram Nova (`metadata.asr_info.vendor = "deepgram"` and
`api = "listen_v1"`), `final` marks a stable transcript **segment**, not a
completed user turn. The controller joins finalized segments per session until
`speech_final = true`, including an empty endpoint result. Interim transcripts
replace only the live suffix. The first qualifying speech result interrupts the
assistant immediately; one endpoint creates one LLM request. Other ASR providers
and Deepgram Flux retain their existing `final` contract.

Unfinished Nova utterances are bounded to 256 finalized segments and 32,768
characters, with at most eight retained sessions. Overflow discards that whole
utterance until its actual endpoint and emits a warning without transcript text;
the next utterance can proceed. User leave and stop clear unfinished state. This
consumer does not synthesize an endpoint from silence timers or punctuation.
Provider endpointing remains the fast path. When the Nova adapter also forwards
`message_type: UtteranceEnd`, its `last_word_end_ms` must exactly match both the
last pending finalized word and the latest recognized word. Only then can that
explicit provider gap release an otherwise unfinished utterance. Duplicate,
negative, unknown and stale markers do not submit text; newer interim speech
prevents an older gap from cutting the current utterance. No local silence timer
or punctuation heuristic is used. Enable the optional provider gap with
`params.utterance_end_ms: 1000` and `interim_results: true`; normal
`speech_final` results still submit immediately.

### Output Data

#### Text Data
```json
{
  "text": "string",
  "is_final": "bool",
  "end_of_segment": "bool",
  "stream_id": "uint32"
}
```

### Commands

#### Input Commands
- `on_user_joined`: Triggered when a user joins the session
- `on_user_left`: Triggered when a user leaves the session

#### Output Commands
- `flush`: Sends flush commands to LLM, TTS, and RTC components

## Configuration

The extension supports the following configuration options:

```json
{
  "greeting": "Hello there, I'm TEN Agent",
  "flush_destinations": ["agora_rtc"]
}
```

### Configuration Parameters

- `greeting` (string, default: "Hello there, I'm TEN Agent"): The greeting message to display when the first user joins
- `flush_destinations` (non-empty array of extension names, default: `["agora_rtc"]`): Explicit command destinations to flush after interrupting the LLM and TTS. Names are trimmed and duplicate destinations are sent only once, in the configured order. This option requires a version of `main_python` whose config, implementation, and manifest all support it; passing the property to an older version does not enable avatar interruption.

For an avatar graph without an outgoing avatar `flush` route to RTC, configure
`["avatar", "agora_rtc"]` so the avatar clears its queued audio before RTC is
flushed. If the graph already forwards the avatar's `flush` command to RTC,
configure `["avatar"]` to avoid flushing RTC twice. Static routes from
`main_control` do not add destinations to this explicitly addressed command.

## Dependencies

- `ten_runtime_python` (version 0.10): Core TEN Framework runtime
- `ten_ai_base` (version 0.6.9): AI base functionality

## Usage

### Installation

The extension is part of the TEN Framework and can be installed through the TEN package manager:

```bash
ten install main_python
```

### Integration

This extension is designed to work with other TEN Framework components:

- **ASR Extension**: Provides speech recognition results
- **LLM Extension**: Processes natural language and generates responses
- **TTS Extension**: Converts text to speech
- **RTC Extension**: Handles real-time communication
- **Message Collector**: Captures and displays conversation data

### Workflow

1. **User Joins**: When a user joins, the extension sends a greeting if configured
2. **Speech Processing**: ASR results are processed and captions are generated
3. **LLM Processing**: Final speech segments are sent to the LLM for processing
4. **Response Generation**: LLM responses are converted to speech and displayed as captions
5. **Streaming**: Both intermediate and final results are handled for smooth interaction

## Development

### Building

The extension uses the standard TEN Framework build system:

```bash
ten build main_python
```

### Testing

Run the extension tests:

```bash
ten test main_python
```

## Architecture

The extension implements the `AsyncExtension` interface and provides:

- **Lifecycle Management**: Proper initialization, start, stop, and cleanup
- **Event Handling**: Processes commands and data events asynchronously
- **State Management**: Tracks user count and conversation state
- **Data Routing**: Routes data between different framework components

## License

This extension is part of the TEN Framework and is licensed under the Apache License, Version 2.0.

## Contributing

Contributions are welcome! Please refer to the main TEN Framework documentation for contribution guidelines.
