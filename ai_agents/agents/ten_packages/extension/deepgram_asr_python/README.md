# Deepgram Nova ASR Extension

## Configuration

For Nova, `params.utterance_end_ms: 1000` with `interim_results: true` enables
the provider's additional gap signal. The adapter forwards `UtteranceEnd` as
an empty, non-final `asr_result` carrying `metadata.asr_info.message_type` and
`last_word_end_ms`. This marker is neither another transcript final nor an
`asr_finalize_end` acknowledgement. Result word ends and gap markers are mapped
onto the same user-audio timeline, including injected silence and reconnect
offsets. The provider's `last_word_end: -1` marker is ignored.

The voice-assistant main controller can use this explicit event when Nova
omits `speech_final`. Consumers must match its timestamp to pending finalized
words and reject stale markers after newer speech. It is opt-in at the provider;
the ordinary endpoint path does not wait for this gap. See the
[provider specification](https://developers.deepgram.com/docs/utterance-end).

### Configuration Parameters

All parameters are configured through the `params` object:

### nova model

```json
{
    "params": {
        "key": "${env:DEEPGRAM_API_KEY}",
        "url": "wss://api.deepgram.com/v1/listen",
        "model": "nova-3",
        "language": "en-US",
        "sample_rate": 16000,
        "encoding": "linear16",
        "interim_results": true,
        "punctuate": true,
        "keep_alive": true
    }
}
```

### flux model

```json
{
    "params": {
        "key": "${env:DEEPGRAM_API_KEY}",
        "url": "wss://api.deepgram.com/v2/listen",
        "model": "flux-general-en",
        "sample_rate": 16000,
        "encoding": "linear16",
        "eager_eot_threshold": 0.6,
        "eot_threshold": 0.8,
        "eot_timeout_ms": 700
    }
}
```
