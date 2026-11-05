The completion regression runs without provider credentials or network calls.
It uses the real Gemini generator, LLM2 command producer, and the tracked
voice-assistant consumer through its TTS end message. A fake transport checks
the installed `llm-interface.json` result field types before delivering results.

After installing an example's TEN system packages, run from the repository root:

```sh
TEN_TEST_APP=/path/to/installed/tenapp
PYTHONPATH="$TEN_TEST_APP/ten_packages/system/ten_ai_base/interface:$TEN_TEST_APP/ten_packages/system/ten_runtime_python/interface" \
  python3 -m unittest discover \
  -s ai_agents/agents/ten_packages/extension/gemini_llm2_python/tests
```

`TEN_TEST_MAIN_ROOT` can select a separately installed `main_python` package.
Without it, the tests load the tracked voice-assistant example. The tests cover
streaming text accumulation, complete non-streaming text, empty streams, and
missing or null completion timestamps; they do not require a running agent.

The HTTP streaming regression uses a real `httpx.AsyncClient` with a gated
synthetic byte stream. It requires the first delta before provider EOF, checks
explicit stream completion, and verifies response cleanup on cancellation,
consumer close and HTTP errors, plus recovery of a subsequent request.
