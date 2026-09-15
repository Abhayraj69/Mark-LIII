# Tool-routing eval harness

Measures the one thing the unit test suite doesn't: given an utterance, does
the assistant call the *right* tool (or correctly call none)? `prompt.txt`,
the 27+ tool declarations, and three wire-format translations can all drift
independently — this is the regression net for that.

## Files

- `routing_cases.jsonl` — the corpus. One JSON object per line:
  ```json
  {"text": "volume up", "expect_tool": "computer_settings",
   "expect_args_subset": {"action": "volume_up"}, "tags": ["fast_intent"]}
  ```
  - `expect_tool`: the tool name expected, or `null` if the utterance should
    produce prose with no tool call at all.
  - `expect_args_subset` (optional): a dict of args that must appear in the
    call's arguments (subset match — extra args are fine).
  - `tags`: free-form (`action`, `inline`, `fast_intent`, `prose`, `ambiguous`)
    used only for readability of the failure list, not for scoring.
- `run_routing_eval.py` — the runner (see its module docstring for full usage).
- `baseline.json` — last committed scores, written by `--save-baseline`.

## Running it

```bash
python tests/eval/run_routing_eval.py                        # score + print report
python tests/eval/run_routing_eval.py --backends ollama       # just one backend
python tests/eval/run_routing_eval.py --ollama-model qwen3:1.7b   # use a different
                                                                    # pulled model for
                                                                    # this run only —
                                                                    # never touches config
python tests/eval/run_routing_eval.py --save-baseline         # commit new baseline.json
python tests/eval/run_routing_eval.py --compare               # exit 1 on >2pt regression
```

It always also runs every case through `core/fast_intent.py`'s local matcher
— that's not a "backend" with config to skip, so it always reports.

Ollama is skipped automatically if unreachable; Claude is skipped
automatically if no API key is configured. Skips are logged, never a crash —
this mirrors `plugin_loader.py`'s crash-isolation stance. **No tool is ever
executed** — the harness only reads the first `tool_call` a backend proposes
and never imports `_dispatch_tool`.

## CI / pytest wiring

The normal `pytest -q` run stays offline and fast — this harness needs a
live Ollama server (and optionally a Claude API key) and takes much longer
per case than a mocked unit test. It only runs when opted in:

```bash
JARVIS_ROUTING_EVAL=1 python -m pytest tests/test_routing_eval.py -q
```

That test calls `run_routing_eval.main(["--compare"])` and asserts a
non-regressing exit code. Update `baseline.json` deliberately (`--save-baseline`)
whenever a prompt or tool-declaration change is expected to move the numbers,
and note the before/after in the PR description — this file is a tripwire,
not a moving target to silently keep green.

## Extending the corpus

Add a line per new action (at least one clearly-routed case, ideally a second
phrased so it wouldn't trivially pattern-match). If the new action overlaps
in meaning with an existing one, add an `"tags": ["ambiguous"]` case so the
confusion table surfaces it instead of one silently starving the other of
routing share.
