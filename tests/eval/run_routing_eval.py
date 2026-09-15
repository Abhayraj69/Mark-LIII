"""
Tool-routing evaluation harness — the measuring stick for "utterance -> correct
tool" behaviour, which nothing else in the test suite checks.

WHY this exists
----------------
prompt.txt is 11 KB of routing rules, the tool set spans 8 inline tools, 23
bundled actions, and any enabled plugins/connectors, and all of it is
translated to two more wire formats (core/tool_schema.py). Every new action,
every edited prompt line, and every model swap can silently steal routing
from another tool, and none of that shows up in the 170+ unit tests, because
they mock the LLM rather than asking one to route real utterances.

This script asks the REAL configured backends (Ollama/OpenAI-compatible via
core/llm_client.py, and Claude via core/claude_bridge.py) to pick a tool for
each case in routing_cases.jsonl, using the SAME tool declarations and system
prompt main.py builds — imported directly, without starting the Qt UI or a
Gemini Live session (Gemini Live can't be driven from a script; its agreement
with the text backends is not measured here). It also runs every case through
core/fast_intent.py's local shortcut matcher. No tool call is ever executed —
_dispatch_tool is never imported or invoked — this only ever reads the first
tool_call a backend proposes.

Usage
-----
    python tests/eval/run_routing_eval.py                  # score, print report
    python tests/eval/run_routing_eval.py --compare         # + fail (exit 1) on
                                                              a >2-point accuracy
                                                              regression vs. baseline
    python tests/eval/run_routing_eval.py --save-baseline    # overwrite baseline.json
    python tests/eval/run_routing_eval.py --backends ollama  # only run one backend
    python tests/eval/run_routing_eval.py --ollama-model qwen3:1.7b   # override the
                                                              configured model for
                                                              this run only (never
                                                              written to disk)

See README.md in this directory for the full corpus format and CI wiring.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_ROOT     = _EVAL_DIR.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CASES_PATH    = _EVAL_DIR / "routing_cases.jsonl"
BASELINE_PATH = _EVAL_DIR / "baseline.json"

REGRESSION_THRESHOLD = 0.02   # 2 percentage points


def _load_cases(path: Path = CASES_PATH) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {e}") from e
    return cases


def _build_tool_declarations() -> tuple[list[dict], str]:
    """Mirrors JarvisLive.__init__ / _all_tool_declarations (main.py) and
    _load_system_prompt — imported directly so this harness sees the exact
    tool set and prompt text the real app would, without constructing a
    JarvisLive (which needs a live UI, mic, and Gemini session)."""
    import main as jarvis_main
    from core.action_loader import discover_actions
    from core.plugin_loader import discover_plugins
    from tool_connectors.registry import ToolRegistry

    base_dir = Path(jarvis_main.__file__).resolve().parent
    _quiet = lambda _msg: None

    inline_names = {t["name"] for t in jarvis_main.TOOL_DECLARATIONS}
    action_registry = discover_actions(
        actions_dir=base_dir / "actions", reserved_names=inline_names, logger=_quiet,
    )
    core_names = inline_names | action_registry.names()
    plugin_registry = discover_plugins(
        plugins_dir=base_dir / "plugins", core_tool_names=core_names, logger=_quiet,
    )
    names_incl_plugins = core_names | {d["name"] for d in plugin_registry.get_tool_declarations()}
    connector_registry = ToolRegistry(logger=_quiet).discover()
    connector_decls = connector_registry.get_tool_declarations(reserved_names=names_incl_plugins)

    declarations = (
        jarvis_main.TOOL_DECLARATIONS
        + action_registry.get_tool_declarations()
        + plugin_registry.get_tool_declarations()
        + connector_decls
    )
    system_prompt = jarvis_main._load_system_prompt()
    return declarations, system_prompt


def _first_tool_call(resp: dict) -> tuple[str | None, dict]:
    calls = resp.get("tool_calls") or []
    if not calls:
        return None, {}
    fn = calls[0].get("function", {})
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return fn.get("name"), args


def _args_subset_matches(expected: dict, actual: dict) -> bool:
    return all(actual.get(k) == v for k, v in expected.items())


def _run_fast_intent(cases: list[dict]) -> dict:
    from core import fast_intent

    results = []
    for case in cases:
        intent = fast_intent.detect(case["text"])
        predicted = intent.tool if intent else None
        args = intent.args if intent else {}
        results.append(_score_case(case, predicted, args))
    return _summarize("fast_intent", results)


def _run_backend(name: str, cases: list[dict], declarations: list[dict],
                  system_prompt: str, ollama_model: str | None) -> dict | None:
    if name == "ollama":
        from core import llm_client
        from core.tool_schema import gemini_tools_to_openai

        reachable = llm_client.ensure_ollama_running(timeout=3)
        if not reachable:
            print(f"[eval] SKIP ollama — server unreachable at "
                  f"{llm_client.get_llm_settings()[0]}")
            return None

        if ollama_model:
            _orig_get_llm_settings = llm_client.get_llm_settings
            url, _ = _orig_get_llm_settings()
            llm_client.get_llm_settings = lambda: (url, ollama_model)  # this-process only

        tools = gemini_tools_to_openai(declarations)

        def _ask(text: str) -> dict:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            return llm_client.call_llm(messages, tools, timeout=60)

    elif name == "claude":
        from core import claude_bridge
        from core.tool_schema import gemini_tools_to_anthropic

        cfg = claude_bridge._load_config() if hasattr(claude_bridge, "_load_config") else {}
        api_key, _, _ = claude_bridge.get_claude_settings(cfg)
        if not api_key:
            print("[eval] SKIP claude — no API key configured "
                  "(plugin_config.claude_engine.api_key)")
            return None

        tools = gemini_tools_to_anthropic(declarations)

        def _ask(text: str) -> dict:
            messages = [{"role": "user", "content": text}]
            return claude_bridge.call_claude(messages, tools, system=system_prompt, timeout=60)

    else:
        raise ValueError(f"unknown backend: {name}")

    results = []
    for case in cases:
        try:
            resp = _ask(case["text"])
            predicted, args = _first_tool_call(resp)
        except Exception as e:
            print(f"[eval] {name}: case {case['text']!r} raised {type(e).__name__}: {e}")
            predicted, args = "__error__", {}
        results.append(_score_case(case, predicted, args))
    return _summarize(name, results)


def _score_case(case: dict, predicted: str | None, args: dict) -> dict:
    expected = case.get("expect_tool")
    tool_correct = predicted == expected
    args_ok = True
    if tool_correct and expected and case.get("expect_args_subset"):
        args_ok = _args_subset_matches(case["expect_args_subset"], args)
    return {
        "text": case["text"], "expected": expected, "predicted": predicted,
        "tags": case.get("tags", []), "tool_correct": tool_correct, "args_ok": args_ok,
    }


def _summarize(name: str, results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["tool_correct"])
    per_tool: dict[str, dict] = {}
    for r in results:
        for key in {r["expected"], r["predicted"]} - {None}:
            slot = per_tool.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
        if r["expected"] is not None:
            if r["tool_correct"]:
                per_tool[r["expected"]]["tp"] += 1
            else:
                per_tool[r["expected"]]["fn"] += 1
        if r["predicted"] is not None and not r["tool_correct"]:
            per_tool.setdefault(r["predicted"], {"tp": 0, "fp": 0, "fn": 0})
            per_tool[r["predicted"]]["fp"] += 1

    per_tool_scored = {}
    for tool, c in per_tool.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else None
        recall    = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else None
        per_tool_scored[tool] = {"tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                                  "precision": precision, "recall": recall}

    failures = [r for r in results if not r["tool_correct"]]
    return {
        "backend":  name,
        "total":    total,
        "correct":  correct,
        "accuracy": (correct / total) if total else 0.0,
        "per_tool": per_tool_scored,
        "failures": failures,
    }


def _print_report(summary: dict) -> None:
    print(f"\n=== {summary['backend']} — {summary['correct']}/{summary['total']} "
          f"({summary['accuracy']*100:.1f}%) ===")
    for tool, s in sorted(summary["per_tool"].items()):
        p = f"{s['precision']*100:.0f}%" if s["precision"] is not None else "n/a"
        r = f"{s['recall']*100:.0f}%" if s["recall"] is not None else "n/a"
        print(f"  {tool:<20} precision={p:>5}  recall={r:>5}  "
              f"(tp={s['tp']} fp={s['fp']} fn={s['fn']})")
    if summary["failures"]:
        print("  --- failures ---")
        for f in summary["failures"][:20]:
            print(f"  {f['text']!r}: expected {f['expected']!r}, got {f['predicted']!r}"
                  f" {f['tags']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="ollama,claude",
                         help="comma-separated backend names to run (ollama, claude)")
    parser.add_argument("--ollama-model", default=None,
                         help="override the configured Ollama model for this run only")
    parser.add_argument("--compare", action="store_true",
                         help="exit 1 if any backend's accuracy regressed >2 points vs baseline.json")
    parser.add_argument("--save-baseline", action="store_true",
                         help="overwrite baseline.json with this run's results")
    parser.add_argument("--cases", default=str(CASES_PATH))
    args = parser.parse_args(argv)

    cases = _load_cases(Path(args.cases))
    declarations, system_prompt = _build_tool_declarations()
    print(f"[eval] {len(cases)} cases, {len(declarations)} tool declarations loaded.")

    all_summaries: dict[str, dict] = {}

    fast_summary = _run_fast_intent(cases)
    _print_report(fast_summary)
    all_summaries["fast_intent"] = fast_summary

    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        started = time.monotonic()
        summary = _run_backend(name, cases, declarations, system_prompt, args.ollama_model)
        if summary is None:
            continue
        summary["elapsed_s"] = round(time.monotonic() - started, 1)
        _print_report(summary)
        all_summaries[name] = summary

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_size":  len(cases),
        "backends": {
            name: {"accuracy": s["accuracy"], "total": s["total"], "correct": s["correct"]}
            for name, s in all_summaries.items()
        },
    }

    exit_code = 0
    if args.compare:
        if not BASELINE_PATH.exists():
            print(f"[eval] --compare requested but {BASELINE_PATH} does not exist yet — "
                  f"run with --save-baseline first.")
            exit_code = 1
        else:
            baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            for name, s in result["backends"].items():
                base = baseline.get("backends", {}).get(name)
                if base is None:
                    continue
                drop = base["accuracy"] - s["accuracy"]
                if drop > REGRESSION_THRESHOLD:
                    print(f"[eval] REGRESSION: {name} accuracy dropped "
                          f"{drop*100:.1f} points ({base['accuracy']*100:.1f}% -> "
                          f"{s['accuracy']*100:.1f}%)")
                    exit_code = 1
            if exit_code == 0:
                print("[eval] No regression vs baseline.")

    if args.save_baseline:
        BASELINE_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[eval] Wrote {BASELINE_PATH}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
