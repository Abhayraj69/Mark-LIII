"""
actions/sequence_recall.py — multi-step action recall ("macros").

Lets the assistant save a named list of tool calls once (`manage_sequence`
with action="save") and replay every step, in order, on request later
(action="run"/"recall"/"replay"). Two ways to build one:

  * Dictated up front — action="save" with an explicit `steps` list, the
    original mechanism.
  * Recorded live — action="record_start" begins capturing every tool call
    main.py's dispatcher makes from then on (see core/sequence_memory.py's
    record_step, hooked into main.py's _dispatch_tool), action="record_stop"
    saves what was captured, action="record_discard" throws it away.

Steps may contain {placeholder} values (e.g. "open {file}"); action="replay"
(an alias of "run"/"recall") accepts a `params` object to fill them in, and
action="parametrize" turns an already-recorded literal value into a
placeholder after the fact so a macro doesn't need to be re-recorded just to
generalize it.

Backed by the permanent store in core/sequence_memory.py, so a sequence
survives restarts and is never trimmed the way long_term.json facts can be
under memory pressure.

Execution model: this handler does not know how to run any tool itself — it
asks main.py to, through the `dispatch` context callable that
core/action_loader.py now threads into every handler alongside player/speak/
response/session_memory (see core/action_loader.py's _CTX_KEYS and
main.py's _dispatch_tool). That keeps exactly one tool router in the app;
this file only sequences calls into it.
"""

from __future__ import annotations

from core.sequence_memory import (
    _SELF_RECORDING_FORBIDDEN as _FORBIDDEN_STEP_TOOLS,
    delete_sequence,
    discard_recording,
    get_sequence,
    is_recording,
    list_sequences,
    parametrize as _parametrize,
    record_run,
    render_steps,
    save_sequence,
    start_recording,
    stop_recording,
)


def _format_sequence(seq) -> str:
    lines = [f"'{seq.name}' — {len(seq.steps)} step(s)"]
    if seq.description:
        lines.append(f"  {seq.description}")
    for i, step in enumerate(seq.steps, 1):
        extra = f" ({step.note})" if step.note else ""
        gated = " [requires confirmation]" if step.confirm else ""
        lines.append(f"  {i}. {step.tool} {step.args or ''}{extra}{gated}")
    if seq.run_count:
        lines.append(f"  Run {seq.run_count} time(s); last: {seq.last_run or 'never'}")
    return "\n".join(lines)


def manage_sequence(parameters: dict, speak=None, dispatch=None) -> str:
    action = (parameters.get("action") or "").strip().lower()
    name = (parameters.get("name") or "").strip()

    if action == "save":
        steps = parameters.get("steps")
        if not isinstance(steps, list):
            return "Provide 'steps' as a list of {tool, args} objects to save a sequence."
        for step in steps:
            tool = (step.get("tool") or "").strip() if isinstance(step, dict) else ""
            if tool in _FORBIDDEN_STEP_TOOLS:
                return f"A sequence cannot contain '{tool}' as a step."
        try:
            return save_sequence(name, steps, parameters.get("description", ""))
        except ValueError as e:
            return f"Could not save sequence: {e}"

    if action in ("run", "recall", "replay"):
        seq = get_sequence(name)
        if seq is None:
            return f"No saved sequence named '{name}'. Use action='list' to see what's saved."
        if dispatch is None:
            return "Sequence replay is unavailable in this context (no dispatcher)."

        params = parameters.get("params")
        params = params if isinstance(params, dict) else {}
        steps, missing = render_steps(seq, params)
        if missing:
            return (f"Sequence '{name}' needs {{{'}, {'.join(sorted(missing))}}} — "
                     f"pass them in 'params' to replay it.")

        results = []
        paused = False
        for i, step in enumerate(steps, 1):
            if step.tool in _FORBIDDEN_STEP_TOOLS:
                results.append(f"{i}. {step.tool}: skipped (not allowed inside a sequence)")
                continue
            try:
                outcome = dispatch(step.tool, dict(step.args))
                results.append(f"{i}. {step.tool}: {outcome}")
                if step.confirm and isinstance(outcome, str) and outcome.startswith("[CONFIRMATION_PENDING]"):
                    remaining = len(steps) - i
                    if remaining:
                        results.append(f"Paused — {remaining} remaining step(s) will not run until confirmed.")
                    paused = True
                    break
            except Exception as e:
                results.append(f"{i}. {step.tool}: failed ({e})")
                break  # stop the sequence at the first hard failure
        if not paused:
            record_run(name)
        if speak:
            try:
                verb = "Reached a confirmation in" if paused else "Ran"
                speak(f"{verb} sequence '{name}', {len(results)} of {len(steps)} step(s).")
            except Exception:
                pass
        return f"Sequence '{name}' {'paused' if paused else 'complete'}:\n" + "\n".join(results)

    if action == "record_start":
        if is_recording():
            return "Already recording — say 'stop recording' or 'discard recording' first."
        try:
            return start_recording(name, parameters.get("description", ""))
        except ValueError as e:
            return f"Could not start recording: {e}"

    if action == "record_stop":
        try:
            return stop_recording()
        except ValueError as e:
            return f"Could not stop recording: {e}"

    if action == "record_discard":
        try:
            return discard_recording()
        except ValueError as e:
            return f"Could not discard recording: {e}"

    if action == "parametrize":
        value = parameters.get("value")
        placeholder = parameters.get("placeholder")
        if not value or not placeholder:
            return "Provide both 'value' (the literal text to replace) and 'placeholder' (its new name)."
        try:
            return _parametrize(name, str(value), str(placeholder))
        except ValueError as e:
            return f"Could not parametrize: {e}"

    if action == "list":
        seqs = list_sequences()
        if not seqs:
            return "No sequences saved yet."
        return "Saved sequences:\n" + "\n".join(f"- {s.name} ({len(s.steps)} steps)" for s in seqs)

    if action == "show":
        seq = get_sequence(name)
        return _format_sequence(seq) if seq else f"No saved sequence named '{name}'."

    if action == "delete":
        return f"Deleted sequence '{name}'." if delete_sequence(name) else f"No saved sequence named '{name}'."

    return ("Specify action: save | record_start | record_stop | record_discard | "
            "run | recall | replay | parametrize | list | show | delete.")


TOOL = {
    "name": "manage_sequence",
    "description": (
        "Save, record, replay, parametrize, list, inspect, or delete a named "
        "multi-step action sequence ('macro'). Two ways to create one: "
        "action='save' when the user dictates the steps up front ('remember these "
        "steps as X') — pass 'name' and 'steps' (a list of {tool, args} objects, "
        "in order; you decide them from the tools available or what was just done "
        "in this conversation); or action='record_start' when the user says "
        "'record this as X' / 'start recording' — every tool call made from then "
        "on becomes a step automatically, until action='record_stop' ('stop "
        "recording') saves it or action='record_discard' throws it away. "
        "Use action='run' (or 'recall'/'replay') when the user says 'run X', 'do "
        "my X routine', or 'run X with file=Y' — this replays every saved step; "
        "pass 'params' (an object) to fill in any {placeholder} values a step's "
        "args contain, e.g. {\"file\": \"report.pdf\"}. Use action='parametrize' "
        "to turn an already-saved literal value into a {placeholder} for future "
        "replays — pass 'name', 'value' (the exact text to replace), and "
        "'placeholder' (its new name). Use action='list' to see saved sequence "
        "names, action='show' to see one sequence's steps, and action='delete' to "
        "remove one. Sequences persist permanently until explicitly deleted."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": ("save | record_start | record_stop | record_discard | "
                                 "run | recall | replay | parametrize | list | show | delete"),
            },
            "name": {
                "type": "STRING",
                "description": ("The sequence's name (required for everything except "
                                 "record_stop/record_discard/list, which act on whatever "
                                 "recording is currently in progress or on all saved sequences)."),
            },
            "description": {
                "type": "STRING",
                "description": "Optional one-line description of what the sequence does (save/record_start only).",
            },
            "steps": {
                "type": "ARRAY",
                "description": (
                    "Ordered list of steps to save, each an object with 'tool' "
                    "(the exact tool name to call), 'args' (its parameters object), "
                    "and optionally 'note' (why this step exists). Required for save."
                ),
                "items": {"type": "OBJECT"},
            },
            "params": {
                "type": "OBJECT",
                "description": (
                    "Values to fill in for any {placeholder} in the sequence's step "
                    "args, e.g. {\"file\": \"report.pdf\"} for a step using \"{file}\". "
                    "Only used by run/recall/replay."
                ),
            },
            "value": {
                "type": "STRING",
                "description": "The exact literal text to replace with a placeholder (parametrize only).",
            },
            "placeholder": {
                "type": "STRING",
                "description": "The new placeholder name, e.g. 'file' or 'city' (parametrize only).",
            },
        },
        "required": ["action"],
    },
    "handler": manage_sequence,
}
