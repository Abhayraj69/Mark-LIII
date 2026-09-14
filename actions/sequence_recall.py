"""
actions/sequence_recall.py — multi-step action recall ("macros").

Lets the assistant save a named list of tool calls once (`manage_sequence`
with action="save") and replay every step, in order, on request later
(action="run"). Backed by the permanent store in core/sequence_memory.py, so
a sequence survives restarts and is never trimmed the way long_term.json
facts can be under memory pressure.

Execution model: this handler does not know how to run any tool itself — it
asks main.py to, through the `dispatch` context callable that
core/action_loader.py now threads into every handler alongside player/speak/
response/session_memory (see core/action_loader.py's _CTX_KEYS and
main.py's _dispatch_tool). That keeps exactly one tool router in the app;
this file only sequences calls into it.
"""

from __future__ import annotations

from core.sequence_memory import (
    delete_sequence,
    get_sequence,
    list_sequences,
    record_run,
    save_sequence,
)

# Replaying a sequence must never be able to record/run another sequence —
# that would allow infinite or exponential recursion (A calls B calls A).
_FORBIDDEN_STEP_TOOLS = {"manage_sequence"}


def _format_sequence(seq) -> str:
    lines = [f"'{seq.name}' — {len(seq.steps)} step(s)"]
    if seq.description:
        lines.append(f"  {seq.description}")
    for i, step in enumerate(seq.steps, 1):
        extra = f" ({step.note})" if step.note else ""
        lines.append(f"  {i}. {step.tool} {step.args or ''}{extra}")
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

    if action in ("run", "recall"):
        seq = get_sequence(name)
        if seq is None:
            return f"No saved sequence named '{name}'. Use action='list' to see what's saved."
        if dispatch is None:
            return "Sequence replay is unavailable in this context (no dispatcher)."

        results = []
        for i, step in enumerate(seq.steps, 1):
            if step.tool in _FORBIDDEN_STEP_TOOLS:
                results.append(f"{i}. {step.tool}: skipped (not allowed inside a sequence)")
                continue
            try:
                outcome = dispatch(step.tool, dict(step.args))
                results.append(f"{i}. {step.tool}: {outcome}")
            except Exception as e:
                results.append(f"{i}. {step.tool}: failed ({e})")
                break  # stop the sequence at the first hard failure
        record_run(name)
        if speak:
            try:
                speak(f"Ran sequence '{name}', {len(results)} of {len(seq.steps)} step(s).")
            except Exception:
                pass
        return f"Sequence '{name}' complete:\n" + "\n".join(results)

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

    return "Specify action: save | run | list | show | delete."


TOOL = {
    "name": "manage_sequence",
    "description": (
        "Save, replay, list, inspect, or delete a named multi-step action sequence "
        "('macro'). Use action='save' when the user says 'remember these steps as X' "
        "or 'save this as a routine called X' — pass 'name' and 'steps' (a list of "
        "{tool, args} objects describing each tool call to replay, in order; you "
        "decide the steps from the tools you have available or from what was just "
        "done in this conversation). Use action='run' (or 'recall') when the user "
        "says 'run X', 'do my X routine', or 'recall X' — this replays every saved "
        "step for that name. Use action='list' to see saved sequence names, "
        "action='show' to see one sequence's steps, and action='delete' to remove one. "
        "Sequences persist permanently until explicitly deleted."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {
                "type": "STRING",
                "description": "save | run | recall | list | show | delete",
            },
            "name": {
                "type": "STRING",
                "description": "The sequence's name (required for save/run/recall/show/delete).",
            },
            "description": {
                "type": "STRING",
                "description": "Optional one-line description of what the sequence does (save only).",
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
        },
        "required": ["action"],
    },
    "handler": manage_sequence,
}
