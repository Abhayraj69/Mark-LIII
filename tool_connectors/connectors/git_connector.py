"""Git connector — status/diff/branch info are READ_ONLY; commit is
REVERSIBLE (git revert/reset can undo it) and always goes through the
confirmation gate before it actually runs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tool_connectors.audit import get_base_dir
from tool_connectors.base import (
    ActionSafety,
    Capability,
    ExecutionResult,
    ToolConnector,
    ToolExecutionError,
)


class GitConnector(ToolConnector):
    name = "git"

    def __init__(self, repo_path: str | Path | None = None):
        self.repo_path = Path(repo_path or get_base_dir()).resolve()

    def authenticate(self) -> bool:
        # Local git operations over the existing working copy need no
        # separate credentials — reachability of the binary and repo is the
        # real gate, covered by health_check().
        return True

    def health_check(self) -> bool:
        if shutil.which("git") is None:
            return False
        return (self.repo_path / ".git").exists()

    def list_capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="status",
                description="Show working-tree status (staged/unstaged/untracked files).",
                safety=ActionSafety.READ_ONLY,
                parameters={"type": "OBJECT", "properties": {}},
            ),
            Capability(
                name="diff",
                description="Show the current unstaged diff, optionally for one file.",
                safety=ActionSafety.READ_ONLY,
                parameters={
                    "type": "OBJECT",
                    "properties": {"path": {"type": "STRING", "description": "Optional file path to limit the diff to."}},
                },
            ),
            Capability(
                name="branch_info",
                description="Show the current branch and its upstream tracking status.",
                safety=ActionSafety.READ_ONLY,
                parameters={"type": "OBJECT", "properties": {}},
            ),
            Capability(
                name="commit",
                description="Commit currently staged changes with a message.",
                safety=ActionSafety.REVERSIBLE,
                parameters={
                    "type": "OBJECT",
                    "properties": {"message": {"type": "STRING", "description": "Commit message."}},
                    "required": ["message"],
                },
            ),
        ]

    def execute(self, action: str, params: dict) -> ExecutionResult:
        if not self.health_check():
            raise ToolExecutionError(self.name, action, "git is not installed or this is not a git repository.")

        if action == "status":
            out = self._git("status", "--short", "--branch")
            return ExecutionResult(True, out, out or "Working tree clean.")

        if action == "diff":
            path = params.get("path")
            args = ["diff"] + ([path] if path else [])
            out = self._git(*args)
            return ExecutionResult(True, out, out or "No unstaged changes.")

        if action == "branch_info":
            branch = self._git("rev-parse", "--abbrev-ref", "HEAD")
            upstream = self._git("status", "-sb")
            return ExecutionResult(True, {"branch": branch, "upstream": upstream}, f"On {branch}. {upstream}")

        if action == "commit":
            message = params.get("message", "").strip()
            if not message:
                raise ToolExecutionError(self.name, action, "A commit message is required.")
            out = self._git("commit", "-m", message)
            return ExecutionResult(True, out, out)

        raise ToolExecutionError(self.name, action, f"Unknown action '{action}'.")

    def _git(self, *args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.repo_path), *args],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as e:
            raise ToolExecutionError(self.name, args[0] if args else "git", f"git invocation failed: {e}", original=e)

        if proc.returncode != 0:
            raise ToolExecutionError(self.name, args[0] if args else "git", proc.stderr.strip() or "git command failed.")
        return proc.stdout.strip()
