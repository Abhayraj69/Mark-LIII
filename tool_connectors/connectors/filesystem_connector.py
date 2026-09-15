"""Local filesystem connector — read/search only, and every path is resolved
and checked against the project root before anything touches disk. No write,
delete, or move action exists on this connector at all: a read-only
capability list is what makes the sandbox trustworthy, not just the safety
tag next to it.
"""

from __future__ import annotations

from pathlib import Path

from tool_connectors.audit import get_base_dir
from tool_connectors.base import (
    ActionSafety,
    Capability,
    ExecutionResult,
    ToolConnector,
    ToolExecutionError,
)

_MAX_READ_BYTES = 200_000
_MAX_SEARCH_HITS = 200
_MAX_SEARCH_FILES = 5000


class FileSystemConnector(ToolConnector):
    name = "filesystem"

    def __init__(self, project_root: str | Path | None = None):
        # Defaults to the JARVIS project root (not the process cwd, which can
        # differ depending on how JARVIS was launched) so auto-discovery
        # sandboxes to the right directory with no config required.
        self.project_root = Path(project_root or get_base_dir()).resolve()

    def authenticate(self) -> bool:
        return True

    def health_check(self) -> bool:
        return self.project_root.is_dir()

    def list_capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="read_file",
                description="Read a text file's contents, given a path relative to the project root.",
                safety=ActionSafety.READ_ONLY,
                parameters={
                    "type": "OBJECT",
                    "properties": {"path": {"type": "STRING", "description": "Path relative to the project root."}},
                    "required": ["path"],
                },
            ),
            Capability(
                name="list_dir",
                description="List entries in a directory relative to the project root.",
                safety=ActionSafety.READ_ONLY,
                parameters={
                    "type": "OBJECT",
                    "properties": {"path": {"type": "STRING", "description": "Directory relative to the project root; '.' for root."}},
                },
            ),
            Capability(
                name="search_files",
                description="Search for a text substring across files under the project root.",
                safety=ActionSafety.READ_ONLY,
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Substring to search for."},
                        "path": {"type": "STRING", "description": "Subdirectory to restrict the search to (optional)."},
                    },
                    "required": ["query"],
                },
            ),
        ]

    def execute(self, action: str, params: dict) -> ExecutionResult:
        if action == "read_file":
            target = self._resolve(params.get("path", ""))
            if not target.is_file():
                raise ToolExecutionError(self.name, action, f"No such file: {params.get('path')}")
            try:
                data = target.read_bytes()[:_MAX_READ_BYTES]
                text = data.decode("utf-8", errors="replace")
            except Exception as e:
                raise ToolExecutionError(self.name, action, f"Could not read file: {e}", original=e)
            return ExecutionResult(True, text, text)

        if action == "list_dir":
            target = self._resolve(params.get("path", "."))
            if not target.is_dir():
                raise ToolExecutionError(self.name, action, f"No such directory: {params.get('path', '.')}")
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return ExecutionResult(True, entries, "\n".join(entries) or "(empty)")

        if action == "search_files":
            query = params.get("query", "")
            if not query:
                raise ToolExecutionError(self.name, action, "A search query is required.")
            base = self._resolve(params.get("path", "."))
            if not base.is_dir():
                raise ToolExecutionError(self.name, action, f"No such directory: {params.get('path', '.')}")
            hits = self._search(base, query)
            summary = "\n".join(f"{p}:{n}: {line}" for p, n, line in hits) or "No matches."
            return ExecutionResult(True, hits, summary)

        raise ToolExecutionError(self.name, action, f"Unknown action '{action}'.")

    # -- sandbox enforcement --------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Resolves rel_path against the project root and refuses anything
        that escapes it — via '..', an absolute path, or a symlink pointing
        outside — by checking the final resolved path is still inside root."""
        candidate = (self.project_root / (rel_path or ".")).resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError:
            raise ToolExecutionError(
                self.name, "path", f"'{rel_path}' is outside the project root — refused.",
            )
        return candidate

    def _search(self, base: Path, query: str) -> list[tuple[str, int, str]]:
        hits: list[tuple[str, int, str]] = []
        files_scanned = 0
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            files_scanned += 1
            if files_scanned > _MAX_SEARCH_FILES:
                break
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    hits.append((str(path.relative_to(self.project_root)), i, line.strip()[:200]))
                    if len(hits) >= _MAX_SEARCH_HITS:
                        return hits
        return hits
