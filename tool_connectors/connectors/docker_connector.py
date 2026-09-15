"""Docker connector — read-only by default, per the spec: container status
and logs only. No start/stop/rm actions are exposed here at all, rather than
exposing them and relying on the safety tier alone — the smallest attack
surface is the one that isn't there."""

from __future__ import annotations

import shutil
import subprocess

from tool_connectors.base import (
    ActionSafety,
    Capability,
    ExecutionResult,
    ToolConnector,
    ToolExecutionError,
)


class DockerConnector(ToolConnector):
    name = "docker"

    def authenticate(self) -> bool:
        return True

    def health_check(self) -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            proc = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
            return proc.returncode == 0
        except Exception:
            return False

    def list_capabilities(self) -> list[Capability]:
        return [
            Capability(
                name="container_status",
                description="List running/stopped containers and their status.",
                safety=ActionSafety.READ_ONLY,
                parameters={"type": "OBJECT", "properties": {}},
            ),
            Capability(
                name="container_logs",
                description="Show the tail of a container's logs.",
                safety=ActionSafety.READ_ONLY,
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "container": {"type": "STRING", "description": "Container name or ID."},
                        "lines": {"type": "NUMBER", "description": "How many trailing lines (default 100)."},
                    },
                    "required": ["container"],
                },
            ),
        ]

    def execute(self, action: str, params: dict) -> ExecutionResult:
        if shutil.which("docker") is None:
            raise ToolExecutionError(self.name, action, "docker is not installed or not on PATH.")

        if action == "container_status":
            out = self._docker("ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}")
            return ExecutionResult(True, out, out or "No containers found.")

        if action == "container_logs":
            container = params.get("container", "").strip()
            if not container:
                raise ToolExecutionError(self.name, action, "A container name or ID is required.")
            lines = str(int(params.get("lines", 100) or 100))
            out = self._docker("logs", "--tail", lines, container)
            return ExecutionResult(True, out, out or "(no log output)")

        raise ToolExecutionError(self.name, action, f"Unknown action '{action}'.")

    def _docker(self, *args: str) -> str:
        try:
            proc = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=30)
        except Exception as e:
            raise ToolExecutionError(self.name, args[0] if args else "docker", f"docker invocation failed: {e}", original=e)

        if proc.returncode != 0:
            raise ToolExecutionError(self.name, args[0] if args else "docker", proc.stderr.strip() or "docker command failed.")
        return proc.stdout.strip()
