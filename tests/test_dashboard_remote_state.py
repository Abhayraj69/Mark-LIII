"""Unit tests for the phone dashboard's confirm/undo/suggestion parity with
the HUD — /api/confirm, /api/undo, /api/suggestion in dashboard/server.py.

Auth is bypassed by injecting a token directly into DashboardServer._tokens
(the same set /login populates) rather than exercising the AES/pairing flow,
which is what "crypto layer stubbed" means here — these endpoints don't touch
encryption at all, they're plain authenticated JSON POSTs."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None

if TestClient is not None:
    from dashboard.server import DashboardServer  # noqa: E402
    from core import confirm as confirm_gate  # noqa: E402
    from core import undo as undo_stack  # noqa: E402


@unittest.skipIf(TestClient is None, "fastapi not installed")
class TestDashboardRemoteState(unittest.TestCase):
    def setUp(self):
        confirm_gate._pending = None
        confirm_gate._show_cb = confirm_gate._hide_cb = confirm_gate._log_cb = None
        undo_stack._stack.clear()
        undo_stack._on_change = None

        self.server = DashboardServer()
        self.token = "test-token"
        self.server._tokens.add(self.token)
        self.client = TestClient(self.server.app)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def _auth_post(self, path, json=None):
        return self.client.post(path, json=json or {}, headers=self.headers)

    def test_confirm_requires_auth(self):
        resp = self.client.post("/api/confirm", json={"id": "x", "accept": True})
        self.assertEqual(resp.status_code, 401)

    def test_confirm_round_trip_accept(self):
        ran = []
        confirm_gate.bind(show=lambda t, d: None, hide=lambda: None, log=lambda m: None)
        sentence = confirm_gate.request(
            key="k1", title="Shut down", detail="", run=lambda: ran.append(True) or "done",
        )
        self.assertIn("CONFIRMATION_PENDING", sentence)
        pending = confirm_gate.pending_info()
        self.assertIsNotNone(pending)

        def _dashboard_confirm(confirm_id, accepted):
            current = confirm_gate.pending_info()
            if current is None or current["key"] != confirm_id:
                return False
            confirm_gate.resolve(accepted)
            return True

        self.server.set_confirm_callback(_dashboard_confirm)

        resp = self._auth_post("/api/confirm", {"id": pending["key"], "accept": True})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

        # single-use: resolving again with the same (now-stale) id is refused
        resp2 = self._auth_post("/api/confirm", {"id": pending["key"], "accept": True})
        self.assertFalse(resp2.json()["ok"])

    def test_confirm_stale_id_is_refused(self):
        confirm_gate.bind(show=lambda t, d: None, hide=lambda: None, log=lambda m: None)
        confirm_gate.request(key="real", title="X", detail="", run=lambda: "done")

        def _dashboard_confirm(confirm_id, accepted):
            current = confirm_gate.pending_info()
            if current is None or current["key"] != confirm_id:
                return False
            confirm_gate.resolve(accepted)
            return True

        self.server.set_confirm_callback(_dashboard_confirm)
        resp = self._auth_post("/api/confirm", {"id": "stale-id", "accept": True})
        self.assertFalse(resp.json()["ok"])

    def test_undo_pop(self):
        undo_stack.push_undo("moved file.txt", lambda: "restored")
        self.server.set_undo_callback(undo_stack.undo_last)

        resp = self._auth_post("/api/undo")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("moved file.txt", resp.json()["message"])
        self.assertFalse(undo_stack.can_undo())

    def test_suggestion_decision_invoked(self):
        seen = []
        self.server.set_suggestion_callback(lambda accepted: seen.append(accepted))

        resp = self._auth_post("/api/suggestion", {"accept": True})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(seen, [True])

    def test_endpoints_503_when_callback_unset(self):
        for path in ("/api/confirm", "/api/undo", "/api/suggestion"):
            resp = self._auth_post(path, {"id": "x", "accept": True})
            self.assertEqual(resp.status_code, 503, path)


if __name__ == "__main__":
    unittest.main()
