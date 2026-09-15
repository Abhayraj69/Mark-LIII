"""Wires the tool-routing eval harness (tests/eval/) into pytest, opt-in only.

WHY opt-in: the harness needs a live Ollama server (and optionally a Claude
API key) and takes far longer per case than a mocked unit test — running it
on every `pytest -q` would make the normal offline suite slow and flaky on
machines without a local LLM running. Set JARVIS_ROUTING_EVAL=1 to include it,
e.g.: JARVIS_ROUTING_EVAL=1 python -m pytest tests/test_routing_eval.py -q
"""
import os
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_EVAL_DIR = _ROOT / "tests" / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))


@unittest.skipUnless(
    os.environ.get("JARVIS_ROUTING_EVAL") == "1",
    "set JARVIS_ROUTING_EVAL=1 to run the live tool-routing eval (needs Ollama/Claude)",
)
class TestRoutingEvalNoRegression(unittest.TestCase):
    def test_no_accuracy_regression_vs_baseline(self):
        import run_routing_eval  # noqa: E402 — tests/eval/run_routing_eval.py

        exit_code = run_routing_eval.main(["--compare"])
        self.assertEqual(exit_code, 0, "routing accuracy regressed vs. tests/eval/baseline.json")


if __name__ == "__main__":
    unittest.main()
