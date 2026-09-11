from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.models import (
    AgentConfig,
    ContextConfig,
    HarnessConfig,
    LLMEndpointConfig,
)
from metaharness.state import RunStateStore
from metaharness.web.server import create_server


class WebSecurityTests(unittest.TestCase):
    def test_untrusted_plan_review_and_paths_are_html_escaped(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = root / "runs"
            run = runs / "xss"
            RunStateStore(run / "state.json").initialize(
                "xss", worktree="<img src=x onerror=alert(1)>"
            )
            (run / "planner.raw.md").write_text(
                "<script>window.PWNED=true</script>", encoding="utf-8"
            )
            (run / "reviewer.raw.md").write_text(
                "<img src=x onerror=alert(1)>", encoding="utf-8"
            )
            (run / "review.json").write_text(
                '{"summary":"<script>window.PWNED=true</script>","findings":"<img src=x onerror=alert(1)>"}',
                encoding="utf-8",
            )
            endpoint = LLMEndpointConfig("https://example.invalid", "/chat", "model")
            config = HarnessConfig(
                repo=root,
                base_ref="HEAD",
                runs_root=runs,
                worktrees_root=root / "worktrees",
                require_clean_base=True,
                planner=endpoint,
                reviewer=endpoint,
                context=ContextConfig(),
                agent=AgentConfig(),
                checks=(),
                allow_no_required_checks=True,
            )
            server = create_server(config, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_port)
                connection.request("GET", "/runs/xss")
                response = connection.getresponse()
                content = response.read().decode("utf-8")
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            self.assertEqual(response.status, 200)
            self.assertIn("&lt;script&gt;window.PWNED=true&lt;/script&gt;", content)
            self.assertIn("&lt;img src=x onerror=alert(1)&gt;", content)
            self.assertNotIn("<script>window.PWNED", content)
            self.assertNotIn('<img src=x onerror=alert(1)>', content)


if __name__ == "__main__":
    unittest.main()
