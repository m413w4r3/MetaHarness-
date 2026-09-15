"""Direct publication-boundary checks kept under the requested test name."""

from __future__ import annotations

import inspect
import unittest

from metaharness.gitops import commit_candidate_tree, push_run_branch
from metaharness.orchestrator import Orchestrator


class PublicationBoundaryTests(unittest.TestCase):
    def test_candidate_primitives_and_final_publication_are_separate(self) -> None:
        self.assertTrue(callable(commit_candidate_tree))
        self.assertTrue(callable(push_run_branch))
        source = inspect.getsource(Orchestrator._complete_candidate_publication)
        self.assertIn("publish_fast_forward_base", source)
        self.assertNotIn("commit_candidate_tree", source)


if __name__ == "__main__":
    unittest.main()
