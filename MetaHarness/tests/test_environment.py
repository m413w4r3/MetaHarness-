import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.config import load_config
from metaharness.environment import EnvironmentFileError, build_runtime_environment, parse_env_file


class EnvironmentFileTests(unittest.TestCase):
    def test_supported_assignments_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("# comment\nA=plain\nexport B=two\nC=\"quoted\"\nD='single'\nE=\n", encoding="utf-8")
            self.assertEqual(parse_env_file(path), {"A": "plain", "B": "two", "C": "quoted", "D": "single", "E": ""})

    def test_duplicate_is_rejected_without_value(self) -> None:
        secret = "env-file-secret-value"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(f"TOKEN={secret}\nTOKEN=other\n", encoding="utf-8")
            with self.assertRaises(EnvironmentFileError) as raised:
                parse_env_file(path)
            self.assertNotIn(secret, str(raised.exception))

    def test_merge_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.write_text("A=one\nB=first\n", encoding="utf-8")
            second.write_text("B=second\nC=three\n", encoding="utf-8")
            self.assertEqual(
                build_runtime_environment((first, second), {"C": "process", "D": "four"}),
                {"A": "one", "B": "second", "C": "process", "D": "four"},
            )

    def test_config_expands_from_runtime_mapping_and_does_not_leak_secret(self) -> None:
        secret = "runtime-secret-value"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_file = root / ".env"
            env_file.write_text(f"BRIDGE_API_KEY={secret}\nVALUE=from-file\n", encoding="utf-8")
            config_file = root / "config.toml"
            config_file.write_text(
                'repo = "."\nbase_ref = "HEAD"\nruns_root = "runs"\nworktrees_root = "worktrees"\nallow_no_required_checks = true\n'
                '[environment]\nfiles = [".env"]\n'
                '[ui]\ndefault_planner_profile = "planner"\ndefault_implementer_profile = "worker"\ndefault_reviewer_profile = "reviewer"\n'
                '[model_profiles.planner]\ndisplay_name = "Planner"\nroles = ["planner"]\ndriver = "openai-chat"\nprovider = "bridge"\nmodel = "m"\nselection_mode = "request"\nbase_url = "http://127.0.0.1:1"\nendpoint_path = "/${VALUE}"\napi_key_env = "BRIDGE_API_KEY"\n'
                '[model_profiles.worker]\ndisplay_name = "Worker"\nroles = ["implementer"]\ndriver = "external"\nprovider = "bridge"\nmodel = "m"\nselection_mode = "cli"\nargv = ["worker"]\n'
                '[model_profiles.reviewer]\ndisplay_name = "Reviewer"\nroles = ["reviewer"]\ndriver = "openai-chat"\nprovider = "bridge"\nmodel = "m"\nselection_mode = "request"\nbase_url = "http://127.0.0.1:1"\nendpoint_path = "/review"\n'
                '',
                encoding="utf-8",
            )
            config = load_config(config_file)
            self.assertEqual(config.planner.endpoint_path, "/from-file")
            self.assertEqual(config.runtime_environment["BRIDGE_API_KEY"], secret)
            self.assertNotIn(secret, repr(config))


if __name__ == "__main__":
    unittest.main()
