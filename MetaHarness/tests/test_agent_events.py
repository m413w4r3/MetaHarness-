import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.agent.events import (
    extract_final,
    extract_terminal_result,
    extract_usage,
    iter_events,
    parse_event,
)
from metaharness.claude.agent import _scan_events


class AgentEventsTests(unittest.TestCase):
    def test_malformed_lines_are_ignored(self) -> None:
        events = list(iter_events(["not json\n", "[]\n", '{"type":"started"}\n']))

        self.assertEqual(events, [{"type": "started"}])
        self.assertIsNone(parse_event("{broken"))

    def test_usage_is_found_in_codex_envelope(self) -> None:
        event = {
            "type": "turn.completed",
            "msg": {"usage": {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19}},
        }

        self.assertEqual(
            extract_usage(event),
            {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        )

    def test_final_message_is_found_in_completed_turn(self) -> None:
        event = {
            "type": "turn.completed",
            "item": {"type": "agent_message", "text": "Implemented."},
        }

        self.assertEqual(extract_final(event), "Implemented.")

    def test_terminal_success_is_extracted(self) -> None:
        event = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "num_turns": 3,
            "stop_reason": "end_turn",
            "errors": [],
            "unknown": "ignored",
        }

        self.assertEqual(
            extract_terminal_result(event),
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 3,
                "stop_reason": "end_turn",
                "errors": (),
            },
        )

    def test_terminal_error_max_turns_is_extracted(self) -> None:
        event = {
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": True,
            "num_turns": 13,
            "stop_reason": "max_turns",
            "errors": ["turn limit reached"],
        }

        terminal = extract_terminal_result(event)

        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertEqual(terminal["subtype"], "error_max_turns")
        self.assertTrue(terminal["is_error"])
        self.assertEqual(terminal["num_turns"], 13)
        self.assertEqual(terminal["errors"], ("turn limit reached",))

    def test_terminal_without_num_turns_is_supported(self) -> None:
        terminal = extract_terminal_result({"type": "result", "subtype": "success"})

        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertIsNone(terminal["num_turns"])

    def test_malformed_terminal_errors_are_ignored(self) -> None:
        terminal = extract_terminal_result(
            {
                "type": "result",
                "errors": {"message": "not a list"},
                "num_turns": True,
                "is_error": 1,
                "stop_reason": 42,
            }
        )

        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertEqual(terminal["errors"], ())
        self.assertIsNone(terminal["num_turns"])
        self.assertIsNone(terminal["is_error"])
        self.assertIsNone(terminal["stop_reason"])

    def test_unknown_event_is_not_terminal(self) -> None:
        self.assertIsNone(extract_terminal_result({"type": "message", "subtype": "success"}))

    def test_scan_events_keeps_last_terminal_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(event)
                    for event in (
                        {"type": "result", "subtype": "success", "num_turns": 2},
                        {"type": "unknown", "subtype": "ignored"},
                        {"type": "result", "subtype": "error_max_turns", "num_turns": 13},
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            _, _, terminal = _scan_events(path)

        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertEqual(terminal["subtype"], "error_max_turns")
        self.assertEqual(terminal["num_turns"], 13)


if __name__ == "__main__":
    unittest.main()
