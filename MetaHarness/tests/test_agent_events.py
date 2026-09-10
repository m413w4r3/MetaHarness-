import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.agent.events import (
    extract_final,
    extract_usage,
    iter_events,
    parse_event,
)


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


if __name__ == "__main__":
    unittest.main()
