from __future__ import annotations

import json
import unittest

from metaharness.redaction import redact, redact_mapping


class RedactionTests(unittest.TestCase):
    def test_text_redaction_rejects_structured_failure_data(self) -> None:
        with self.assertRaisesRegex(TypeError, "human-readable text"):
            redact({"reason": "CHECK_REPAIR_EXHAUSTED"}, ())  # type: ignore[arg-type]

    def test_structured_redaction_preserves_shape_and_redacts_escaped_values(self) -> None:
        secret = 'token-"slash\\value-12345'
        detail = {
            "exception_type": "RuntimeError",
            "message": f"failure: {secret}",
            "nested": {"credential": secret},
            "items": [secret, 2],
        }

        redacted = redact_mapping(detail, (secret,))

        self.assertEqual(redacted["exception_type"], "RuntimeError")
        self.assertEqual(redacted["message"], "failure: [REDACTED]")
        self.assertEqual(redacted["nested"], {"credential": "[REDACTED]"})
        self.assertEqual(redacted["items"], ["[REDACTED]", 2])
        self.assertNotIn(secret, json.dumps(redacted))


if __name__ == "__main__":
    unittest.main()
