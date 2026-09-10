import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.llm.wire import (  # noqa: E402
    AmbiguousFieldError,
    parse_labeled_document,
)


FIELDS = {"status": ("STATUS", "State")}
SECTIONS = {
    "implementation": ("Implementation", "IMPLEMENTATION"),
    "validation": ("Validation",),
}


class WireParserTests(unittest.TestCase):
    def test_markdown_fields_sections_preamble_and_code_fence(self):
        text = """Intro from the model.

**Status:** READY

## implementation
Keep this Markdown:

```python
STATUS: FAIL
```

- a bullet
## Validation
Tests pass.

Conclusion text.
"""
        parsed = parse_labeled_document(
            text, field_aliases=FIELDS, section_aliases=SECTIONS
        )
        self.assertEqual(parsed.fields, {"status": "READY"})
        self.assertIn("```python\nSTATUS: FAIL\n```", parsed.sections["implementation"])
        self.assertEqual(parsed.sections["validation"], "Tests pass.\n\nConclusion text.")
        self.assertEqual(parsed.preamble, "Intro from the model.\n\n**Status:** READY")
        self.assertEqual(parsed.raw, text)

    def test_aliases_case_whitespace_equals_and_bullets(self):
        text = "- status = ready\n\n[ IMPLEMENTATION ]\ncontent"
        parsed = parse_labeled_document(
            text, field_aliases=FIELDS, section_aliases=SECTIONS
        )
        self.assertEqual(parsed.fields["status"], "ready")
        self.assertEqual(parsed.sections["implementation"], "content")

    def test_section_marker_variants_are_supported(self):
        documents = (
            "IMPLEMENTATION:\nbody",
            "[IMPLEMENTATION]\nbody",
            "=== IMPLEMENTATION ===\nbody",
            "## Implementation\nbody",
        )
        for document in documents:
            with self.subTest(document=document):
                parsed = parse_labeled_document(
                    document, field_aliases=FIELDS, section_aliases=SECTIONS
                )
                self.assertEqual(parsed.sections["implementation"], "body")

    def test_whole_response_fenced_markdown_is_unwrapped(self):
        text = """```markdown
STATUS: READY

## Implementation
```python
print('snippet')
```
```"""
        parsed = parse_labeled_document(
            text, field_aliases=FIELDS, section_aliases=SECTIONS
        )
        self.assertEqual(parsed.fields["status"], "READY")
        self.assertIn("print('snippet')", parsed.sections["implementation"])

    def test_python_code_fence_is_ignored_as_metadata(self):
        text = """```python
STATUS: FAIL
## Implementation
not a section
```
STATUS: READY"""
        parsed = parse_labeled_document(
            text, field_aliases=FIELDS, section_aliases=SECTIONS
        )
        self.assertEqual(parsed.fields, {"status": "READY"})
        self.assertEqual(parsed.sections, {})

    def test_contradictory_status_is_ambiguous(self):
        text = "STATUS: READY\nSome explanation\nStatus = BLOCKED"
        with self.assertRaisesRegex(AmbiguousFieldError, "ambiguous"):
            parse_labeled_document(
                text, field_aliases=FIELDS, section_aliases=SECTIONS
            )

    def test_same_status_repeated_is_not_ambiguous(self):
        parsed = parse_labeled_document(
            "STATUS: READY\nStatus: ready",
            field_aliases=FIELDS,
            section_aliases=SECTIONS,
        )
        self.assertEqual(parsed.fields["status"], "ready")

    def test_explicit_delimited_section_has_postamble(self):
        text = "=== IMPLEMENTATION ===\nbody\n=== IMPLEMENTATION ===\nConclusion"
        parsed = parse_labeled_document(
            text, field_aliases=FIELDS, section_aliases=SECTIONS
        )
        self.assertEqual(parsed.sections["implementation"], "body")
        self.assertEqual(parsed.postamble, "Conclusion")


if __name__ == "__main__":
    unittest.main()
