import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.remote.auth import (
    DEFAULT_MAX_TOKEN_BYTES,
    TokenFileError,
    bearer_token_from_header,
    load_token_file,
    token_matches,
)

SECRET = "gateway-secret-token-value"


class LoadTokenFileTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write(self, data: bytes, name: str = "token") -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path

    def assert_refused(self, path: Path, **kwargs: int) -> TokenFileError:
        with self.assertRaises(TokenFileError) as raised:
            load_token_file(path, **kwargs)
        self.assertIsInstance(raised.exception, ValueError)
        self.assertNotIn(SECRET, str(raised.exception))
        self.assertNotIn(SECRET, repr(raised.exception))
        return raised.exception

    def test_valid_token(self) -> None:
        path = self.write(SECRET.encode("utf-8"))
        self.assertEqual(load_token_file(path), SECRET)
        self.assertEqual(load_token_file(str(path)), SECRET)

    def test_final_newline(self) -> None:
        self.assertEqual(load_token_file(self.write(f"{SECRET}\n".encode())), SECRET)

    def test_final_crlf(self) -> None:
        self.assertEqual(load_token_file(self.write(f"{SECRET}\r\n".encode())), SECRET)

    def test_expanduser(self) -> None:
        (self.root / "token").write_bytes(SECRET.encode("utf-8"))
        with mock.patch.dict(os.environ, {"HOME": str(self.root)}):
            self.assertEqual(load_token_file("~/token"), SECRET)

    def test_empty_file(self) -> None:
        for content in (b"", b"\n", b"\r\n"):
            with self.subTest(content=content):
                self.assert_refused(self.write(content))

    def test_token_too_large(self) -> None:
        oversize = b"a" * (DEFAULT_MAX_TOKEN_BYTES + 1)
        self.assert_refused(self.write(oversize))
        self.assert_refused(self.write(b"a" * 65, "small"), max_bytes=64)
        exact = self.write(b"a" * 64, "exact")
        self.assertEqual(load_token_file(exact, max_bytes=64), "a" * 64)
        full = self.write(b"b" * DEFAULT_MAX_TOKEN_BYTES, "full")
        self.assertEqual(load_token_file(full), "b" * DEFAULT_MAX_TOKEN_BYTES)

    def test_invalid_utf8(self) -> None:
        self.assert_refused(self.write(f"{SECRET}\n".encode() + b"\xff\xfe"))

    def test_embedded_newline(self) -> None:
        for content in (f"{SECRET}\nextra\n", f"{SECRET}\rlextra", f"{SECRET}\r"):
            with self.subTest(content=content):
                self.assert_refused(self.write(content.encode("utf-8")))

    def test_missing_file_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            load_token_file(self.root / "absent")

    def test_invalid_max_bytes(self) -> None:
        with self.assertRaises(ValueError):
            load_token_file(self.write(SECRET.encode("utf-8")), max_bytes=0)


class BearerHeaderTests(unittest.TestCase):
    def test_bearer_token(self) -> None:
        self.assertEqual(bearer_token_from_header("Bearer abc"), "abc")
        self.assertEqual(bearer_token_from_header("bearer abc"), "abc")
        self.assertEqual(bearer_token_from_header("BEARER abc"), "abc")
        self.assertEqual(bearer_token_from_header("Bearer  abc "), "abc")
        self.assertEqual(bearer_token_from_header(" Bearer abc"), "abc")
        self.assertEqual(bearer_token_from_header("Bearer\tabc"), "abc")

    def test_other_schemes(self) -> None:
        for value in (None, "", "Basic abc", "Token abc", "abc", "BearerToken abc"):
            with self.subTest(value=value):
                self.assertIsNone(bearer_token_from_header(value))

    def test_empty_bearer(self) -> None:
        for value in ("Bearer", "Bearer ", "Bearer\t"):
            with self.subTest(value=value):
                self.assertIsNone(bearer_token_from_header(value))


class TokenMatchesTests(unittest.TestCase):
    def test_matching_and_mismatching(self) -> None:
        self.assertTrue(token_matches(SECRET, SECRET))
        self.assertTrue(token_matches("é" * 3, "é" * 3))
        for candidate in ("wrong", "", SECRET + "x", SECRET[:-1], "é" * 2, None):
            with self.subTest(candidate=candidate):
                self.assertFalse(token_matches(candidate, SECRET))
        self.assertFalse(token_matches(SECRET, ""))


if __name__ == "__main__":
    unittest.main()
