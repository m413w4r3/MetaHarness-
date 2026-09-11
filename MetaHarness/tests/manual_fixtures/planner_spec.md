Add bounded retry support to the existing HTTP client.

Requirements:
- Retry only HTTP 429, 502, 503 and 504.
- Maximum 3 total attempts.
- Use exponential backoff of 100 ms then 200 ms.
- Never retry HTTP 400 or 401.
- Preserve the current public function signature.
- Add unit tests for retryable success, exhausted retries, and non-retryable errors.
