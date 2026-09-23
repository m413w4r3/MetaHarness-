# Providers

MetaHarness uses one deliberately small, OpenAI-compatible text contract
for planner and reviewer endpoints.

## Minimal HTTP contract

The endpoint is configurable and receives one `POST` request with this shape:

```json
{
  "model": "provider-specific-label",
  "messages": [{"role": "user", "content": "..."}],
  "stream": false
}
```

The required response value is:

```text
choices[0].message.content
```

The text protocol deliberately does not require JSON output, a `system` role, or a
`response_format` field. Structured Outputs are intentionally unused so the
planner and reviewer have the same wire contract on every backend.

Transport rules:

- `extra_body` is static and validated at config load; it cannot set
  `messages`, `stream`, `stream_options`, `model`, `response_format`,
  `tools`, `tool_choice`, `functions` or `function_call`.
- `base_url` must be an absolute `http(s)` URL without credentials, query or
  fragment; `endpoint_path` must be a path without `.`/`..` segments.
- Text is read from `choices[0].message.content` as a string or a list of
  `text`/`output_text` parts. Missing, null or non-text content, and
  `finish_reason` `length`/`content_filter`, are protocol errors. `model` and
  `usage` are informational and tolerated in any shape.
- Redirects are never followed (the key is never forwarded elsewhere).
- `timeout_seconds` bounds each attempt, including a slowly trickled body;
  responses above 32 MiB are rejected.
- Only HTTP 408, 429, 500, 502, 503 and 504 are retried, at most `retries`
  times; 401/403 and network failures are not retried.

## ChatGPT bridge

Preapproval correction uses `ConversationContinuationClient` only when a driver
returns an official `LLMConversationHandle` from its first completion. The
current `openai-chat` adapter sends Chat Completions requests and receives no
conversation handle from `chatgpt-bridge`'s `/v1/chat/completions` response.
Consequently `planner-chatgpt` currently uses a fresh, self-contained
correction request. The bridge's native conversation endpoint uses an explicit
conversation ID plus an expected turn ID supplied by its caller; those values
are not exposed by the Chat Completions facade, and MetaHarness never derives
them from a URL or invents them. For same-conversation support, a bridge driver
must return a durable handle for the completed turn and implement continuation
against the native endpoint, distinguishing explicit `conversation_unavailable`
from ambiguous transport failure. The provider ID must remain stable.

For a ChatGPT-compatible bridge, configure an explicit planner/reviewer profile:

```toml
[model_profiles.chat]
display_name = "Chat bridge"
roles = ["planner", "reviewer"]
driver = "openai-chat"
provider = "openai-compatible"
model = "${META_PLANNER_MODEL}"
selection_mode = "request"
base_url = "${META_PLANNER_BASE_URL}"
endpoint_path = "/v1/chat/completions"
api_key_env = "BRIDGE_API_KEY"

[ui]
default_planner_profile = "chat"
default_reviewer_profile = "chat"
default_implementer_profile = "worker"
```

The `model` value may be only a label interpreted by the bridge. MetaHarness
does not verify or guarantee that it names a native model, and no model name
is invented here.

## WebAI-to-API Gemini

Use the same Chat Completions wire contract with either:

```toml
endpoint_path = "/v1/chat/completions"
```

or, when the bridge exposes it and a stateless interaction is preferred:

```toml
endpoint_path = "/v1/temporary/chat/completions"
```

Do not hardcode a Gemini model name. Set the model field to the label expected
by the selected bridge.

## OpenAI

OpenAI uses the same Chat Completions contract. Configure credentials by
variable name only:

```toml
api_key_env = "OPENAI_API_KEY"
```

`api_key_env` is never the key value. The key is read only to build the
authorization header for the request and is not written to run state,
prompts, logs, or errors. The bootstrap performs no network call.
