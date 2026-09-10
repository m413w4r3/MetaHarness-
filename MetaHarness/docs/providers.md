# Providers

MetaHarness V0 uses one deliberately small, OpenAI-compatible text contract
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

V0 deliberately does not require JSON output, a `system` role, or a
`response_format` field; `response_format` is rejected if supplied through
`extra_body` as well. Structured Outputs are intentionally unused so the
planner and reviewer have the same wire contract on every backend.

## ChatGPT bridge

For a ChatGPT-compatible bridge, configure:

```toml
[planner]
base_url = "${META_PLANNER_BASE_URL}"
endpoint_path = "/v1/chat/completions"
model = "${META_PLANNER_MODEL}"

[planner.extra_body]
new_chat = true

[reviewer]
base_url = "${META_REVIEWER_BASE_URL}"
endpoint_path = "/v1/chat/completions"
model = "${META_REVIEWER_MODEL}"

[reviewer.extra_body]
new_chat = true
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
