# MetaHarness Remote (Android)

Standalone Android client for the MetaHarness remote gateway. It does not depend
on Nimbalyst: the phone reaches the gateway over an HTTPS Tailscale Serve URL.

## Build

```bash
cd android-remote
./gradlew test
./gradlew assembleDebug
```

The debug APK is written to `app/build/outputs/apk/debug/app-debug.apk`.

Toolchain: Gradle 8.10.2, Android Gradle Plugin 8.7.3, Kotlin 2.0.21, JDK 17,
`compileSdk` 35, `targetSdk` 35, `minSdk` 29, Jetpack Compose + Material 3.
`local.properties` (or `ANDROID_HOME`) must point at an Android SDK with
platform 35 and build-tools 35.

## Settings screen

`Server URL` and `Remote token` feed the `Test Connection` button, which calls
`GET <baseUrl>/v1/health` with `Authorization: Bearer <token>`.

* `Server URL` is the only persisted setting (private `SharedPreferences`).
* `Remote token` is kept in memory only — it is mirrored into
  `ConnectionSession`, where the Runs screen reads it, and is never written to
  disk.
* Cleartext HTTP is refused twice: `android:usesCleartextTraffic="false"` in the
  manifest, and the URL validator rejects any base URL that is not `https`.
* OkHttp is built without a logging interceptor, so the `Authorization` header
  is never written to logcat, and `retryOnConnectionFailure` plus redirect
  following are disabled so a request is never replayed.

## Runs screen

The app starts on the Runs board. Entering the screen reads `GET /v1/health`
and `GET /v1/runs`, then repeats both every 3 seconds while the screen is
resumed; pulling the list down or tapping `Refresh` reloads at once, and a pass
already in flight absorbs the next trigger. The `Settings` action in the header
opens the screen above.

Runs are grouped by `state.status`:

| Section | Statuses |
| --- | --- |
| `ACTION REQUIRED` | `blocked`, `awaiting_plan_approval`, `waiting_scope_approval`, `plan_rejected` |
| `ACTIVE` | `created`, `planning`, `worktree_ready`, `preparing`, `implementing`, `contract_repairing`, `validating`, `pre_revision_validating`, `revising`, `revalidating`, `reviewing`, `approved`, `publishing` |
| `FAILED` | `failed`, `interrupted` |
| `COMPLETED` | `published`, `committed` |
| `OTHER` | every other status |

Each card shows the run id, the plan title, the status, `updated_at`, the first
7 characters of the commit SHA and the failure summary (`reason — detail`, the
way the desktop list renders it). A card in `ACTION REQUIRED` carries the badge,
and tapping any card hands its run id to a callback. `+ NEW RUN` opens the New
Run screen; the Run Detail view arrives with a later prompt, so the route it
will use only shows the run id it was opened with.

The board only ever reads the gateway: nothing is cached on the device, the
desktop stays the source of authority, and no local database is involved.

## New Run screen

`+ NEW RUN` opens a form with one `SPEC`, one optional `Run ID` and one
`CREATE RUN` button. A tap validates both fields, then sends exactly one
`POST /v1/runs`:

* the SPEC must hold text, and at most 48 KiB of UTF-8 bytes;
* the run id, when the field is not empty, must match
  `^[A-Za-z0-9][A-Za-z0-9_.-]*$`. An empty field is left out of the payload,
  so the gateway generates the id.

`submitting = true` is held for the duration of the call and disables the
button, so a double tap cannot create two runs. A create that succeeds opens
the Run Detail route with the run id the gateway returned, and the form is
popped: going back from the run returns to the board, never to a filled-in
form that could create the run twice.

A create that times out is *not* a failure: the request reached the gateway,
so the run may exist. Nothing is retried automatically, and the screen says
exactly that:

```text
Response timed out.
The run may have been created.
Refresh the runs list before trying again.
```

## API client

`com.m413w4r3.metaharnessremote.api.MetaHarnessApi` is the client of the
gateway, built from a `baseUrl`, the in-memory `remoteToken` and an injected
`OkHttpClient`:

| Method | Route |
| --- | --- |
| `health()` | `GET /v1/health` |
| `config()` | `GET /v1/config` |
| `modelProfiles()` | `GET /v1/model-profiles` |
| `listRuns()` | `GET /v1/runs` |
| `getRun(runId)` | `GET /v1/runs/<run_id>` |
| `progress(runId, offset)` | `GET /v1/runs/<run_id>/progress?offset=N` |
| `createRun(spec, runId)` | `POST /v1/runs` |

Every call is one exchange with no retry, carries the bearer token and
`Accept: application/json`, reads at most 2 MiB, and raises
`MetaHarnessException(statusCode, message, payload)` on failure without ever
copying the token. A call that got no answer in time raises
`MetaHarnessTimeoutException` instead: the request was sent, so a mutation may
still have been applied. `config()` and `modelProfiles()` stay raw
`JsonObject`s and `getRun()` returns the raw document, so only the displayed
fields are typed.
