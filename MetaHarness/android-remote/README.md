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
and tapping any card opens the Run Detail screen of that run. `+ NEW RUN` opens
the New Run screen.

The board only ever reads the gateway: nothing is cached on the device, the
desktop stays the source of authority, and no local database is involved.

## Run Detail screen

Tapping a card (or creating a run) opens it read-only. Each pass reads
`GET /v1/runs/<run_id>` and then `GET /v1/runs/<run_id>/progress?offset=N`:

| Field | Read from |
| --- | --- |
| run id | `run_id`, else the id the screen was opened with |
| status | `status` |
| failure | `failure`, rendered `reason — detail` as on the board |
| candidate SHA | the current cycle's entry in `candidate`, else `state.candidate_commit_sha`, else `commit_sha` |
| cycle | `cycle` |
| overview | `overview`: `current_label`, `next_label`, `execution_label`, `publish_target`, `resume` |
| plan | `plan.raw`, bounded to 8000 characters, falling back to `plan.contract` |
| implementation steps | `implementation_bundle.steps`, each showing `id`, `title`, `execution_class` and the profile: `profile_id` from `state.steps`, else `recommended_profile` from `state.planner.steps` |

Every field is optional and type-checked, so a run whose document is still
incomplete renders what it has instead of failing. The only route of this
screen that can change a run is the plan decision below, and it is sent only
when the operator asks for it.

Progress is incremental: the offset starts at `0` and every answer replaces it
with its own `next_offset`, so each pass appends only the events that became
visible since the previous one. A reply that does not move the offset forward is
ignored whole, so an event is never listed twice, and the screen keeps the last
1000 events in memory while the gateway keeps the history.

Polling cadence follows the run status, through the same classification as the
board:

| Status | Poll |
| --- | --- |
| `ACTIVE` | every 2 seconds |
| `ACTION REQUIRED` | every 5 seconds |
| `COMPLETED`, `FAILED` | stops |
| any other status | every 5 seconds |

A failed pass keeps the last good detail, reports the error, and retries every
5 seconds. All polling stops when the screen leaves the foreground — the loop
lives in `repeatOnLifecycle(RESUMED)` — so a screen that is not visible costs
nothing.

## Plan approval

While a run waits for a decision on its plan, the Run Detail screen renders a
`PLAN APPROVAL` block between the status card and the plan. It is shown only
when the run document holds all four conditions, each one read exactly as the
desktop form reads it:

```text
status == awaiting_plan_approval
approval.recorded != true
approval.awaiting != false
capabilities.plan_approval != false
```

`capabilities` is the object of the run document, else the one of its
`overview`, else the `capabilities` of `GET /v1/config` — a run document
carries none, so the configuration is read once per screen entry. A capability
that cannot be read does not hide the block: only an explicit `false` does.

| Field | Read from |
| --- | --- |
| steps | `implementation_bundle.steps` (`id`, `title`, `execution_class`; a step without a class is mechanical) |
| recorded implementer | `execution_selection.steps[].step_id` + `.implementer.profile_id` |
| requested profiles | `run_options.profiles` (`mechanical_profile`, `reasoning_profile`, `agentic_profile`, `final_reviewer_profile`, `semantic_reviser_profile`, `check_repair_profile`) |
| selectable profiles | `GET /v1/model-profiles`, role `implementer` for a step, `reviewer` / `reviser` / `repair` for the role fields, filtered by `execution_classes` (or the historic `classes`) when the profile declares any |

Each step starts from what the run recorded for it, then from the profile its
run options routed to that execution class, then from the first compatible
profile. The three role fields start from the same order, ending with the
gateway's own `defaults`. The semantic reviser and the check repair fields are
offered only when the run's own options use them (`semantic_revision_enabled`,
or a correction budget): MetaHarness refuses a profile of a role the run does
not enable.

`APPROVE & CONTINUE` sends exactly one `POST /v1/runs/<run_id>/approval`, and is
enabled only once every step, the final reviewer, and every role the run uses
hold a compatible profile:

```json
{
  "decision": "APPROVE",
  "final_reviewer_profile": "reviewer-heavy",
  "semantic_reviser_profile": "reviser-1",
  "check_repair_profile": "repair-1",
  "step_profiles": {
    "S01": "impl-fast",
    "S02": "impl-heavy"
  }
}
```

That body is the external contract: the step keys are plan step ids, and
translating them into the local field names is the gateway's job. The app never
builds a `step_profile__S01` field, and a role the run does not use is left out
of the body. `REJECT PLAN` opens an `AlertDialog` that states the rejection is
irreversible; confirming it sends `{"decision": "REJECT"}` and nothing else.

One mutation runs at a time: while a decision is in flight both buttons are
disabled and a second tap starts nothing. Whatever the answer was, the screen
then reads `GET /v1/runs/<run_id>` again — an accepted decision moves the run,
and an HTTP 409 means another decision is already recorded, so the block
disappears on the refreshed state. Nothing is ever retried; a decision that
timed out says so and lets the refreshed run, not a second decision, tell the
operator what happened:

```text
Response timed out.
The decision may have been recorded.
The run is read again to show what it recorded.
```

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
| `approveRun(runId, payload)` | `POST /v1/runs/<run_id>/approval` |

Every call is one exchange with no retry, carries the bearer token and
`Accept: application/json`, reads at most 2 MiB, and raises
`MetaHarnessException(statusCode, message, payload)` on failure without ever
copying the token. A call that got no answer in time raises
`MetaHarnessTimeoutException` instead: the request was sent, so a mutation may
still have been applied. `config()` and `modelProfiles()` stay raw
`JsonObject`s and `getRun()` returns the raw document, so only the displayed
fields are typed. `approveRun(runId, payload)` takes the external approval body
described above and answers the decision the gateway recorded.
