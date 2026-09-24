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
* `Remote token` is kept in memory only and is never written to disk.
* Cleartext HTTP is refused twice: `android:usesCleartextTraffic="false"` in the
  manifest, and the URL validator rejects any base URL that is not `https`.
* OkHttp is built without a logging interceptor, so the `Authorization` header
  is never written to logcat, and `retryOnConnectionFailure` plus redirect
  following are disabled so a request is never replayed.

## API client

`com.m413w4r3.metaharnessremote.api.MetaHarnessApi` is the read layer over the
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

Every call is one exchange with no retry, carries the bearer token and
`Accept: application/json`, reads at most 2 MiB, and raises
`MetaHarnessException(statusCode, message, payload)` on failure without ever
copying the token. `config()` and `modelProfiles()` stay raw `JsonObject`s and
`getRun()` returns the raw document, so only the displayed fields are typed.
