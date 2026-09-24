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
