package com.m413w4r3.metaharnessremote.network

import com.google.gson.Gson
import com.google.gson.JsonParseException
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request

/**
 * Read-only gateway client.
 *
 * The OkHttp client is built once, without any logging interceptor, so the
 * `Authorization` header never reaches logcat, and connection retries and
 * redirects are disabled: the token is only ever sent to the configured host,
 * and a request is never replayed.
 */
class GatewayClient(
    private val httpClient: OkHttpClient = defaultHttpClient(),
    private val gson: Gson = Gson(),
) {

    suspend fun health(baseUrl: String, token: String): HealthResult =
        withContext(Dispatchers.IO) {
            try {
                httpClient.newCall(healthRequest(baseUrl, token)).execute().use { response ->
                    if (!response.isSuccessful) {
                        return@withContext HealthResult.Failed(failureMessage(response.code))
                    }
                    val document = parseHealthDocument(response.body?.string())
                    HealthResult.Connected(
                        status = document?.status ?: "ok",
                        apiVersion = document?.api_version,
                    )
                }
            } catch (io: IOException) {
                HealthResult.Failed("Connection failed: ${io.message ?: io.javaClass.simpleName}")
            }
        }

    private fun parseHealthDocument(body: String?): HealthDocument? {
        if (body.isNullOrBlank()) return null
        return try {
            gson.fromJson(body, HealthDocument::class.java)
        } catch (_: JsonParseException) {
            null
        }
    }

    private fun failureMessage(code: Int): String = when (code) {
        401 -> "Unauthorized: the remote token was rejected"
        403 -> "Forbidden: the remote token is not allowed"
        else -> "Gateway replied HTTP $code"
    }

    companion object {
        private const val CONNECT_TIMEOUT_SECONDS = 10L
        private const val READ_TIMEOUT_SECONDS = 20L
        private const val CALL_TIMEOUT_SECONDS = 30L

        /** `GET <baseUrl>/v1/health` with the bearer token; the token is not part of the URL. */
        fun healthRequest(baseUrl: String, token: String): Request =
            Request.Builder()
                .url(GatewayUrls.healthUrl(baseUrl))
                .get()
                .header("Authorization", "Bearer $token")
                .header("Accept", "application/json")
                .build()

        fun defaultHttpClient(): OkHttpClient =
            OkHttpClient.Builder()
                .connectTimeout(CONNECT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                .readTimeout(READ_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                .callTimeout(CALL_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                .retryOnConnectionFailure(false)
                .followRedirects(false)
                .followSslRedirects(false)
                .build()
    }
}
