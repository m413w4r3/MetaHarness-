package com.m413w4r3.metaharnessremote.api

import com.google.gson.Gson
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParseException
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import com.m413w4r3.metaharnessremote.network.GatewayUrls
import java.io.IOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response

/**
 * Read-only client of the MetaHarness Remote gateway (`/v1` routes).
 *
 * Every request carries `Authorization: Bearer <remoteToken>` and
 * `Accept: application/json`, and a request with a body also carries
 * `Content-Type: application/json`. Each call performs exactly one exchange:
 * nothing is retried, and the injected [OkHttpClient] owns timeouts, redirects
 * and connection reuse. Failures raise [MetaHarnessException]; the remote
 * token is never copied into one.
 */
class MetaHarnessApi(
    baseUrl: String,
    private val remoteToken: String,
    private val httpClient: OkHttpClient,
) {

    /** The https base URL, without a trailing slash. */
    private val baseUrl: String = when (val normalized = GatewayUrls.normalize(baseUrl)) {
        is GatewayUrls.BaseUrl.Valid -> normalized.value
        is GatewayUrls.BaseUrl.Invalid -> throw IllegalArgumentException(normalized.reason)
    }

    private val gson = Gson()

    /** `GET /v1/health`. */
    suspend fun health(): HealthResponse = get(HEALTH_PATH, HealthResponse::class.java)

    /** `GET /v1/config`, kept as a raw document. */
    suspend fun config(): JsonObject = get(CONFIG_PATH, JsonObject::class.java)

    /** `GET /v1/model-profiles`, kept as a raw document. */
    suspend fun modelProfiles(): JsonObject = get(MODEL_PROFILES_PATH, JsonObject::class.java)

    /** `GET /v1/runs`. */
    suspend fun listRuns(): List<RunSummary> = get(RUNS_PATH, RunListDocument::class.java).runs

    /** `GET /v1/runs/{runId}`, kept as a raw document. */
    suspend fun getRun(runId: String): RunDetail =
        RunDetail(get(runPath(runId), JsonObject::class.java))

    /** `GET /v1/runs/{runId}/progress?offset=N`. */
    suspend fun progress(runId: String, offset: Long): ProgressResponse {
        require(offset >= 0) { "offset must be a non-negative integer" }
        return get("${runPath(runId)}/progress?offset=$offset", ProgressResponse::class.java)
    }

    /** `GET <baseUrl><path>` with the bearer token and the JSON accept header. */
    internal fun getRequest(path: String): Request = requestBuilder(path).get().build()

    /** `POST <baseUrl><path>` carrying [body] as `application/json`. */
    internal fun postRequest(path: String, body: String): Request =
        requestBuilder(path)
            .post(body.toRequestBody(JSON_MEDIA_TYPE))
            .header(CONTENT_TYPE, JSON_MEDIA_TYPE.toString())
            .build()

    private fun requestBuilder(path: String): Request.Builder =
        Request.Builder()
            .url(baseUrl + path)
            .header(AUTHORIZATION, "Bearer $remoteToken")
            .header(ACCEPT, "application/json")

    private suspend fun <T> get(path: String, type: Class<T>): T = withContext(Dispatchers.IO) {
        val response = try {
            httpClient.newCall(getRequest(path)).execute()
        } catch (io: IOException) {
            throw requestFailure(io)
        }
        response.use { read(it, type) }
    }

    private fun <T> read(response: Response, type: Class<T>): T {
        val body = readBody(response)
        if (!response.isSuccessful) {
            throw MetaHarnessException(
                statusCode = response.code,
                message = failureMessage(response.code),
                payload = parsePayload(body),
            )
        }
        return parse(body, type) ?: throw MetaHarnessException(
            statusCode = null,
            message = "The gateway replied with a malformed JSON document",
        )
    }

    /**
     * Read the body with the same 2 MiB cap the gateway applies, so a
     * misbehaving peer cannot make the client buffer an unbounded response.
     */
    private fun readBody(response: Response): String {
        val source = response.body?.source() ?: return ""
        val hasMoreThanTheCap = try {
            source.request(MAX_RESPONSE_BYTES + 1)
        } catch (io: IOException) {
            throw requestFailure(io)
        }
        if (hasMoreThanTheCap) {
            throw MetaHarnessException(
                statusCode = null,
                message = "The gateway reply exceeds $MAX_RESPONSE_BYTES bytes",
            )
        }
        // The buffer now holds the whole body: it is at most the cap.
        return source.buffer.readByteArray().toString(Charsets.UTF_8)
    }

    private fun <T> parse(body: String, type: Class<T>): T? =
        try {
            if (body.isBlank()) null else gson.fromJson(body, type)
        } catch (_: JsonParseException) {
            null
        }

    /** The decoded error document, or the raw text when it is not JSON. */
    private fun parsePayload(body: String): JsonElement? {
        if (body.isBlank()) return null
        return try {
            JsonParser.parseString(body)
        } catch (_: JsonParseException) {
            JsonPrimitive(body)
        }
    }

    private fun requestFailure(io: IOException): MetaHarnessException {
        val detail = io.message?.takeIf { it.isNotBlank() } ?: io.javaClass.simpleName
        return MetaHarnessException(
            statusCode = null,
            message = "Request failed: ${redact(detail)}",
            cause = io,
        )
    }

    /** The token must never reach a message, whatever the transport reports. */
    private fun redact(value: String): String =
        if (remoteToken.isEmpty()) value else value.replace(remoteToken, "***")

    /**
     * Validate one run-id path component. The gateway refuses the same shapes;
     * refusing them here too keeps a hostile id out of the request line.
     */
    private fun runPath(runId: String): String {
        require(
            RUN_ID.matches(runId) &&
                !runId.contains("..") &&
                !runId.endsWith(".") &&
                !runId.endsWith(".lock")
        ) { "invalid run id" }
        return "$RUNS_PATH/$runId"
    }

    private fun failureMessage(status: Int): String = when (status) {
        401 -> "Unauthorized: the remote token was rejected"
        403 -> "Forbidden: the remote token is not allowed"
        else -> "Gateway replied HTTP $status"
    }

    private companion object {
        const val HEALTH_PATH = "/v1/health"
        const val CONFIG_PATH = "/v1/config"
        const val MODEL_PROFILES_PATH = "/v1/model-profiles"
        const val RUNS_PATH = "/v1/runs"

        const val AUTHORIZATION = "Authorization"
        const val ACCEPT = "Accept"
        const val CONTENT_TYPE = "Content-Type"

        val JSON_MEDIA_TYPE = "application/json".toMediaType()

        /** Matches the gateway's own run-id charset. */
        val RUN_ID = Regex("[A-Za-z0-9][A-Za-z0-9_.-]*")

        /** The cap the gateway applies to its own responses. */
        const val MAX_RESPONSE_BYTES = 2L * 1024 * 1024
    }
}
