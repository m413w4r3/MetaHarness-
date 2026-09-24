package com.m413w4r3.metaharnessremote.api

import com.google.gson.Gson
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParseException
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import com.m413w4r3.metaharnessremote.network.GatewayUrls
import java.io.IOException
import java.io.InterruptedIOException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response

/**
 * Client of the MetaHarness Remote gateway (`/v1` routes).
 *
 * Every request carries `Authorization: Bearer <remoteToken>` and
 * `Accept: application/json`, and a request with a body also carries
 * `Content-Type: application/json`. Each call performs exactly one exchange:
 * nothing is retried, and the injected [OkHttpClient] owns timeouts, redirects
 * and connection reuse. Failures raise [MetaHarnessException], a call that got
 * no answer in time raises [MetaHarnessTimeoutException]; the remote token is
 * never copied into either.
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

    /**
     * `POST /v1/runs` with [spec] and, when it is not blank, [runId].
     *
     * The run id is optional: an empty one is left out of the payload so the
     * gateway generates it. Exactly one exchange is sent — nothing is retried,
     * because a create that timed out may still have created the run, which is
     * what [MetaHarnessTimeoutException] reports.
     */
    suspend fun createRun(spec: String, runId: String?): CreateRunResponse {
        require(spec.trim().isNotEmpty()) { "spec must not be empty" }
        require(spec.toByteArray(Charsets.UTF_8).size <= MAX_SPEC_BYTES) { "spec is too large" }
        val selectedRunId = runId?.trim()?.takeIf { it.isNotEmpty() }
        require(selectedRunId == null || RUN_ID_PATTERN.matches(selectedRunId)) { "invalid run id" }

        val document = post(RUNS_PATH, createRunBody(spec, selectedRunId), CreateRunDocument::class.java)
        val createdRunId = document.runId?.trim().orEmpty()
        if (!RUN_ID_PATTERN.matches(createdRunId)) {
            throw MetaHarnessException(
                statusCode = null,
                message = "The gateway replied without a run id",
            )
        }
        return CreateRunResponse(createdRunId)
    }

    /** The body of `POST /v1/runs`: the spec, and the run id unless it is null. */
    internal fun createRunBody(spec: String, runId: String?): String {
        val payload = JsonObject()
        payload.addProperty("spec", spec)
        if (runId != null) payload.addProperty("run_id", runId)
        return gson.toJson(payload)
    }

    /**
     * `POST /v1/runs/{runId}/approval` with the external approval [payload].
     *
     * The body speaks the external contract: a `decision`, the role profiles,
     * and `step_profiles` keyed by plan step id (`{"S01": "impl-fast"}`).
     * Translating those keys into the local field names belongs to the gateway,
     * so this client never builds a `step_profile__S01` field.
     *
     * Exactly one exchange is sent — nothing is retried, because a decision
     * that timed out may still have been recorded.
     */
    suspend fun approveRun(runId: String, payload: JsonObject): ApprovalResponse {
        val decision = payload.decision() ?: throw IllegalArgumentException(
            "decision must be $APPROVE or $REJECT",
        )
        val document = post(
            "${runPath(runId)}$APPROVAL_PATH",
            gson.toJson(payload),
            ApprovalDocument::class.java,
        )
        return ApprovalResponse(
            decision = document.decision?.trim()?.takeIf { it.isNotEmpty() } ?: decision,
        )
    }

    /**
     * `POST /v1/runs/{runId}/scope-approval` with the [decision] (`APPROVE` or
     * `REJECT`) for the repair scope the run requested.
     *
     * The body carries the decision alone: the delta is the one the run
     * recorded, so no path is ever sent from the phone and no scope can be
     * edited here. Exactly one exchange is sent — nothing is retried, because a
     * decision that timed out may still have been recorded.
     */
    suspend fun approveScope(runId: String, decision: String) {
        require(decision == APPROVE || decision == REJECT) { "decision must be $APPROVE or $REJECT" }
        val payload = JsonObject()
        payload.addProperty(DECISION, decision)
        post("${runPath(runId)}$SCOPE_APPROVAL_PATH", gson.toJson(payload), JsonObject::class.java)
    }

    /**
     * `POST /v1/runs/{runId}/resume` with an empty JSON object.
     *
     * The body is exactly `{}`: the gateway resumes the run from its own
     * checkpoint. Exactly one exchange is sent — nothing is retried, because a
     * resume that timed out may still have been accepted.
     */
    suspend fun resumeRun(runId: String) {
        post("${runPath(runId)}$RESUME_PATH", EMPTY_BODY, JsonObject::class.java)
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

    private suspend fun <T> call(request: Request, type: Class<T>): T = withContext(Dispatchers.IO) {
        val response = try {
            httpClient.newCall(request).execute()
        } catch (io: IOException) {
            throw requestFailure(io)
        }
        response.use { read(it, type) }
    }

    private suspend fun <T> get(path: String, type: Class<T>): T = call(getRequest(path), type)

    private suspend fun <T> post(path: String, body: String, type: Class<T>): T =
        call(postRequest(path, body), type)

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
        // A timeout is not a failure: the request was sent, so the gateway may
        // have applied it. The caller is told the outcome is unknown.
        if (io is InterruptedIOException) {
            return MetaHarnessTimeoutException("Response timed out", cause = io)
        }
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
            RUN_ID_PATTERN.matches(runId) &&
                !runId.contains("..") &&
                !runId.endsWith(".") &&
                !runId.endsWith(".lock")
        ) { "invalid run id" }
        return "$RUNS_PATH/$runId"
    }

    /** The decision of an approval body, or null when it is absent or unknown. */
    private fun JsonObject.decision(): String? =
        get(DECISION)
            ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.takeIf { it == APPROVE || it == REJECT }

    private fun failureMessage(status: Int): String = when (status) {
        401 -> "Unauthorized: the remote token was rejected"
        403 -> "Forbidden: the remote token is not allowed"
        else -> "Gateway replied HTTP $status"
    }

    companion object {
        /** Largest SPEC `POST /v1/runs` accepts, as the local server measures it. */
        const val MAX_SPEC_BYTES = 48 * 1024

        /** The run-id charset the gateway and the local server enforce. */
        val RUN_ID_PATTERN = Regex("[A-Za-z0-9][A-Za-z0-9_.-]*")

        private const val HEALTH_PATH = "/v1/health"
        private const val CONFIG_PATH = "/v1/config"
        private const val MODEL_PROFILES_PATH = "/v1/model-profiles"
        private const val RUNS_PATH = "/v1/runs"
        private const val APPROVAL_PATH = "/approval"
        private const val SCOPE_APPROVAL_PATH = "/scope-approval"
        private const val RESUME_PATH = "/resume"

        /** The body of a resume: the empty document, as the gateway requires. */
        private const val EMPTY_BODY = "{}"

        private const val APPROVE = "APPROVE"
        private const val REJECT = "REJECT"
        private const val DECISION = "decision"

        private const val AUTHORIZATION = "Authorization"
        private const val ACCEPT = "Accept"
        private const val CONTENT_TYPE = "Content-Type"

        private val JSON_MEDIA_TYPE = "application/json".toMediaType()

        /** The cap the gateway applies to its own responses. */
        private const val MAX_RESPONSE_BYTES = 2L * 1024 * 1024
    }
}
