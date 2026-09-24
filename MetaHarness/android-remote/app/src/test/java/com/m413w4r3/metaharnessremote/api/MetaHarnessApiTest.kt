package com.m413w4r3.metaharnessremote.api

import com.google.gson.JsonParser
import java.io.IOException
import java.net.SocketTimeoutException
import kotlinx.coroutines.runBlocking
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import okio.Buffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** Answers every call locally, so no test touches the network. */
private class RecordingInterceptor(
    private val respond: (Request) -> Response,
) : Interceptor {

    val requests = mutableListOf<Request>()

    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        requests += request
        return respond(request)
    }
}

private val JSON = "application/json".toMediaType()

private fun jsonResponse(request: Request, code: Int, body: String): Response =
    Response.Builder()
        .request(request)
        .protocol(Protocol.HTTP_1_1)
        .code(code)
        .message("test")
        .body(body.toResponseBody(JSON))
        .build()

/** The body a recorded request would send. */
private fun bodyOf(request: Request): String {
    val buffer = Buffer()
    request.body?.writeTo(buffer)
    return buffer.readUtf8()
}

class MetaHarnessApiTest {

    private val baseUrl = "https://gateway.test"
    private val token = "remote-token-value"

    private fun apiWith(
        respond: (Request) -> Response,
    ): Pair<MetaHarnessApi, RecordingInterceptor> {
        val interceptor = RecordingInterceptor(respond)
        val client = OkHttpClient.Builder().addInterceptor(interceptor).build()
        return MetaHarnessApi(baseUrl, token, client) to interceptor
    }

    private fun apiReturning(
        body: String,
        code: Int = 200,
    ): Pair<MetaHarnessApi, RecordingInterceptor> =
        apiWith { request -> jsonResponse(request, code, body) }

    @Test
    fun `health sends the credentials and parses the document`() = runBlocking {
        val (api, interceptor) = apiReturning(
            """{"service":"metaharness","api_version":1,"status":"ok","future":true}"""
        )

        val health = api.health()

        assertEquals(HealthResponse(service = "metaharness", status = "ok", apiVersion = 1), health)
        val request = interceptor.requests.single()
        assertEquals("GET", request.method)
        assertEquals("$baseUrl/v1/health", request.url.toString())
        assertEquals("Bearer $token", request.header("Authorization"))
        assertEquals("application/json", request.header("Accept"))
    }

    @Test
    fun `every read carries the token and accept header and is issued once`() = runBlocking {
        val (api, interceptor) = apiReturning("{}")

        api.health()
        api.config()
        api.modelProfiles()
        api.listRuns()
        api.getRun("run-1")
        api.progress("run-1", 42)

        assertEquals(
            listOf(
                "/v1/health",
                "/v1/config",
                "/v1/model-profiles",
                "/v1/runs",
                "/v1/runs/run-1",
                "/v1/runs/run-1/progress",
            ),
            interceptor.requests.map { it.url.encodedPath },
        )
        interceptor.requests.forEach { request ->
            assertEquals("GET", request.method)
            assertEquals("Bearer $token", request.header("Authorization"))
            assertEquals("application/json", request.header("Accept"))
            assertTrue("the token never enters the url", !request.url.toString().contains(token))
            assertNull("a bodyless request carries no content type", request.header("Content-Type"))
        }
    }

    @Test
    fun `config and model profiles stay raw documents`() = runBlocking {
        val (api, _) = apiWith { request ->
            when (request.url.encodedPath) {
                "/v1/config" -> jsonResponse(request, 200, """{"repository":{"repo":"/srv/repo"}}""")
                else -> jsonResponse(request, 200, """{"profiles":[{"id":"p1"}],"defaults":{}}""")
            }
        }

        val config = api.config()
        val profiles = api.modelProfiles()

        assertEquals("/srv/repo", config.getAsJsonObject("repository").get("repo").asString)
        assertEquals("p1", profiles.getAsJsonArray("profiles")[0].asJsonObject.get("id").asString)
    }

    @Test
    fun `list runs maps the run summaries`() = runBlocking {
        val (api, _) = apiReturning(
            """
            {"runs":[
              {"run_id":"run-1","status":"DONE","updated_at":"2026-09-24T20:00:00Z",
               "plan_title":"Add comment","commit_sha":"abc123","candidate":{},"failure":null},
              {"run_id":"run-2","status":"FAILED","failure":{"reason":"boom"}}
            ]}
            """.trimIndent()
        )

        val runs = api.listRuns()

        assertEquals(2, runs.size)
        assertEquals("run-1", runs[0].runId)
        assertEquals("DONE", runs[0].status)
        assertEquals("2026-09-24T20:00:00Z", runs[0].updatedAt)
        assertEquals("Add comment", runs[0].planTitle)
        assertEquals("abc123", runs[0].commitSha)
        assertTrue(runs[0].failure?.isJsonNull == true)
        assertEquals("run-2", runs[1].runId)
        assertEquals("boom", runs[1].failure?.asJsonObject?.get("reason")?.asString)
        assertNull(runs[1].planTitle)
    }

    @Test
    fun `get run keeps the raw document`() = runBlocking {
        val (api, interceptor) = apiReturning("""{"run_id":"run-1","plan":{"steps":[1,2]}}""")

        val detail = api.getRun("run-1")

        assertEquals(2, detail.raw.getAsJsonObject("plan").getAsJsonArray("steps").size())
        assertEquals("/v1/runs/run-1", interceptor.requests.single().url.encodedPath)
    }

    @Test
    fun `progress asks for the offset and parses the events`() = runBlocking {
        val (api, interceptor) = apiReturning("""{"next_offset":128,"events":["started","planned"]}""")

        val progress = api.progress("run-1", 42)

        assertEquals(ProgressResponse(nextOffset = 128, events = listOf("started", "planned")), progress)
        val url = interceptor.requests.single().url
        assertEquals("/v1/runs/run-1/progress", url.encodedPath)
        assertEquals("42", url.queryParameter("offset"))
    }

    @Test
    fun `a rejected token raises an exception carrying the status and payload`() {
        val (api, _) = apiReturning(
            """{"error":"unauthorized","message":"authentication required"}""",
            code = 401,
        )

        val failure = assertThrows(MetaHarnessException::class.java) {
            runBlocking { api.health() }
        }

        assertEquals(401, failure.statusCode)
        assertEquals(
            "authentication required",
            failure.payload?.asJsonObject?.get("message")?.asString,
        )
        assertTrue(!failure.message!!.contains(token))
    }

    @Test
    fun `a malformed document raises an exception without a status code`() {
        val (api, _) = apiReturning("<html>not json</html>")

        val failure = assertThrows(MetaHarnessException::class.java) {
            runBlocking { api.listRuns() }
        }

        assertNull(failure.statusCode)
        assertNull(failure.payload)
    }

    @Test
    fun `a transport failure is reported once and never copies the token`() {
        val (api, interceptor) = apiWith { throw IOException("connect failed for $token") }

        val failure = assertThrows(MetaHarnessException::class.java) {
            runBlocking { api.health() }
        }

        assertNull(failure.statusCode)
        assertTrue(!failure.message!!.contains(token))
        assertTrue(failure.message!!.contains("***"))
        assertEquals(1, interceptor.requests.size)
    }

    @Test
    fun `an oversized reply is refused instead of buffered`() {
        val (api, _) = apiReturning("\"${"a".repeat(2 * 1024 * 1024)}\"")

        val failure = assertThrows(MetaHarnessException::class.java) {
            runBlocking { api.config() }
        }

        assertNull(failure.statusCode)
        assertTrue(failure.message!!.contains("bytes"))
    }

    @Test
    fun `a hostile run id or offset is refused before any request`() {
        val (api, interceptor) = apiReturning("{}")

        listOf("../run", "run/../x", ".", "", "run?x", "run%2Fx", "run.", "run.lock").forEach { id ->
            assertThrows(IllegalArgumentException::class.java) {
                runBlocking { api.getRun(id) }
            }
        }
        assertThrows(IllegalArgumentException::class.java) {
            runBlocking { api.progress("run-1", -1) }
        }
        assertTrue(interceptor.requests.isEmpty())
    }

    @Test
    fun `a post request carries a json content type`() {
        val (api, _) = apiReturning("{}")

        val request = api.postRequest("/v1/runs", """{"spec":"x"}""")

        assertEquals("POST", request.method)
        assertEquals("application/json", request.header("Content-Type"))
        assertEquals("Bearer $token", request.header("Authorization"))
        assertEquals("application/json", request.header("Accept"))
        assertEquals("$baseUrl/v1/runs", request.url.toString())
    }

    @Test
    fun `create run posts the spec and the run id once`() = runBlocking {
        val (api, interceptor) = apiReturning(
            """{"ok":true,"run_id":"run-1","location":"/runs/run-1","accepted":true}"""
        )

        val created = api.createRun("Implement the widget.", "  run-1  ")

        assertEquals(CreateRunResponse(runId = "run-1"), created)
        val request = interceptor.requests.single()
        assertEquals("POST", request.method)
        assertEquals("$baseUrl/v1/runs", request.url.toString())
        assertEquals("Bearer $token", request.header("Authorization"))
        assertEquals("application/json", request.header("Accept"))
        assertEquals("application/json", request.header("Content-Type"))
        val payload = JsonParser.parseString(bodyOf(request)).asJsonObject
        assertEquals("Implement the widget.", payload.get("spec").asString)
        assertEquals("run-1", payload.get("run_id").asString)
    }

    @Test
    fun `create run leaves a blank run id out of the payload`() = runBlocking {
        val (api, interceptor) = apiReturning("""{"run_id":"20260924T201500Z-ab12cd34ef"}""")

        api.createRun("Do it.", null)
        api.createRun("Do it.", "   ")

        assertEquals(2, interceptor.requests.size)
        interceptor.requests.forEach { request ->
            val payload = JsonParser.parseString(bodyOf(request)).asJsonObject
            assertTrue("a blank run id is omitted", !payload.has("run_id"))
            assertEquals("Do it.", payload.get("spec").asString)
        }
    }

    @Test
    fun `create run keeps the spec as typed and checks its size in utf8 bytes`() = runBlocking {
        val (api, interceptor) = apiReturning("""{"run_id":"run-1"}""")
        val spec = "  first line\nsecond line  "

        api.createRun(spec, null)
        api.createRun("a".repeat(48 * 1024 / 2) + "é".repeat(48 * 1024 / 4), null)

        assertEquals(spec, JsonParser.parseString(bodyOf(interceptor.requests[0])).asJsonObject.get("spec").asString)
        assertEquals(2, interceptor.requests.size)
    }

    @Test
    fun `create run refuses a blank or oversized spec and a hostile run id before any request`() {
        val (api, interceptor) = apiReturning("""{"run_id":"run-1"}""")

        listOf("", "   ", "\n\t", "a".repeat(48 * 1024 + 1)).forEach { spec ->
            assertThrows(IllegalArgumentException::class.java) {
                runBlocking { api.createRun(spec, null) }
            }
        }
        listOf("-run", ".run", "run/1", "run id", "run?x", "run.é", "..").forEach { runId ->
            assertThrows(IllegalArgumentException::class.java) {
                runBlocking { api.createRun("Do it.", runId) }
            }
        }
        assertTrue(interceptor.requests.isEmpty())
    }

    @Test
    fun `a timed out create is reported once and never retried`() {
        val (api, interceptor) = apiWith { throw SocketTimeoutException("timeout") }

        val failure = assertThrows(MetaHarnessTimeoutException::class.java) {
            runBlocking { api.createRun("Do it.", "run-1") }
        }

        assertNull(failure.statusCode)
        assertEquals(1, interceptor.requests.size)
        assertEquals("POST", interceptor.requests.single().method)
    }

    @Test
    fun `an answer without a usable run id is refused`() {
        val (api, _) = apiReturning("""{"ok":true}""")

        val failure = assertThrows(MetaHarnessException::class.java) {
            runBlocking { api.createRun("Do it.", null) }
        }

        assertTrue(failure !is MetaHarnessTimeoutException)
        assertTrue(failure.message!!.contains("run id"))
    }

    @Test
    fun `approve run posts the external payload once and reads the decision back`() = runBlocking {
        val (api, interceptor) = apiReturning("""{"ok":true,"decision":"APPROVE"}""")
        val payload = JsonParser.parseString(
            """{"decision":"APPROVE","final_reviewer_profile":"reviewer-heavy","step_profiles":{"S01":"impl-fast"}}""",
        ).asJsonObject

        val response = api.approveRun("run-1", payload)

        assertEquals(ApprovalResponse(decision = "APPROVE"), response)
        val request = interceptor.requests.single()
        assertEquals("POST", request.method)
        assertEquals("$baseUrl/v1/runs/run-1/approval", request.url.toString())
        assertEquals("Bearer $token", request.header("Authorization"))
        assertEquals("application/json", request.header("Accept"))
        assertEquals("application/json", request.header("Content-Type"))
        val body = JsonParser.parseString(bodyOf(request)).asJsonObject
        assertEquals("APPROVE", body.get("decision").asString)
        assertEquals("reviewer-heavy", body.get("final_reviewer_profile").asString)
        assertEquals("impl-fast", body.getAsJsonObject("step_profiles").get("S01").asString)
        assertTrue(
            "the client never builds a local field name",
            !body.toString().contains("step_profile__"),
        )
    }

    @Test
    fun `approve run sends a rejection as the decision alone`() = runBlocking {
        val (api, interceptor) = apiReturning("""{"ok":true,"decision":"REJECT"}""")
        val payload = JsonParser.parseString("""{"decision":"REJECT"}""").asJsonObject

        val response = api.approveRun("run-1", payload)

        assertEquals(ApprovalResponse(decision = "REJECT"), response)
        assertEquals("""{"decision":"REJECT"}""", bodyOf(interceptor.requests.single()))
    }

    @Test
    fun `approve run keeps the sent decision when the answer names none`() = runBlocking {
        val (api, _) = apiReturning("""{"ok":true}""")
        val payload = JsonParser.parseString("""{"decision":"APPROVE"}""").asJsonObject

        assertEquals(ApprovalResponse(decision = "APPROVE"), api.approveRun("run-1", payload))
    }

    @Test
    fun `approve run refuses a body without a decision and a hostile run id`() {
        val (api, interceptor) = apiReturning("""{"ok":true,"decision":"APPROVE"}""")

        listOf("""{"decision":"MAYBE"}""", """{"decision":1}""", """{"final_reviewer_profile":"p"}""")
            .forEach { body ->
                assertThrows(IllegalArgumentException::class.java) {
                    runBlocking { api.approveRun("run-1", JsonParser.parseString(body).asJsonObject) }
                }
            }
        listOf("../run", "run/1", ".", "", "run.lock").forEach { runId ->
            assertThrows(IllegalArgumentException::class.java) {
                runBlocking {
                    api.approveRun(
                        runId,
                        JsonParser.parseString("""{"decision":"APPROVE"}""").asJsonObject,
                    )
                }
            }
        }
        assertTrue(interceptor.requests.isEmpty())
    }

    @Test
    fun `a refused decision is reported once and never retried`() {
        val (api, interceptor) = apiReturning(
            """{"error":"conflict","message":"run is not awaiting plan approval"}""",
            code = 409,
        )

        val failure = assertThrows(MetaHarnessException::class.java) {
            runBlocking {
                api.approveRun(
                    "run-1",
                    JsonParser.parseString("""{"decision":"APPROVE"}""").asJsonObject,
                )
            }
        }

        assertEquals(409, failure.statusCode)
        assertEquals("run is not awaiting plan approval", failure.payload?.asJsonObject?.get("message")?.asString)
        assertEquals(1, interceptor.requests.size)
        assertEquals("POST", interceptor.requests.single().method)
    }

    @Test
    fun `a timed out decision is reported as an unknown outcome`() {
        val (api, interceptor) = apiWith { throw SocketTimeoutException("timeout") }

        val failure = assertThrows(MetaHarnessTimeoutException::class.java) {
            runBlocking {
                api.approveRun(
                    "run-1",
                    JsonParser.parseString("""{"decision":"REJECT"}""").asJsonObject,
                )
            }
        }

        assertNull(failure.statusCode)
        assertTrue(!failure.message!!.contains(token))
        assertEquals(1, interceptor.requests.size)
    }
}
