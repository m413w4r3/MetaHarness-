package com.m413w4r3.metaharnessremote.api

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.annotations.SerializedName

/**
 * Document of `GET /v1/health`. Unknown fields are ignored.
 */
data class HealthResponse(
    val service: String? = null,
    val status: String? = null,
    @SerializedName("api_version") val apiVersion: Int? = null,
)

/**
 * One entry of `GET /v1/runs`.
 *
 * Every field is nullable: a summary is rendered from a run state that may
 * still be incomplete. [failure] is the raw failure document; a JSON null
 * arrives as [com.google.gson.JsonNull].
 */
data class RunSummary(
    @SerializedName("run_id") val runId: String? = null,
    val status: String? = null,
    @SerializedName("updated_at") val updatedAt: String? = null,
    @SerializedName("plan_title") val planTitle: String? = null,
    @SerializedName("commit_sha") val commitSha: String? = null,
    val failure: JsonElement? = null,
)

/**
 * Document of `GET /v1/runs/{runId}/progress`: the offset to resume from and
 * the newly visible event summaries.
 */
data class ProgressResponse(
    @SerializedName("next_offset") val nextOffset: Long = 0L,
    val events: List<String> = emptyList(),
)

/**
 * Document of `GET /v1/runs/{runId}`.
 *
 * The run detail schema is large and still moving, so only the raw document is
 * exposed here; callers read the fields they need from [raw].
 */
data class RunDetail(val raw: JsonObject)

/** Envelope of `GET /v1/runs`. */
internal data class RunListDocument(val runs: List<RunSummary> = emptyList())
