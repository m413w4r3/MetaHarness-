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

/**
 * Answer of `POST /v1/runs`.
 *
 * [runId] is the id of the run the gateway created, never blank: the Run
 * Detail screen opens from it, so an answer without a usable id is a failure
 * of the call rather than a response without an id.
 */
data class CreateRunResponse(val runId: String)

/**
 * Answer of `POST /v1/runs/{runId}/approval`.
 *
 * [decision] is the decision the gateway recorded: the one its answer names,
 * or, when it names none, the one that was sent — the call succeeded, so the
 * decision is durable. The screen reads the run again to show what it decided.
 */
data class ApprovalResponse(val decision: String)

/** Envelope of `GET /v1/runs`. */
internal data class RunListDocument(val runs: List<RunSummary> = emptyList())

/** Envelope of `POST /v1/runs`, parsed before the created run id is checked. */
internal data class CreateRunDocument(
    @SerializedName("run_id") val runId: String? = null,
)

/** Document of `POST /v1/runs/{runId}/approval`, parsed before its decision is checked. */
internal data class ApprovalDocument(
    @SerializedName("decision") val decision: String? = null,
)
