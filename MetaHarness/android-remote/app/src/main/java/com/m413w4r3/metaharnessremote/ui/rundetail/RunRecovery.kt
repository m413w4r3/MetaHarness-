package com.m413w4r3.metaharnessremote.ui.rundetail

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.m413w4r3.metaharnessremote.api.MetaHarnessTimeoutException

/**
 * The scope approval and the resume of the Run Detail screen.
 *
 * Both blocks are pure reads of the documents the gateway publishes:
 * `GET /v1/runs/{runId}` carries the repair scope the run requested, the
 * decision already recorded, and the resume checkpoint under `overview.resume`;
 * `GET /v1/config` carries the capabilities that say whether each action is
 * available at all. Nothing here retries anything: a decision and a resume are
 * single mutations the operator asks for.
 */

/** The status of a run that waits for a decision on its repair scope. */
private const val WAITING_SCOPE_APPROVAL = "waiting_scope_approval"

/** The repair scope of one run: the paths the requested delta adds. */
data class ScopeGate(val addedPaths: List<String>)

/** The resume of one run: the checkpoint label, when the run publishes one. */
data class ResumeGate(val label: String?)

/**
 * True while the run waits for a scope decision and publishes the delta it
 * applies to: `status == waiting_scope_approval`, `scope_approval.recorded !=
 * true`, `scope_approval.awaiting != false`, and a non-empty `scope_delta`. A
 * field the run does not publish never blocks the block.
 */
fun awaitingScopeDecision(document: JsonObject): Boolean {
    if (document.text("status")?.lowercase() != WAITING_SCOPE_APPROVAL) return false
    val approval = document.obj("scope_approval")
    if (approval?.bool("recorded") == true) return false
    if (approval?.bool("awaiting") == false) return false
    return scopeDelta(document) != null
}

/**
 * The scope approval block of one run, or null when it must not be shown.
 *
 * [capabilities] is the `capabilities` object of the run document or, when it
 * carries none, the one of `GET /v1/config`; only an explicit
 * `scope_approval: false` hides the block.
 */
fun scopeApprovalGate(document: JsonObject, capabilities: JsonObject?): ScopeGate? {
    if (!awaitingScopeDecision(document)) return null
    if (capabilities?.bool("scope_approval") == false) return null
    return ScopeGate(addedPaths = scopeDelta(document)?.strings("added_paths").orEmpty())
}

/**
 * True when the run publishes a resume the operator may ask for:
 * `overview.resume.resumable == true`, the run document's resume checkpoint.
 */
fun resumableRun(document: JsonObject): Boolean =
    document.obj("overview")?.obj("resume")?.bool("resumable") == true

/**
 * The resume block of one run, or null when it must not be shown.
 *
 * [capabilities] resolves like the one of the scope approval; only an explicit
 * `resume: false` hides the block. The label the run publishes names the
 * checkpoint the resume continues from.
 */
fun resumeGate(document: JsonObject, capabilities: JsonObject?): ResumeGate? {
    if (!resumableRun(document)) return null
    if (capabilities?.bool("resume") == false) return null
    return ResumeGate(document.obj("overview")?.obj("resume")?.text("label"))
}

/** The body of one scope decision: the decision alone. */
fun scopeDecisionPayload(decision: String): JsonObject =
    JsonObject().apply { addProperty(DECISION_FIELD, decision) }

/**
 * What a timed-out resume shows.
 *
 * A timeout is not a failure: the resume reached the gateway, so it may have
 * been accepted. The run is read again — that read, not a second resume, is
 * what tells the operator where the run stands.
 */
const val RESUME_TIMEOUT_MESSAGE =
    "The resume request may have been accepted.\n" +
        "Refresh before retrying."

/**
 * The line the resume block shows when the request failed.
 *
 * A timeout carries its own words; every other answer is described like one of
 * a decision — a conflict is prefixed, a refusal keeps the gateway's message —
 * and nothing is ever sent twice.
 */
fun resumeFailure(failure: Throwable): String =
    if (failure is MetaHarnessTimeoutException) RESUME_TIMEOUT_MESSAGE else approvalFailure(failure)

/**
 * The scope delta of the run: the document's own, else the one the state
 * recorded, because a contract-repair delta lives beside its step and the
 * state names it. An empty object is no delta at all.
 */
private fun scopeDelta(document: JsonObject): JsonObject? =
    document.obj("scope_delta")?.takeIf { it.size() > 0 }
        ?: document.obj("state")?.obj("scope_delta")?.takeIf { it.size() > 0 }

/** The string entries of one array field, or an empty list when it is absent. */
private fun JsonObject.strings(key: String): List<String> {
    val array: JsonArray = get(key)?.takeIf { it.isJsonArray }?.asJsonArray ?: return emptyList()
    return array.mapNotNull { element ->
        element.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.takeIf { it.isNotEmpty() }
    }
}

/** The trimmed string at [key], or null when it is absent, blank or not a string. */
private fun JsonObject.text(key: String): String? =
    get(key)
        ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
        ?.asString
        ?.trim()
        ?.takeIf { it.isNotEmpty() }

/** The boolean at [key], or null when it is absent or not a boolean. */
private fun JsonObject.bool(key: String): Boolean? =
    get(key)
        ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isBoolean }
        ?.asBoolean

/** The object at [key], or null when it is absent or not an object. */
private fun JsonObject.obj(key: String): JsonObject? =
    get(key)?.takeIf { it.isJsonObject }?.asJsonObject
