package com.m413w4r3.metaharnessremote.ui.rundetail

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.m413w4r3.metaharnessremote.api.MetaHarnessTimeoutException

/**
 * The scope approval, the plan recovery and the resume of the Run Detail screen.
 *
 * Every block is a pure read of the documents the gateway publishes:
 * `GET /v1/runs/{runId}` carries the repair scope the run requested, the
 * decision already recorded, the plan recovery it offers under
 * `plan_recovery`, and the resume checkpoint under `overview.resume`;
 * `GET /v1/config` carries the capabilities that say whether each action is
 * available at all. Nothing here retries anything: a decision, a replacement
 * plan and a resume are single mutations the operator asks for.
 */

/** The status of a run that waits for a decision on its repair scope. */
private const val WAITING_SCOPE_APPROVAL = "waiting_scope_approval"

/** The repair scope of one run: the paths the requested delta adds. */
data class ScopeGate(val addedPaths: List<String>)

/** The resume of one run: the checkpoint label, when the run publishes one. */
data class ResumeGate(val label: String?)

/** The plan recovery of one run: why it is offered, and what it accepts. */
data class PlanRecoveryGate(
    /** Why the run may recover a plan, when the gateway publishes a reason. */
    val reason: String?,
    /** The largest replacement the gateway accepts, in UTF-8 bytes. */
    val maxBytes: Long?,
    /** The plan the run rejected, when the document still holds it. */
    val rejectedPlan: String?,
)

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

/**
 * True when the run publishes a plan recovery the operator may attempt:
 * `plan_recovery.eligible == true`, the eligibility the gateway computes from
 * the durable run shape. The orchestrator re-checks everything, so this is a
 * gate to show the form, never a promise that a replacement is accepted.
 */
fun planRecoverable(document: JsonObject): Boolean =
    document.obj("plan_recovery")?.bool("eligible") == true

/**
 * The plan recovery block of one run, or null when it must not be shown:
 * `plan_recovery.eligible == true`, and no explicit `recover_plan: false`.
 *
 * The block only collects the replacement: the text goes back through the
 * exact parser and policies a planner answer goes through, so validating the
 * META PLAN v2 document itself stays the gateway's authority.
 */
fun planRecoveryGate(document: JsonObject, capabilities: JsonObject?): PlanRecoveryGate? {
    if (!planRecoverable(document)) return null
    if (capabilities?.bool("recover_plan") == false) return null
    val recovery = document.obj("plan_recovery")
    return PlanRecoveryGate(
        reason = recovery?.text("reason"),
        maxBytes = recovery?.bytes("max_bytes"),
        rejectedPlan = rejectedPlan(document),
    )
}

/**
 * The UTF-8 size of one replacement plan, counted the way the gateway counts
 * it: the bytes of the text as sent, whitespace included.
 */
fun replacementPlanBytes(plan: String): Long = plan.toByteArray(Charsets.UTF_8).size.toLong()

/**
 * True when [plan] may be sent: it holds text, and its UTF-8 size fits
 * [maxBytes]. A bound the gateway does not publish — or publishes as zero —
 * accepts nothing, exactly as the desktop form reads it.
 */
fun replacementPlanAccepted(plan: String, bytes: Long, maxBytes: Long?): Boolean =
    plan.isNotBlank() && maxBytes != null && maxBytes > 0 && bytes <= maxBytes

/**
 * The confirmation a replacement is sent under.
 *
 * A replacement is published as the plan authority of the run, so the operator
 * states the intent before the single request leaves the phone.
 */
const val RECOVER_CONFIRMATION_MESSAGE =
    "Replace the rejected plan with this META PLAN v2?\n" +
        "MetaHarness will validate it before continuing."

/**
 * What a timed-out replacement shows.
 *
 * A timeout is not a failure: the request reached the gateway, so the plan may
 * have been recorded. The run is read again — that read, not a second
 * replacement, is what tells the operator where the run stands.
 */
const val RECOVER_TIMEOUT_MESSAGE =
    "Response timed out.\n" +
        "The replacement plan may have been recorded.\n" +
        "The run is read again to show what it recorded."

/**
 * The line the plan recovery block shows when the request failed.
 *
 * A timeout carries its own words; every other answer is described like one of
 * a decision — a conflict is prefixed, a refusal keeps the gateway's message —
 * and nothing is ever sent twice.
 */
fun recoverFailure(failure: Throwable): String =
    if (failure is MetaHarnessTimeoutException) RECOVER_TIMEOUT_MESSAGE else approvalFailure(failure)

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

/**
 * The plan the run rejected: the planner's raw answer, else the plan the
 * document publishes. Both are bounded before they are rendered, because the
 * gateway may hold a whole planner reply in either.
 */
private fun rejectedPlan(document: JsonObject): String? =
    boundedText(document.contents("planner_raw"))
        ?: boundedText(document.obj("plan")?.contents("raw"))

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

/** The integer at [key], or null when it is absent, negative or not a number. */
private fun JsonObject.bytes(key: String): Long? {
    val primitive = get(key)?.takeIf { it.isJsonPrimitive }?.asJsonPrimitive ?: return null
    val value = if (primitive.isNumber) primitive.asLong else primitive.asString.trim().toLongOrNull()
    return value?.takeIf { it >= 0 }
}

/** The string at [key] as sent, or null when it is absent, empty or not a string. */
private fun JsonObject.contents(key: String): String? =
    get(key)
        ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
        ?.asString
        ?.takeIf { it.isNotEmpty() }

/** The object at [key], or null when it is absent or not an object. */
private fun JsonObject.obj(key: String): JsonObject? =
    get(key)?.takeIf { it.isJsonObject }?.asJsonObject
