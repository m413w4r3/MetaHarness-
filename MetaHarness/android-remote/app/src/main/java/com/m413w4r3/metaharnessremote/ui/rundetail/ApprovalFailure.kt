package com.m413w4r3.metaharnessremote.ui.rundetail

import com.google.gson.JsonElement
import com.m413w4r3.metaharnessremote.api.MetaHarnessException
import com.m413w4r3.metaharnessremote.api.MetaHarnessTimeoutException

/**
 * What a timed-out decision shows.
 *
 * A timeout is not a failure: the request reached the gateway, so the decision
 * may have been recorded. The run is read again — that read, not a second
 * decision, is what tells the operator where the run stands.
 */
const val APPROVAL_TIMEOUT_MESSAGE =
    "Response timed out.\n" +
        "The decision may have been recorded.\n" +
        "The run is read again to show what it recorded."

/** Shown in front of a decision MetaHarness refused because the run moved on. */
private const val REFUSED_PREFIX = "Refused: "

/** `POST /v1/runs/{runId}/approval` answered a conflict: another state is current. */
private const val CONFLICT_STATUS = 409

/**
 * The line the approval block shows when a decision failed.
 *
 * A conflict carries the words of the gateway — the run is not awaiting a plan
 * decision any more, or a decision is already recorded — and the screen reads
 * the run again whatever the answer was. Nothing is ever sent twice.
 */
fun approvalFailure(failure: Throwable): String = when (failure) {
    is MetaHarnessTimeoutException -> APPROVAL_TIMEOUT_MESSAGE

    is MetaHarnessException -> {
        val message = failure.payload.gatewayMessage() ?: failure.describe()
        if (failure.statusCode == CONFLICT_STATUS) REFUSED_PREFIX + message else message
    }

    else -> failure.describe()
}

/** The `message` of a gateway error document, or null when it carries none. */
private fun JsonElement?.gatewayMessage(): String? {
    val document = this?.takeIf { it.isJsonObject }?.asJsonObject ?: return null
    val message = document.get("message")?.takeIf { it.isJsonPrimitive }?.asString ?: return null
    return message.trim().takeIf { it.isNotEmpty() }
}

private fun Throwable.describe(): String =
    message?.trim()?.takeIf { it.isNotEmpty() } ?: javaClass.simpleName
