package com.m413w4r3.metaharnessremote.ui.newrun

import com.google.gson.JsonElement
import com.m413w4r3.metaharnessremote.api.MetaHarnessException
import com.m413w4r3.metaharnessremote.api.MetaHarnessTimeoutException

/**
 * What a timed-out CREATE RUN shows.
 *
 * A timeout is not a failure: the request reached the gateway, so the run may
 * exist. The operator refreshes the runs list before deciding anything, and
 * the app never sends the create again on its own.
 */
const val CREATE_TIMEOUT_MESSAGE =
    "Response timed out.\n" +
        "The run may have been created.\n" +
        "Refresh the runs list before trying again."

/**
 * The line the New Run screen shows when a CREATE RUN failed.
 *
 * A timeout keeps its own wording; every other failure shows the reason the
 * gateway sent with its answer, falling back to the transport message.
 */
fun newRunFailure(failure: Throwable): String = when (failure) {
    is MetaHarnessTimeoutException -> CREATE_TIMEOUT_MESSAGE
    is MetaHarnessException -> failure.payload.gatewayMessage() ?: failure.describe()
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
