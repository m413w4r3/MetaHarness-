package com.m413w4r3.metaharnessremote.api

import com.google.gson.JsonElement

/**
 * Failure of a single MetaHarness Remote exchange.
 *
 * [statusCode] is the HTTP status of the response, or null when the request
 * never completed (connection failure, malformed or oversized body).
 * [payload] is the decoded JSON document of the answer when it carried one.
 *
 * The remote token is never part of the message, of [payload] or of any other
 * field: this exception can be logged or shown as-is.
 */
class MetaHarnessException(
    val statusCode: Int?,
    message: String,
    val payload: JsonElement? = null,
    cause: Throwable? = null,
) : Exception(message, cause)
