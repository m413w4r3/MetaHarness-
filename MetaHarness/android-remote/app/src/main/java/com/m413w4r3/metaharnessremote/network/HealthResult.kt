package com.m413w4r3.metaharnessremote.network

/** Outcome of `GET /v1/health`. */
sealed interface HealthResult {
    data class Connected(val status: String, val apiVersion: Int?) : HealthResult
    data class Failed(val message: String) : HealthResult
}

/** Subset of the gateway health document, kept tolerant to added fields. */
internal data class HealthDocument(
    val status: String? = null,
    val api_version: Int? = null,
)
