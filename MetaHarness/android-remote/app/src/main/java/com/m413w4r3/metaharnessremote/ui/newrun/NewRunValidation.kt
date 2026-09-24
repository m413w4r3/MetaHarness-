package com.m413w4r3.metaharnessremote.ui.newrun

import com.m413w4r3.metaharnessremote.api.MetaHarnessApi

/**
 * Result of validating the New Run form.
 *
 * [Valid] carries what `POST /v1/runs` is sent: the SPEC byte for byte, and
 * the run id, or null when the field is empty and the gateway generates it.
 */
sealed interface NewRunValidation {

    data class Valid(val spec: String, val runId: String?) : NewRunValidation

    /** A refused form; a null error means that field is fine. */
    data class Invalid(
        val specError: String? = null,
        val runIdError: String? = null,
    ) : NewRunValidation
}

/**
 * Validate the form before any request: the SPEC must hold text within
 * [MetaHarnessApi.MAX_SPEC_BYTES] UTF-8 bytes, and the run id, when the
 * operator typed one, must match [MetaHarnessApi.RUN_ID_PATTERN].
 *
 * Both fields are reported at once, so one tap shows everything to fix.
 */
fun validateNewRun(spec: String, runId: String): NewRunValidation {
    val specError = when {
        spec.isBlank() -> SPEC_REQUIRED
        spec.toByteArray(Charsets.UTF_8).size > MetaHarnessApi.MAX_SPEC_BYTES -> SPEC_TOO_LARGE
        else -> null
    }
    val selectedRunId = runId.trim().takeIf { it.isNotEmpty() }
    val runIdError = when {
        selectedRunId == null || MetaHarnessApi.RUN_ID_PATTERN.matches(selectedRunId) -> null
        else -> RUN_ID_INVALID
    }
    return if (specError == null && runIdError == null) {
        NewRunValidation.Valid(spec = spec, runId = selectedRunId)
    } else {
        NewRunValidation.Invalid(specError = specError, runIdError = runIdError)
    }
}

private const val SPEC_REQUIRED = "SPEC is required"
private const val SPEC_TOO_LARGE = "SPEC must be at most 48 KiB of UTF-8 text"
private const val RUN_ID_INVALID =
    "Run ID must start with a letter or digit and may only contain letters, digits, `_`, `.` and `-`"
