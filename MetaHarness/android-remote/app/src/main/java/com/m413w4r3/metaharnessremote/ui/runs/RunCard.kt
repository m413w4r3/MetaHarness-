package com.m413w4r3.metaharnessremote.ui.runs

import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.m413w4r3.metaharnessremote.api.RunSummary

/** One run, reduced to the fields the Runs screen displays. */
data class RunCard(
    val runId: String,
    val planTitle: String?,
    val status: String?,
    val updatedAt: String?,
    val commitSha: String?,
    val failure: String?,
    val category: RunCategory,
)

/** One board section and its runs, in the order the gateway returned them. */
data class RunSection(val category: RunCategory, val runs: List<RunCard>)

/**
 * Group `GET /v1/runs` into the board sections, in [RunCategory] order.
 *
 * Empty sections are dropped and summaries without a usable run id are
 * ignored: a card the operator cannot open has nothing to show.
 */
fun runSections(runs: List<RunSummary>): List<RunSection> {
    val cards = runs.mapNotNull(::runCard)
    return RunCategory.entries.mapNotNull { category ->
        cards
            .filter { it.category == category }
            .takeIf { it.isNotEmpty() }
            ?.let { RunSection(category, it) }
    }
}

/** Reduce one summary to a card, or null when it carries no run id. */
fun runCard(summary: RunSummary): RunCard? {
    val runId = summary.runId?.trim().orEmpty()
    if (runId.isEmpty()) return null
    return RunCard(
        runId = runId,
        planTitle = summary.planTitle.present(),
        status = summary.status.present(),
        updatedAt = summary.updatedAt.present(),
        commitSha = shortCommitSha(summary.commitSha),
        failure = failureSummary(summary.failure),
        category = RunCategory.of(summary.status),
    )
}

/** The leading [SHORT_SHA_LENGTH] characters of a commit SHA, or null. */
fun shortCommitSha(sha: String?): String? = sha.present()?.take(SHORT_SHA_LENGTH)

/**
 * One line summarising a `state.failure` document, or null when the run has
 * none. A document renders as `reason — detail`, mirroring the desktop list;
 * a bare value renders as itself.
 */
fun failureSummary(failure: JsonElement?): String? {
    if (failure == null || failure.isJsonNull) return null
    val text = when {
        failure.isJsonObject -> failure.asJsonObject.failureText()
        failure.isJsonPrimitive -> failure.asString
        else -> failure.toString()
    }
    val line = oneLine(text)
    if (line.isEmpty()) return null
    return if (line.length <= BOUNDED_FAILURE_CHARS) line
    else line.take(BOUNDED_FAILURE_CHARS).trimEnd() + "…"
}

private fun JsonObject.failureText(): String {
    val reason = text("reason")
    val detail = text("detail")
    val summary = listOfNotNull(reason, detail).joinToString(REASON_DETAIL_SEPARATOR)
    return summary.ifEmpty { if (isEmpty) "" else toString() }
}

private fun JsonObject.text(key: String): String? =
    get(key)?.takeIf { it.isJsonPrimitive }?.asString?.let(::oneLine)?.takeIf { it.isNotEmpty() }

private fun String?.present(): String? = this?.trim()?.takeIf { it.isNotEmpty() }

private fun oneLine(value: String): String = value.replace(WHITESPACE, " ").trim()

private val WHITESPACE = Regex("\\s+")

private const val SHORT_SHA_LENGTH = 7
private const val BOUNDED_FAILURE_CHARS = 240
private const val REASON_DETAIL_SEPARATOR = " — "
