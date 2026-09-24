package com.m413w4r3.metaharnessremote.ui.rundetail

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.m413w4r3.metaharnessremote.ui.runs.failureSummary

/** One implementation step of the plan the run executes. */
data class RunStep(
    val id: String,
    val title: String?,
    val executionClass: String?,
    /** The implementer the run recorded for this step, when it has one. */
    val profileId: String?,
    /** The planner's recommendation, shown when no implementer was recorded. */
    val recommendedProfile: String?,
) {
    /** The profile the step runs with: the recorded one, else the recommendation. */
    val profile: String? get() = profileId ?: recommendedProfile

    /** What [profile] is: a recorded selection, or only a recommendation. */
    val profileLabel: String get() = if (profileId != null) PROFILE_LABEL else RECOMMENDED_LABEL
}

/** One label/value line of the overview block. */
data class OverviewLine(val label: String, val value: String)

/** Everything the Run Detail screen renders of one run. */
data class RunDetailView(
    val runId: String,
    val status: String?,
    val planTitle: String?,
    val updatedAt: String?,
    /** The failure summary, in the same words as the board's card. */
    val failure: String?,
    val candidateSha: String?,
    val cycle: Int?,
    val overview: List<OverviewLine>,
    val plan: String?,
    val steps: List<RunStep>,
)

/**
 * Read the fields the screen displays out of one `GET /v1/runs/{runId}` reply.
 *
 * The run detail document is large and still moving, so every field is optional
 * and every type is checked before use: a missing or differently shaped field is
 * dropped rather than failing the screen. [runId] is the id the screen was opened
 * with, and is what the screen shows when the document carries none.
 */
fun runDetailView(document: JsonObject, runId: String): RunDetailView {
    val state = document.obj("state")
    val cycle = document.int("cycle") ?: state?.int("cycle")
    return RunDetailView(
        runId = document.text("run_id") ?: runId,
        status = document.text("status"),
        planTitle = document.text("plan_title")
            ?: document.obj("task_plan")?.text("title")
            ?: state?.obj("planner")?.text("title"),
        updatedAt = document.text("updated_at"),
        failure = failureSummary(document.get("failure")),
        candidateSha = candidateSha(document, state, cycle),
        cycle = cycle,
        overview = overviewLines(document.obj("overview")),
        plan = boundedText(document.obj("plan")?.text("raw"))
            ?: boundedText(document.obj("plan")?.text("contract")),
        steps = runSteps(document, state),
    )
}

/**
 * The candidate commit of the current cycle, the shortest path to the commit
 * the operator would inspect.
 *
 * `candidate` is keyed by cycle (`"001"`, `"002"`, …), so the entry of the
 * current cycle comes first, then the commit the state recorded for the run,
 * then the summary's commit sha, which the board already shows.
 */
private fun candidateSha(document: JsonObject, state: JsonObject?, cycle: Int?): String? {
    val currentCycle = cycle?.let { number ->
        document.obj("candidate")?.obj(number.toString().padStart(CYCLE_DIGITS, '0'))
    }
    return currentCycle?.text("commit_sha")
        ?: state?.text("candidate_commit_sha")
        ?: document.text("commit_sha")
}

/**
 * The overview lines of the desktop run page: what the run does now, what comes
 * next, how it executes, where it publishes and how it could resume.
 */
private fun overviewLines(overview: JsonObject?): List<OverviewLine> {
    if (overview == null) return emptyList()
    val lines = mutableListOf<OverviewLine>()
    fun add(label: String, value: String?) {
        if (value != null) lines += OverviewLine(label, value)
    }
    add(CURRENT_LABEL, overview.text("current_label"))
    add(NEXT_LABEL, overview.text("next_label"))
    add(EXECUTION_LABEL, overview.text("execution_label"))
    add(
        PUBLISH_LABEL,
        listOfNotNull(overview.text("publish_target"), overview.text("publish_target_detail"))
            .joinToString(" · ")
            .ifEmpty { null },
    )
    val resume = overview.obj("resume")
    val resumable = resume?.bool("resumable")
    val resumeLabel = resume?.text("label")
    add(
        RESUME_LABEL,
        when {
            resumable == null -> resumeLabel
            resumable -> resumeLabel ?: RESUMABLE
            else -> listOfNotNull(resumeLabel, NOT_RESUMABLE).joinToString(" · ")
        },
    )
    return lines
}

/**
 * The steps of the approved plan, read from `implementation_bundle.steps` and
 * enriched with the profile the run recorded for the same step id.
 *
 * Each step keeps its bundle order; an entry without a usable id is dropped,
 * because a step the operator cannot identify has nothing to show.
 */
private fun runSteps(document: JsonObject, state: JsonObject?): List<RunStep> {
    val entries = document.obj("implementation_bundle")?.array("steps") ?: return emptyList()
    val selected = stepProfiles(state?.array("steps"), PROFILE_ID)
    val recommended = stepProfiles(state?.obj("planner")?.array("steps"), RECOMMENDED_PROFILE)
    return entries.mapNotNull { element ->
        val step = element.takeIf { it.isJsonObject }?.asJsonObject ?: return@mapNotNull null
        val id = step.text("id") ?: return@mapNotNull null
        RunStep(
            id = id,
            title = step.text("title"),
            executionClass = step.text("execution_class"),
            profileId = selected[id],
            recommendedProfile = recommended[id],
        )
    }
}

/** `id -> profile` of one step list, keeping the first profile seen per id. */
private fun stepProfiles(steps: JsonArray?, key: String): Map<String, String> {
    if (steps == null) return emptyMap()
    val profiles = LinkedHashMap<String, String>()
    for (element in steps) {
        val step = element.takeIf { it.isJsonObject }?.asJsonObject ?: continue
        val id = step.text("id") ?: continue
        step.text(key)?.let { profiles.putIfAbsent(id, it) }
    }
    return profiles
}

/** The longest free-text field the screen renders, before it is cut. */
internal const val MAX_FREE_TEXT_CHARS = 8_000

/**
 * Bound one long free-text field. The gateway bounds the document it sends, but
 * that bound is far more text than a phone can render in one screen.
 */
internal fun boundedText(value: String?, limit: Int = MAX_FREE_TEXT_CHARS): String? {
    val text = value?.takeIf { it.isNotBlank() } ?: return null
    return if (text.length <= limit) text else text.take(limit).trimEnd() + "…"
}

/** The trimmed string at [key], or null when it is absent, blank or not a string. */
private fun JsonObject.text(key: String): String? =
    get(key)
        ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
        ?.asString
        ?.trim()
        ?.takeIf { it.isNotEmpty() }

/** The cycle number at [key], or null when it cannot be read as a positive one. */
private fun JsonObject.int(key: String): Int? {
    val primitive = get(key)?.takeIf { it.isJsonPrimitive }?.asJsonPrimitive ?: return null
    val value = if (primitive.isNumber) {
        primitive.asInt
    } else {
        primitive.asString.trim().toIntOrNull()
    }
    return value?.takeIf { it >= 1 }
}

/** The boolean at [key], or null when it is absent or not a boolean. */
private fun JsonObject.bool(key: String): Boolean? =
    get(key)
        ?.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isBoolean }
        ?.asBoolean

/** The object at [key], or null when it is absent or not an object. */
private fun JsonObject.obj(key: String): JsonObject? =
    get(key)?.takeIf { it.isJsonObject }?.asJsonObject

/** The array at [key], or null when it is absent or not an array. */
private fun JsonObject.array(key: String): JsonArray? =
    get(key)?.takeIf { it.isJsonArray }?.asJsonArray

/** Cycle directories are named `001`, `002`, … */
private const val CYCLE_DIGITS = 3

private const val PROFILE_ID = "profile_id"
private const val RECOMMENDED_PROFILE = "recommended_profile"

private const val PROFILE_LABEL = "PROFILE"
private const val RECOMMENDED_LABEL = "RECOMMENDED"

private const val CURRENT_LABEL = "CURRENT"
private const val NEXT_LABEL = "NEXT"
private const val EXECUTION_LABEL = "EXECUTION"
private const val PUBLISH_LABEL = "PUBLISH TARGET"
private const val RESUME_LABEL = "RESUME"
private const val RESUMABLE = "resumable"
private const val NOT_RESUMABLE = "not resumable"
