package com.m413w4r3.metaharnessremote.ui.rundetail

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject

/**
 * The plan approval block of the Run Detail screen.
 *
 * Everything here is a pure read of the documents the gateway publishes:
 * `GET /v1/runs/{runId}` carries the gate, the plan steps, the recorded
 * execution selection and the run options; `GET /v1/model-profiles` carries the
 * profiles the operator may choose; `GET /v1/config` carries the capabilities
 * that say whether plan approval is available at all.
 *
 * Nothing here builds a local field name: an approval body speaks
 * `step_profiles` keyed by plan step id, and translating those keys into the
 * local names belongs to the gateway.
 */

/** The roles a profile may hold, as the gateway publishes them. */
internal const val REVIEWER_ROLE = "reviewer"
internal const val REVISER_ROLE = "reviser"
internal const val REPAIR_ROLE = "repair"
private const val IMPLEMENTER_ROLE = "implementer"

/** The two decisions the gateway accepts. */
internal const val APPROVE = "APPROVE"
internal const val REJECT = "REJECT"

/** The external field names of an approval body. */
internal const val DECISION_FIELD = "decision"
internal const val STEP_PROFILES_FIELD = "step_profiles"

/** The status of a run that waits for a decision on its plan. */
private const val AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"

/** The class a step that declares none runs as, as on the desktop form. */
private const val DEFAULT_EXECUTION_CLASS = "MECHANICAL"

/** The run-options fields that route each execution class. */
private const val MECHANICAL_PROFILE = "mechanical_profile"
private const val REASONING_PROFILE = "reasoning_profile"
private const val AGENTIC_PROFILE = "agentic_profile"

/** One model profile the form may select. */
data class ProfileOption(
    val id: String,
    /** The roles of the profile, lowercase; `implementer` may run a step. */
    val roles: Set<String> = emptySet(),
    /**
     * The execution classes the profile declares, uppercase. Empty means it
     * declares none, which allows every class — the desktop form reads the
     * field the same way.
     */
    val executionClasses: Set<String> = emptySet(),
) {

    /** The profile may implement [step]: it holds the role and the class. */
    fun implementsStep(step: ApprovalStep): Boolean =
        IMPLEMENTER_ROLE in roles &&
            (executionClasses.isEmpty() || step.executionClass in executionClasses)

    /** The profile may act in [role]. */
    fun holds(role: String): Boolean = role in roles
}

/** One step of the plan the decision applies to. */
data class ApprovalStep(
    val id: String,
    val title: String?,
    /** Always set: a step that declares none is treated as mechanical. */
    val executionClass: String,
)

/** The model profiles of `GET /v1/model-profiles` and the defaults it publishes. */
data class ApprovalProfiles(
    val options: List<ProfileOption> = emptyList(),
    /** The requested profile per role field, keyed as `defaults` publishes it. */
    val defaults: Map<String, String> = emptyMap(),
) {

    /** The published profile [id] names, or null when there is none. */
    fun option(id: String?): ProfileOption? =
        id?.takeIf { it.isNotEmpty() }?.let { wanted -> options.firstOrNull { it.id == wanted } }

    /** True when [id] is a published profile holding [role]. */
    fun offers(id: String?, role: String): Boolean = option(id)?.holds(role) == true

    /** Every profile that may implement [step], in the gateway's order. */
    fun implementers(step: ApprovalStep): List<ProfileOption> = options.filter { it.implementsStep(step) }

    /** Every profile holding [role], in the gateway's order. */
    fun withRole(role: String): List<ProfileOption> = options.filter { it.holds(role) }
}

/**
 * What a plan approval needs, read from one run document.
 *
 * The two booleans come from the immutable run options of the run, never from
 * today's configuration: a reviser or a repair profile that the run does not
 * use is refused by MetaHarness, so the form neither offers nor sends one.
 */
data class ApprovalGate(
    val steps: List<ApprovalStep>,
    /** The run enables semantic revision: the semantic reviser applies. */
    val semanticRevisionEnabled: Boolean,
    /** The run holds a correction budget: the check repair profile applies. */
    val checkRepairEnabled: Boolean,
)

/** The operator's choices, or the ones the form starts from. */
data class ApprovalSelection(
    val finalReviewerProfile: String = "",
    val semanticReviserProfile: String = "",
    val checkRepairProfile: String = "",
    /** The implementer per plan step id. */
    val stepProfiles: Map<String, String> = emptyMap(),
)

/**
 * True while the run waits for a plan decision and has recorded none:
 * `status == awaiting_plan_approval`, `approval.recorded != true` and
 * `approval.awaiting != false`. A field the run does not publish never blocks
 * the block, exactly as on the desktop.
 */
fun awaitingPlanDecision(document: JsonObject): Boolean {
    if (document.text("status")?.lowercase() != AWAITING_PLAN_APPROVAL) return false
    val approval = document.obj("approval") ?: return true
    if (approval.bool("recorded") == true) return false
    return approval.bool("awaiting") != false
}

/**
 * The approval block of one run, or null when it must not be shown.
 *
 * [capabilities] is the `capabilities` object of the run document or, when it
 * carries none, the one of `GET /v1/config`; a missing or differently shaped
 * object does not hide the block, and only an explicit `plan_approval: false`
 * does.
 */
fun planApprovalGate(document: JsonObject, capabilities: JsonObject?): ApprovalGate? {
    if (!awaitingPlanDecision(document)) return null
    if (!planApprovalAvailable(capabilities)) return null
    val pipeline = document.obj("run_options")?.obj("pipeline")
    return ApprovalGate(
        steps = approvalSteps(document),
        semanticRevisionEnabled = pipeline?.bool("semantic_revision_enabled") == true,
        checkRepairEnabled = pipeline?.positive("max_check_repair_attempts") == true ||
            pipeline?.positive("max_correction_cycles") == true,
    )
}

/**
 * False only when the gateway reports plan approval as explicitly unavailable:
 * a missing or differently shaped capability is not a refusal.
 */
fun planApprovalAvailable(capabilities: JsonObject?): Boolean =
    capabilities?.bool("plan_approval") != false

/** The capabilities that gate the approval: the run's own, else the configuration's. */
fun approvalCapabilities(document: JsonObject, config: JsonObject?): JsonObject? =
    document.obj("capabilities")
        ?: document.obj("overview")?.obj("capabilities")
        ?: config?.obj("capabilities")

/** Read `GET /v1/model-profiles`; an unknown or differently shaped field is dropped. */
fun approvalProfiles(document: JsonObject): ApprovalProfiles =
    ApprovalProfiles(
        options = document.array("profiles")?.mapNotNull(::profileOption).orEmpty(),
        defaults = document.obj("defaults")?.stringFields().orEmpty(),
    )

/**
 * The choices the form starts from: what the run recorded for a step, else the
 * profile its run options routed to that execution class, else the first
 * compatible profile. The three role profiles follow the same order, from the
 * run options, then from the gateway's own defaults.
 */
fun defaultApprovalSelection(
    document: JsonObject,
    gate: ApprovalGate,
    profiles: ApprovalProfiles,
): ApprovalSelection {
    val requested = document.obj("run_options")?.obj("profiles")
    val recorded = recordedStepProfiles(document)
    fun roleProfile(key: String, role: String): String =
        sequenceOf(requested?.text(key), profiles.defaults[key], profiles.withRole(role).firstOrNull()?.id)
            .firstOrNull { profiles.offers(it, role) }
            .orEmpty()
    return ApprovalSelection(
        finalReviewerProfile = roleProfile("final_reviewer_profile", REVIEWER_ROLE),
        semanticReviserProfile = roleProfile("semantic_reviser_profile", REVISER_ROLE),
        checkRepairProfile = roleProfile("check_repair_profile", REPAIR_ROLE),
        stepProfiles = gate.steps.associate { step ->
            step.id to sequenceOf(
                recorded[step.id],
                requested?.text(routeField(step.executionClass)),
                profiles.implementers(step).firstOrNull()?.id,
            ).firstOrNull { id -> profiles.option(id)?.implementsStep(step) == true }.orEmpty()
        },
    )
}

/**
 * The body of an approval decision, or null when the selection is not complete
 * enough to send one.
 *
 * The three role profiles are sent only when the run's own options use them: a
 * profile of a role the run does not enable is refused by MetaHarness, so the
 * form never sends one it cannot honour.
 */
fun approvalPayload(
    selection: ApprovalSelection,
    gate: ApprovalGate,
    profiles: ApprovalProfiles,
): JsonObject? {
    if (gate.steps.isEmpty()) return null
    val reviewer = selection.finalReviewerProfile.takeIf { profiles.offers(it, REVIEWER_ROLE) } ?: return null
    val reviser = selection.semanticReviserProfile.takeIf { profiles.offers(it, REVISER_ROLE) }
    val repair = selection.checkRepairProfile.takeIf { profiles.offers(it, REPAIR_ROLE) }
    if (gate.semanticRevisionEnabled && reviser == null) return null
    if (gate.checkRepairEnabled && repair == null) return null
    val stepProfiles = JsonObject()
    for (step in gate.steps) {
        val implementer = profiles.option(selection.stepProfiles[step.id])
            ?.takeIf { it.implementsStep(step) }
            ?: return null
        stepProfiles.addProperty(step.id, implementer.id)
    }
    return JsonObject().apply {
        addProperty(DECISION_FIELD, APPROVE)
        addProperty("final_reviewer_profile", reviewer)
        if (gate.semanticRevisionEnabled) addProperty("semantic_reviser_profile", reviser)
        if (gate.checkRepairEnabled) addProperty("check_repair_profile", repair)
        add(STEP_PROFILES_FIELD, stepProfiles)
    }
}

/** The body of a rejection: the decision and nothing else. */
fun rejectionPayload(): JsonObject = JsonObject().apply { addProperty(DECISION_FIELD, REJECT) }

/** The steps of `implementation_bundle.steps`; one without an id is dropped. */
private fun approvalSteps(document: JsonObject): List<ApprovalStep> =
    document.obj("implementation_bundle").elements("steps").mapNotNull { element ->
        val step = element.asObjectOrNull() ?: return@mapNotNull null
        ApprovalStep(
            id = step.text("id") ?: return@mapNotNull null,
            title = step.text("title"),
            executionClass = step.text("execution_class")?.uppercase() ?: DEFAULT_EXECUTION_CLASS,
        )
    }

/** `step id -> implementer` of `execution_selection.steps`, keeping the first. */
private fun recordedStepProfiles(document: JsonObject): Map<String, String> {
    val profiles = LinkedHashMap<String, String>()
    for (element in document.obj("execution_selection").elements("steps")) {
        val step = element.asObjectOrNull() ?: continue
        val id = step.text("step_id") ?: continue
        val profile = step.obj("implementer")?.text("profile_id") ?: continue
        profiles.putIfAbsent(id, profile)
    }
    return profiles
}

/** The elements of an array field, or an empty list when the object carries none. */
private fun JsonObject?.elements(key: String): List<JsonElement> =
    this?.get(key)?.takeIf { it.isJsonArray }?.asJsonArray?.toList().orEmpty()

/** The run-options field that routes one execution class. */
private fun routeField(executionClass: String): String = when (executionClass) {
    "REASONING" -> REASONING_PROFILE
    "AGENTIC" -> AGENTIC_PROFILE
    else -> MECHANICAL_PROFILE
}

/** One entry of the `profiles` array, or null when it carries no usable id. */
private fun profileOption(element: JsonElement): ProfileOption? {
    val profile = element.asObjectOrNull() ?: return null
    val id = profile.text("id") ?: return null
    val classes = profile.array("execution_classes") ?: profile.array("classes")
    return ProfileOption(
        id = id,
        roles = profile.strings("roles").map(String::lowercase).toSet(),
        executionClasses = classes.strings().map(String::uppercase).toSet(),
    )
}

/** The string entries of an array, or an empty list when it is not one. */
private fun JsonArray?.strings(): List<String> =
    this?.mapNotNull { element ->
        element.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.takeIf { it.isNotEmpty() }
    }.orEmpty()

private fun JsonObject.strings(key: String): List<String> = array(key).strings()

private fun JsonObject.stringFields(): Map<String, String> =
    entrySet().mapNotNull { (key, value) ->
        value.takeIf { it.isJsonPrimitive && it.asJsonPrimitive.isString }
            ?.asString
            ?.trim()
            ?.takeIf { it.isNotEmpty() }
            ?.let { key to it }
    }.toMap()

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

/** True when [key] holds a number greater than zero, as a number or as a string. */
private fun JsonObject.positive(key: String): Boolean {
    val primitive = get(key)?.takeIf { it.isJsonPrimitive }?.asJsonPrimitive ?: return false
    val value = if (primitive.isNumber) primitive.asDouble else primitive.asString.trim().toDoubleOrNull()
    return value != null && value > 0
}

/** The object at [key], or null when it is absent or not an object. */
private fun JsonObject.obj(key: String): JsonObject? =
    get(key)?.takeIf { it.isJsonObject }?.asJsonObject

/** The array at [key], or null when it is absent or not an array. */
private fun JsonObject.array(key: String): JsonArray? =
    get(key)?.takeIf { it.isJsonArray }?.asJsonArray

private fun JsonElement.asObjectOrNull(): JsonObject? = takeIf { it.isJsonObject }?.asJsonObject
