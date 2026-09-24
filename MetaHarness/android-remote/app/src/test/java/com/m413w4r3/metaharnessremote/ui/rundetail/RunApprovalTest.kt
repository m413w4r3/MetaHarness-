package com.m413w4r3.metaharnessremote.ui.rundetail

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

private fun approvalDocument(json: String): JsonObject = JsonParser.parseString(json).asJsonObject

/** One profile per role and execution class, in the order the gateway publishes them. */
private const val PROFILE_CATALOGUE = """
    [
      {"id": "impl-any", "roles": ["implementer"]},
      {"id": "impl-mech", "roles": ["implementer"], "execution_classes": ["MECHANICAL"]},
      {"id": "impl-reason", "roles": ["implementer"], "classes": ["REASONING"]},
      {"id": "impl-agentic", "roles": ["implementer"], "execution_classes": ["AGENTIC"]},
      {"id": "reviewer-heavy", "roles": ["reviewer"]},
      {"id": "reviewer-light", "roles": ["reviewer"]},
      {"id": "reviser-1", "roles": ["reviser"]},
      {"id": "repair-1", "roles": ["repair"]},
      {"id": "planner-1", "roles": ["planner"]}
    ]
"""

private const val DEFAULT_STEPS = """
    [
      {"id": "S01", "title": "First step", "execution_class": "MECHANICAL"},
      {"id": "S02", "title": "Second step", "execution_class": "REASONING"}
    ]
"""

private const val DEFAULT_RUN_OPTIONS = """
    {
      "pipeline": {
        "semantic_revision_enabled": false,
        "max_check_repair_attempts": 0,
        "max_review_repair_cycles": 0
      },
      "profiles": {
        "mechanical_profile": "impl-mech",
        "reasoning_profile": "impl-reason",
        "final_reviewer_profile": "reviewer-heavy"
      }
    }
"""

private fun runDocument(
    status: String = "awaiting_plan_approval",
    approval: String = """{"recorded": false, "decision": null}""",
    steps: String = DEFAULT_STEPS,
    executionSelection: String = "null",
    runOptions: String = DEFAULT_RUN_OPTIONS,
): JsonObject = approvalDocument(
    """
    {
      "status": "$status",
      "approval": $approval,
      "implementation_bundle": {"steps": $steps},
      "execution_selection": $executionSelection,
      "run_options": $runOptions
    }
    """,
)

private fun profilesOf(
    profiles: String = PROFILE_CATALOGUE,
    defaults: String = "{}",
): ApprovalProfiles = approvalProfiles(approvalDocument("""{"profiles": $profiles, "defaults": $defaults}"""))

/** The gate and the selection of one run document, as one pass of the screen builds them. */
private fun formOf(
    document: JsonObject = runDocument(),
    profiles: ApprovalProfiles = profilesOf(),
    capabilities: JsonObject? = null,
): Pair<ApprovalGate, ApprovalSelection> {
    val gate = planApprovalGate(document, capabilities)!!
    return gate to defaultApprovalSelection(document, gate, profiles)
}

class RunApprovalTest {

    @Test
    fun `the block needs the awaiting status, no decision and an available capability`() {
        val awaiting = runDocument()

        assertTrue(planApprovalGate(awaiting, null) != null)
        assertTrue(planApprovalGate(awaiting, approvalDocument("""{"plan_approval": true}""")) != null)
        assertTrue(planApprovalGate(awaiting, approvalDocument("{}")) != null)
        assertNull(planApprovalGate(awaiting, approvalDocument("""{"plan_approval": false}""")))
        assertTrue(planApprovalGate(approvalDocument("""{"status": "awaiting_plan_approval"}"""), null) != null)

        assertTrue(planApprovalAvailable(null))
        assertTrue(planApprovalAvailable(approvalDocument("{}")))
        assertTrue(planApprovalAvailable(approvalDocument("""{"plan_approval": "no"}""")))
        assertFalse(planApprovalAvailable(approvalDocument("""{"plan_approval": false}""")))

        assertNull(planApprovalGate(runDocument(status = "implementing"), null))
        assertNull(planApprovalGate(approvalDocument("{}"), null))
        assertNull(
            planApprovalGate(
                runDocument(approval = """{"recorded": true, "decision": "APPROVE"}"""),
                null,
            ),
        )
        assertNull(planApprovalGate(runDocument(approval = """{"awaiting": false}"""), null))
        assertTrue(
            planApprovalGate(runDocument(approval = """{"recorded": false, "awaiting": true}"""), null) != null,
        )
    }

    @Test
    fun `the gate reads the run options of the run`() {
        fun gate(options: String) = planApprovalGate(runDocument(runOptions = options), null)!!

        val both = gate("""{"pipeline": {"semantic_revision_enabled": true, "max_check_repair_attempts": 2}}""")
        assertTrue(both.semanticRevisionEnabled)
        assertTrue(both.checkRepairEnabled)

        val cycles = gate("""{"pipeline": {"max_review_repair_cycles": "1"}}""")
        assertFalse(cycles.semanticRevisionEnabled)
        assertTrue(cycles.checkRepairEnabled)

        val none = gate("""{"profiles": {}}""")
        assertFalse(none.semanticRevisionEnabled)
        assertFalse(none.checkRepairEnabled)
    }

    @Test
    fun `the capability of the run comes before the one of the configuration`() {
        val config = approvalDocument("""{"capabilities": {"plan_approval": true}}""")

        assertTrue(approvalCapabilities(runDocument(), config)!!.get("plan_approval").asBoolean)
        assertNull(approvalCapabilities(runDocument(), approvalDocument("{}")))
        assertFalse(
            approvalCapabilities(
                approvalDocument("""{"capabilities": {"plan_approval": false}}"""),
                config,
            )!!.get("plan_approval").asBoolean,
        )
        assertTrue(
            approvalCapabilities(
                approvalDocument("""{"overview": {"capabilities": {"plan_approval": true}}}"""),
                approvalDocument("""{"capabilities": {"plan_approval": false}}"""),
            )!!.get("plan_approval").asBoolean,
        )
    }

    @Test
    fun `the steps come from the implementation bundle`() {
        val gate = planApprovalGate(runDocument(), null)!!

        assertEquals(listOf("S01", "S02"), gate.steps.map { it.id })
        assertEquals(listOf("First step", "Second step"), gate.steps.map { it.title })
        assertEquals(listOf("MECHANICAL", "REASONING"), gate.steps.map { it.executionClass })

        val shaped = planApprovalGate(
            runDocument(
                steps = """
                    [
                      {"id": "S01", "title": "Kept", "execution_class": "reasoning"},
                      {"title": "No id", "execution_class": "MECHANICAL"},
                      "not an object",
                      {"id": "S02"}
                    ]
                """,
            ),
            null,
        )!!

        assertEquals(listOf("S01", "S02"), shaped.steps.map { it.id })
        assertEquals("REASONING", shaped.steps[0].executionClass)
        assertEquals("MECHANICAL", shaped.steps[1].executionClass)
        assertNull(shaped.steps[1].title)
    }

    @Test
    fun `an implementer holds the role of its profile and the class of the step`() {
        val profiles = profilesOf()

        assertEquals(
            listOf("impl-any", "impl-mech"),
            profiles.implementers(ApprovalStep("S01", "First", "MECHANICAL")).map { it.id },
        )
        assertEquals(
            listOf("impl-any", "impl-reason"),
            profiles.implementers(ApprovalStep("S02", "Second", "REASONING")).map { it.id },
        )
        assertEquals(
            listOf("impl-any", "impl-agentic"),
            profiles.implementers(ApprovalStep("S03", "Third", "AGENTIC")).map { it.id },
        )
        assertEquals(listOf("reviewer-heavy", "reviewer-light"), profiles.withRole(REVIEWER_ROLE).map { it.id })
        assertEquals(listOf("planner-1"), profiles.withRole("planner").map { it.id })
    }

    @Test
    fun `a step default keeps the recorded implementer, then the routed profile`() {
        val profiles = profilesOf()

        assertEquals(
            mapOf("S01" to "impl-mech", "S02" to "impl-reason"),
            formOf(runDocument(), profiles).second.stepProfiles,
        )

        val recorded = runDocument(
            executionSelection = """{"steps": [{"step_id": "S01", "implementer": {"profile_id": "impl-any"}}]}""",
        )
        assertEquals(
            mapOf("S01" to "impl-any", "S02" to "impl-reason"),
            formOf(recorded, profiles).second.stepProfiles,
        )

        // A recorded profile the step cannot use is dropped for the routing, and a
        // run that routed nothing falls back to the first compatible profile.
        val unusable = runDocument(
            executionSelection = """{"steps": [{"step_id": "S01", "implementer": {"profile_id": "impl-reason"}}]}""",
        )
        assertEquals("impl-mech", formOf(unusable, profiles).second.stepProfiles["S01"])
        assertEquals(
            mapOf("S01" to "impl-any", "S02" to "impl-any"),
            formOf(runDocument(runOptions = """{"profiles": {}}"""), profiles).second.stepProfiles,
        )
    }

    @Test
    fun `a role default comes from the run options, then from the gateway`() {
        val profiles = profilesOf(
            defaults = """{"final_reviewer_profile": "reviewer-light", "semantic_reviser_profile": "reviser-1"}""",
        )
        val requested = runDocument(
            runOptions = """{"profiles": {"final_reviewer_profile": "reviewer-heavy"}}""",
        )
        val selection = formOf(requested, profiles).second

        assertEquals("reviewer-heavy", selection.finalReviewerProfile)
        assertEquals("reviser-1", selection.semanticReviserProfile)
        assertEquals("repair-1", selection.checkRepairProfile)

        val hostile = runDocument(
            runOptions = """{"profiles": {"final_reviewer_profile": "impl-any", "check_repair_profile": ""}}""",
        )
        val fallen = formOf(hostile, profiles).second

        assertEquals("reviewer-light", fallen.finalReviewerProfile)
        assertEquals("repair-1", fallen.checkRepairProfile)
    }

    @Test
    fun `the approve payload speaks the external contract and no local field name`() {
        val profiles = profilesOf()
        val document = runDocument(
            runOptions = """
                {
                  "pipeline": {"semantic_revision_enabled": true, "max_check_repair_attempts": 1},
                  "profiles": {
                    "mechanical_profile": "impl-mech",
                    "reasoning_profile": "impl-reason",
                    "final_reviewer_profile": "reviewer-heavy",
                    "semantic_reviser_profile": "reviser-1",
                    "check_repair_profile": "repair-1"
                  }
                }
            """,
        )
        val (gate, selection) = formOf(document, profiles)

        val payload = approvalPayload(selection, gate, profiles)!!

        assertEquals(
            listOf(
                "decision",
                "final_reviewer_profile",
                "semantic_reviser_profile",
                "check_repair_profile",
                "step_profiles",
            ),
            payload.keySet().toList(),
        )
        assertEquals("APPROVE", payload.get("decision").asString)
        assertEquals("reviewer-heavy", payload.get("final_reviewer_profile").asString)
        assertEquals("reviser-1", payload.get("semantic_reviser_profile").asString)
        assertEquals("repair-1", payload.get("check_repair_profile").asString)
        assertEquals(
            mapOf("S01" to "impl-mech", "S02" to "impl-reason"),
            payload.getAsJsonObject(STEP_PROFILES_FIELD)
                .entrySet()
                .associate { (stepId, profile) -> stepId to profile.asString },
        )
        assertFalse("the client never builds a local field name", payload.toString().contains("step_profile__"))
    }

    @Test
    fun `the approve payload names a role only when the run uses it`() {
        val profiles = profilesOf()
        val (gate, selection) = formOf(runDocument(), profiles)

        val payload = approvalPayload(selection, gate, profiles)!!

        assertFalse("semantic revision is off", payload.has("semantic_reviser_profile"))
        assertFalse("there is no correction budget", payload.has("check_repair_profile"))
        assertEquals("reviewer-heavy", payload.get("final_reviewer_profile").asString)
        assertEquals("impl-mech", payload.getAsJsonObject(STEP_PROFILES_FIELD).get("S01").asString)
    }

    @Test
    fun `an incomplete selection has no payload`() {
        val profiles = profilesOf()
        val (gate, complete) = formOf(runDocument(), profiles)

        assertTrue(approvalPayload(complete, gate, profiles) != null)
        assertNull(approvalPayload(complete.copy(finalReviewerProfile = ""), gate, profiles))
        assertNull(approvalPayload(complete.copy(finalReviewerProfile = "impl-any"), gate, profiles))
        assertNull(approvalPayload(complete.copy(stepProfiles = emptyMap()), gate, profiles))
        assertNull(approvalPayload(complete.copy(stepProfiles = mapOf("S01" to "impl-mech")), gate, profiles))
        assertNull(
            approvalPayload(
                complete.copy(stepProfiles = mapOf("S01" to "impl-mech", "S02" to "impl-mech")),
                gate,
                profiles,
            ),
        )

        val everyRole = gate.copy(semanticRevisionEnabled = true, checkRepairEnabled = true)
        assertNull(
            approvalPayload(
                complete.copy(semanticReviserProfile = "", checkRepairProfile = ""),
                everyRole,
                profiles,
            ),
        )
        assertTrue(approvalPayload(complete, everyRole, profiles) != null)
        assertNull(approvalPayload(complete, gate.copy(steps = emptyList()), profiles))

        // A step the plan does not hold is never named.
        val extra = complete.copy(stepProfiles = complete.stepProfiles + ("S03" to "impl-any"))
        assertFalse(approvalPayload(extra, gate, profiles)!!.getAsJsonObject(STEP_PROFILES_FIELD).has("S03"))
    }

    @Test
    fun `the rejection carries the decision alone`() {
        val payload = rejectionPayload()

        assertEquals("""{"decision":"REJECT"}""", payload.toString())
        assertEquals(1, payload.entrySet().size)
    }

    @Test
    fun `profiles of an unexpected shape are dropped`() {
        val profiles = approvalProfiles(
            approvalDocument(
                """
                {
                  "profiles": [
                    7,
                    {"id": 3},
                    {"id": "ok"},
                    {"id": "bad-roles", "roles": "reviewer"},
                    {"id": "ignored", "roles": ["  "]}
                  ],
                  "defaults": {"final_reviewer_profile": "ok", "half": 3}
                }
                """,
            ),
        )

        assertEquals(listOf("ok", "bad-roles", "ignored"), profiles.options.map { it.id })
        assertTrue(profiles.withRole(REVIEWER_ROLE).isEmpty())
        assertEquals(mapOf("final_reviewer_profile" to "ok"), profiles.defaults)

        assertEquals(emptyList<String>(), approvalProfiles(approvalDocument("{}")).options.map { it.id })
        assertEquals(
            emptyList<String>(),
            approvalProfiles(approvalDocument("""{"profiles": "nope", "defaults": 7}""")).options.map { it.id },
        )
    }
}
