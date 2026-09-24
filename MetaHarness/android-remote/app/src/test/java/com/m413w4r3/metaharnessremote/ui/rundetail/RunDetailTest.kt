package com.m413w4r3.metaharnessremote.ui.rundetail

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

private fun document(json: String): JsonObject = JsonParser.parseString(json).asJsonObject

/** One detail document shaped like the gateway's, with a movable candidate map. */
private fun fullDocument(
    candidate: String = """{"001": {"commit_sha": "3333333cccc"}, "002": {"commit_sha": "4444444dddd"}}""",
    state: String = """
        {
          "status": "implementing",
          "cycle": 2,
          "candidate_commit_sha": "5555555eeee",
          "steps": [{"id": "S01", "profile_id": "impl-heavy"}],
          "planner": {
            "title": "Planner title",
            "steps": [
              {"id": "S01", "recommended_profile": "impl-fast"},
              {"id": "S02", "recommended_profile": "impl-heavy"}
            ]
          }
        }
    """,
): String = """
    {
      "run_id": "run-1",
      "status": "implementing",
      "updated_at": "2026-09-24T10:00:00Z",
      "plan_title": "Add the widget",
      "commit_sha": "1111111aaaa",
      "failure": null,
      "cycle": 2,
      "state": $state,
      "candidate": $candidate,
      "overview": {
        "current_label": "Implement S01",
        "next_label": "Validate the step",
        "execution_label": "auto · 2 steps",
        "publish_target": "origin/main",
        "publish_target_detail": "branch run/run-1",
        "resume": {"resumable": true, "label": "Retry S01"}
      },
      "plan": {"raw": "PLAN RAW", "contract": "PLAN CONTRACT"},
      "implementation_bundle": {
        "schema_version": 1,
        "steps": [
          {"id": "S01", "title": "First step", "execution_class": "MECHANICAL",
           "depends_on": "NONE", "contract_sha256": "x"},
          {"id": "S02", "title": "Second step", "execution_class": "REASONING",
           "depends_on": "S01", "contract_sha256": "y"}
        ]
      }
    }
"""

class RunDetailTest {

    @Test
    fun `every displayed field is read from the document`() {
        val view = runDetailView(document(fullDocument()), "opened-with")

        assertEquals("run-1", view.runId)
        assertEquals("implementing", view.status)
        assertEquals("Add the widget", view.planTitle)
        assertEquals("2026-09-24T10:00:00Z", view.updatedAt)
        assertNull(view.failure)
        assertEquals(2, view.cycle)
        assertEquals("4444444dddd", view.candidateSha)
        assertEquals("PLAN RAW", view.plan)
    }

    @Test
    fun `the overview mirrors the desktop labels`() {
        val view = runDetailView(document(fullDocument()), "run-1")

        assertEquals(
            listOf("CURRENT", "NEXT", "EXECUTION", "PUBLISH TARGET", "RESUME"),
            view.overview.map { it.label },
        )
        assertEquals("Implement S01", view.overview[0].value)
        assertEquals("Validate the step", view.overview[1].value)
        assertEquals("origin/main · branch run/run-1", view.overview[3].value)
        assertEquals("Retry S01", view.overview[4].value)
    }

    @Test
    fun `a selected implementer wins over the recommendation`() {
        val steps = runDetailView(document(fullDocument()), "run-1").steps

        assertEquals(listOf("S01", "S02"), steps.map { it.id })
        assertEquals(listOf("First step", "Second step"), steps.map { it.title })
        assertEquals(listOf("MECHANICAL", "REASONING"), steps.map { it.executionClass })

        val selected = steps[0]
        assertEquals("impl-heavy", selected.profileId)
        assertEquals("impl-fast", selected.recommendedProfile)
        assertEquals("impl-heavy", selected.profile)
        assertEquals("PROFILE", selected.profileLabel)

        val recommended = steps[1]
        assertNull(recommended.profileId)
        assertEquals("impl-heavy", recommended.profile)
        assertEquals("RECOMMENDED", recommended.profileLabel)
    }

    @Test
    fun `a failure document renders as reason and detail`() {
        val json = fullDocument()
            .replace(
                "\"failure\": null",
                "\"failure\": {\"reason\": \"CHECK_FAILED\", \"detail\": \"pytest -q\"}",
            )

        assertEquals("CHECK_FAILED — pytest -q", runDetailView(document(json), "run-1").failure)
    }

    @Test
    fun `an empty document leaves the screen empty`() {
        val view = runDetailView(document("{}"), "opened-with")

        assertEquals("opened-with", view.runId)
        assertNull(view.status)
        assertNull(view.planTitle)
        assertNull(view.updatedAt)
        assertNull(view.failure)
        assertNull(view.candidateSha)
        assertNull(view.cycle)
        assertNull(view.plan)
        assertTrue(view.overview.isEmpty())
        assertTrue(view.steps.isEmpty())
    }

    @Test
    fun `fields of another type are dropped instead of failing`() {
        val view = runDetailView(
            document(
                """
                {
                  "run_id": 7,
                  "status": 4,
                  "cycle": "not a number",
                  "state": [],
                  "overview": 7,
                  "plan": [],
                  "implementation_bundle": {"steps": "nope"},
                  "candidate": []
                }
                """,
            ),
            "opened-with",
        )

        assertEquals("opened-with", view.runId)
        assertNull(view.status)
        assertNull(view.cycle)
        assertNull(view.plan)
        assertTrue(view.overview.isEmpty())
        assertTrue(view.steps.isEmpty())
    }

    @Test
    fun `a candidate sha falls back to the state then to the commit`() {
        val withoutCandidates = fullDocument(candidate = "null")
        assertEquals(
            "5555555eeee",
            runDetailView(document(withoutCandidates), "run-1").candidateSha,
        )

        val withoutStateCandidate = withoutCandidates.replace("\"candidate_commit_sha\": \"5555555eeee\",", "")
        assertEquals(
            "1111111aaaa",
            runDetailView(document(withoutStateCandidate), "run-1").candidateSha,
        )
    }

    @Test
    fun `a cycle read as a string is kept`() {
        val json = fullDocument().replace("\"cycle\": 2,", "\"cycle\": \"3\",")

        assertEquals(3, runDetailView(document(json), "run-1").cycle)
    }

    @Test
    fun `steps without an id are dropped and keep their order`() {
        val json = fullDocument().replace(
            """{"id": "S01", "title": "First step", "execution_class": "MECHANICAL",
           "depends_on": "NONE", "contract_sha256": "x"},""",
            """{"title": "No id", "execution_class": "MECHANICAL"}, "not an object",""",
        )

        assertEquals(listOf("S02"), runDetailView(document(json), "run-1").steps.map { it.id })
    }

    @Test
    fun `an oversized plan is bounded`() {
        val plan = "x".repeat(MAX_FREE_TEXT_CHARS * 3)
        val json = fullDocument().replace("PLAN RAW", plan)

        val view = runDetailView(document(json), "run-1")

        assertEquals(MAX_FREE_TEXT_CHARS + 1, view.plan?.length)
        assertTrue(view.plan!!.endsWith("…"))
    }

    @Test
    fun `the plan falls back to the contract when the raw plan is missing`() {
        val json = fullDocument().replace("\"raw\": \"PLAN RAW\",", "")

        assertEquals("PLAN CONTRACT", runDetailView(document(json), "run-1").plan)
    }

    @Test
    fun `the plan title falls back to the task plan then to the planner`() {
        assertEquals(
            "Task plan title",
            runDetailView(document("""{"task_plan": {"title": "Task plan title"}}"""), "run-1").planTitle,
        )

        val withoutPlanTitle = fullDocument().replace("\"plan_title\": \"Add the widget\",", "")
        assertEquals("Planner title", runDetailView(document(withoutPlanTitle), "run-1").planTitle)

        val withoutAnyTitle = withoutPlanTitle.replace("\"title\": \"Planner title\",", "")
        assertNull(runDetailView(document(withoutAnyTitle), "run-1").planTitle)
    }
}
