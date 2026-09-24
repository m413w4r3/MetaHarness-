package com.m413w4r3.metaharnessremote.ui.runs

import com.google.gson.JsonParser
import com.m413w4r3.metaharnessremote.api.RunSummary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

private fun summary(
    runId: String? = "run-1",
    status: String? = null,
    planTitle: String? = null,
    commitSha: String? = null,
    updatedAt: String? = null,
    failure: String? = null,
): RunSummary = RunSummary(
    runId = runId,
    status = status,
    updatedAt = updatedAt,
    planTitle = planTitle,
    commitSha = commitSha,
    failure = failure?.let { JsonParser.parseString(it) },
)

class RunCardTest {

    @Test
    fun `sections follow the board order and drop the empty ones`() {
        val runs = listOf(
            summary(runId = "run-failed", status = "failed"),
            summary(runId = "run-blocked", status = "awaiting_plan_approval"),
            summary(runId = "run-active", status = "implementing"),
            summary(runId = "run-done", status = "committed"),
            summary(runId = "run-other", status = "waiting_human"),
        )

        val sections = runSections(runs)

        assertEquals(
            listOf(
                RunCategory.ACTION_REQUIRED,
                RunCategory.ACTIVE,
                RunCategory.FAILED,
                RunCategory.COMPLETED,
                RunCategory.OTHER,
            ),
            sections.map { it.category },
        )
        assertEquals(listOf("run-blocked"), sections[0].runs.map { it.runId })
        assertEquals(listOf("run-active"), sections[1].runs.map { it.runId })
    }

    @Test
    fun `a section keeps the order the gateway returned`() {
        val sections = runSections(
            listOf(
                summary(runId = "run-b", status = "revising"),
                summary(runId = "run-a", status = "reviewing"),
            )
        )

        assertEquals(listOf("run-b", "run-a"), sections.single().runs.map { it.runId })
    }

    @Test
    fun `summaries without a run id are ignored`() {
        val sections = runSections(
            listOf(
                summary(runId = null, status = "failed"),
                summary(runId = "  ", status = "failed"),
                summary(runId = " run-1 ", status = "failed"),
            )
        )

        assertEquals(listOf("run-1"), sections.single().runs.map { it.runId })
    }

    @Test
    fun `a card carries the displayed fields and a short commit`() {
        val card = runSections(
            listOf(
                summary(
                    runId = "run-1",
                    status = "committed",
                    planTitle = "Add comment",
                    commitSha = "0123456789abcdef0123456789abcdef01234567",
                    updatedAt = "2026-09-24T20:00:00Z",
                )
            )
        ).single().runs.single()

        assertEquals("run-1", card.runId)
        assertEquals("Add comment", card.planTitle)
        assertEquals("committed", card.status)
        assertEquals("2026-09-24T20:00:00Z", card.updatedAt)
        assertEquals("0123456", card.commitSha)
        assertEquals(RunCategory.COMPLETED, card.category)
        assertNull(card.failure)
    }

    @Test
    fun `missing fields stay empty and blank ones become null`() {
        val card = runSections(listOf(summary(status = "planning", planTitle = "  "))).single().runs.single()

        assertNull(card.planTitle)
        assertNull(card.commitSha)
        assertNull(card.updatedAt)
        assertNull(card.failure)
    }

    @Test
    fun `short commit sha keeps short ids and drops blanks`() {
        assertEquals("abcdefg", shortCommitSha("abcdefg"))
        assertEquals("0123456", shortCommitSha("0123456789abcdef"))
        assertEquals("abcdefg", shortCommitSha("  abcdefghi  "))
        assertNull(shortCommitSha(null))
        assertNull(shortCommitSha("   "))
    }

    @Test
    fun `failure summary mirrors the desktop rendering`() {
        assertEquals(
            "PUSH_FAILED — remote rejected the update",
            failureSummary(JsonParser.parseString("""{"reason":"PUSH_FAILED","detail":"remote rejected the update"}""")),
        )
        assertEquals("HUMAN_REQUIRED", failureSummary(JsonParser.parseString("""{"reason":"HUMAN_REQUIRED"}""")))
        assertEquals("plain text", failureSummary(JsonParser.parseString(""""plain text"""")))
        assertNull(failureSummary(null))
        assertNull(failureSummary(JsonParser.parseString("null")))
        assertNull(failureSummary(JsonParser.parseString("""{}""")))
    }

    @Test
    fun `failure summary is one bounded line`() {
        val multiline = failureSummary(JsonParser.parseString("""{"reason":"BOOM_WITH_A_LONG_CODE","detail":"line one\n   line two"}"""))
        assertEquals("BOOM_WITH_A_LONG_CODE — line one line two", multiline)

        val huge = "x".repeat(400)
        val bounded = failureSummary(JsonParser.parseString("""{"reason":"$huge"}"""))

        assertTrue(bounded!!.length <= 241)
        assertTrue(bounded.endsWith("…"))
    }
}
