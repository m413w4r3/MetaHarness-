package com.m413w4r3.metaharnessremote.ui.runs

import org.junit.Assert.assertEquals
import org.junit.Test

class RunCategoryTest {

    @Test
    fun `every active status lands in ACTIVE`() {
        val active = listOf(
            "created",
            "planning",
            "worktree_ready",
            "preparing",
            "implementing",
            "contract_repairing",
            "validating",
            "pre_revision_validating",
            "revising",
            "revalidating",
            "reviewing",
            "approved",
            "publishing",
        )

        assertEquals(active.size, active.count { RunCategory.of(it) == RunCategory.ACTIVE })
    }

    @Test
    fun `every operator-bound status lands in ACTION_REQUIRED`() {
        val waiting = listOf(
            "blocked",
            "awaiting_plan_approval",
            "waiting_scope_approval",
            "plan_rejected",
        )

        assertEquals(waiting.size, waiting.count { RunCategory.of(it) == RunCategory.ACTION_REQUIRED })
    }

    @Test
    fun `terminal statuses split between COMPLETED and FAILED`() {
        assertEquals(RunCategory.COMPLETED, RunCategory.of("published"))
        assertEquals(RunCategory.COMPLETED, RunCategory.of("committed"))
        assertEquals(RunCategory.FAILED, RunCategory.of("failed"))
        assertEquals(RunCategory.FAILED, RunCategory.of("interrupted"))
    }

    @Test
    fun `unlisted and missing statuses land in OTHER`() {
        assertEquals(RunCategory.OTHER, RunCategory.of("waiting_human"))
        assertEquals(RunCategory.OTHER, RunCategory.of("waiting_remote"))
        assertEquals(RunCategory.OTHER, RunCategory.of("something_new"))
        assertEquals(RunCategory.OTHER, RunCategory.of(""))
        assertEquals(RunCategory.OTHER, RunCategory.of("   "))
        assertEquals(RunCategory.OTHER, RunCategory.of(null))
    }

    @Test
    fun `classification ignores case and surrounding whitespace`() {
        assertEquals(RunCategory.ACTION_REQUIRED, RunCategory.of("  BLOcked "))
        assertEquals(RunCategory.ACTIVE, RunCategory.of("Implementing"))
        assertEquals(RunCategory.COMPLETED, RunCategory.of("PUBLISHED\n"))
    }

    @Test
    fun `labels and declaration order drive the board`() {
        assertEquals(
            listOf("ACTION REQUIRED", "ACTIVE", "FAILED", "COMPLETED", "OTHER"),
            RunCategory.entries.map { it.label },
        )
    }
}
