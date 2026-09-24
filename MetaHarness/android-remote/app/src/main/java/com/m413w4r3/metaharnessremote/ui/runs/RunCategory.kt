package com.m413w4r3.metaharnessremote.ui.runs

/**
 * Board section of one run, in the order the Runs screen renders them.
 *
 * The gateway reports `state.status` as a lowercase token; anything the
 * MetaHarness pipeline adds later falls into [OTHER] until it is listed here.
 */
enum class RunCategory(val label: String) {
    ACTION_REQUIRED("ACTION REQUIRED"),
    ACTIVE("ACTIVE"),
    FAILED("FAILED"),
    COMPLETED("COMPLETED"),
    OTHER("OTHER"),
    ;

    companion object {
        /** The run stopped and waits for the operator before it can move again. */
        private val ACTION_REQUIRED_STATUSES = setOf(
            "blocked",
            "awaiting_plan_approval",
            "waiting_scope_approval",
            "plan_rejected",
        )

        private val ACTIVE_STATUSES = setOf(
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

        private val COMPLETED_STATUSES = setOf("published", "committed")

        private val FAILED_STATUSES = setOf("failed", "interrupted")

        /** Classify one `state.status`; a missing or unlisted status is [OTHER]. */
        fun of(status: String?): RunCategory {
            val value = status?.trim()?.lowercase().orEmpty()
            return when {
                value in ACTION_REQUIRED_STATUSES -> ACTION_REQUIRED
                value in ACTIVE_STATUSES -> ACTIVE
                value in FAILED_STATUSES -> FAILED
                value in COMPLETED_STATUSES -> COMPLETED
                else -> OTHER
            }
        }
    }
}
