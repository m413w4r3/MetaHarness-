package com.m413w4r3.metaharnessremote.ui.rundetail

import com.m413w4r3.metaharnessremote.api.ProgressResponse
import com.m413w4r3.metaharnessremote.ui.runs.RunCategory

/**
 * The progress events read so far.
 *
 * [offset] is the position the next request asks from: it starts at 0 and then
 * follows the gateway's `next_offset`, so an event is never read twice and the
 * whole history is never fetched again. The retained tail is bounded in memory;
 * the gateway keeps the history.
 */
data class RunProgress(
    val offset: Long = 0L,
    val events: List<String> = emptyList(),
) {

    /**
     * Fold one `GET /v1/runs/{runId}/progress` answer into the tail.
     *
     * Only an answer that moves the offset forward can carry events, so an
     * answer that stays put is ignored whole: nothing is appended twice.
     */
    fun plus(response: ProgressResponse): RunProgress {
        if (response.nextOffset <= offset) return this
        return RunProgress(
            offset = response.nextOffset,
            events = (events + response.events).takeLast(MAX_RETAINED_EVENTS),
        )
    }
}

/**
 * Delay before the next `GET /v1/runs/{runId}` pass, or null when the run is
 * terminal and the screen stops polling.
 *
 * An active run is polled every [ACTIVE_POLL_INTERVAL_MILLIS]; a run that waits
 * for the operator, and any status the board does not classify, every
 * [SLOW_POLL_INTERVAL_MILLIS] — nothing can change on its own in between.
 */
fun runPollIntervalMillis(status: String?): Long? = when (RunCategory.of(status)) {
    RunCategory.ACTIVE -> ACTIVE_POLL_INTERVAL_MILLIS
    RunCategory.ACTION_REQUIRED, RunCategory.OTHER -> SLOW_POLL_INTERVAL_MILLIS
    RunCategory.FAILED, RunCategory.COMPLETED -> null
}

/** Cadence of one pass over a run that is still moving. */
const val ACTIVE_POLL_INTERVAL_MILLIS = 2_000L

/** Cadence of one pass over a run that only the operator can move. */
const val SLOW_POLL_INTERVAL_MILLIS = 5_000L

/** Cadence after a failed pass: the gateway or the token may need fixing. */
const val RETRY_POLL_INTERVAL_MILLIS = SLOW_POLL_INTERVAL_MILLIS

/** How many progress events one screen keeps; older ones stay on the gateway. */
const val MAX_RETAINED_EVENTS = 1_000
