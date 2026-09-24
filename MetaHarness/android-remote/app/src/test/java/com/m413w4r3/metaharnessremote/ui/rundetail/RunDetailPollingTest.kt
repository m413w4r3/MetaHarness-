package com.m413w4r3.metaharnessremote.ui.rundetail

import com.m413w4r3.metaharnessremote.api.ProgressResponse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RunDetailPollingTest {

    @Test
    fun `an active run is polled every two seconds`() {
        listOf("created", "planning", "implementing", "reviewing", "publishing").forEach { status ->
            assertEquals(status, ACTIVE_POLL_INTERVAL_MILLIS, runPollIntervalMillis(status))
        }
    }

    @Test
    fun `a run waiting for the operator is polled slowly`() {
        listOf("blocked", "awaiting_plan_approval", "waiting_scope_approval", "plan_rejected")
            .forEach { status ->
                val interval = runPollIntervalMillis(status)
                assertTrue(status, interval != null && interval >= SLOW_POLL_INTERVAL_MILLIS)
            }
    }

    @Test
    fun `a terminal run stops polling`() {
        listOf("published", "committed", "failed", "interrupted").forEach { status ->
            assertNull(status, runPollIntervalMillis(status))
        }
    }

    @Test
    fun `an unclassified status is polled slowly`() {
        listOf(null, "waiting_human", "waiting_remote", "something-new").forEach { status ->
            val interval = runPollIntervalMillis(status)
            assertTrue("$status", interval != null && interval >= SLOW_POLL_INTERVAL_MILLIS)
        }
    }

    @Test
    fun `progress starts at zero and follows the gateway offset`() {
        assertEquals(0L, RunProgress().offset)

        val advanced = RunProgress().plus(ProgressResponse(nextOffset = 120, events = listOf("first")))

        assertEquals(120L, advanced.offset)
        assertEquals(listOf("first"), advanced.events)
    }

    @Test
    fun `only the events of an advancing answer are appended`() {
        val start = RunProgress(offset = 120, events = listOf("first", "second"))

        val advanced = start.plus(ProgressResponse(nextOffset = 300, events = listOf("third")))

        assertEquals(300L, advanced.offset)
        assertEquals(listOf("first", "second", "third"), advanced.events)
    }

    @Test
    fun `an answer that does not advance adds nothing`() {
        val start = RunProgress(offset = 120, events = listOf("first"))

        val same = start.plus(ProgressResponse(nextOffset = 120, events = listOf("again")))
        val rewound = start.plus(ProgressResponse(nextOffset = 4, events = listOf("again")))

        assertEquals(start, same)
        assertEquals(start, rewound)
    }

    @Test
    fun `an offset that moves without events is kept`() {
        val skipped = RunProgress(offset = 0).plus(ProgressResponse(nextOffset = 512, events = emptyList()))

        assertEquals(512L, skipped.offset)
        assertTrue(skipped.events.isEmpty())
    }

    @Test
    fun `the retained tail is bounded`() {
        var progress = RunProgress()
        repeat(MAX_RETAINED_EVENTS + 50) { index ->
            progress = progress.plus(
                ProgressResponse(nextOffset = index + 1L, events = listOf("event $index")),
            )
        }

        assertEquals(MAX_RETAINED_EVENTS, progress.events.size)
        assertEquals("event 1049", progress.events.last())
        assertEquals(MAX_RETAINED_EVENTS + 50L, progress.offset)
    }
}
