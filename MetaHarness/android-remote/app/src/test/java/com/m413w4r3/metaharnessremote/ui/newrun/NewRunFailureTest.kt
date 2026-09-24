package com.m413w4r3.metaharnessremote.ui.newrun

import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import com.m413w4r3.metaharnessremote.api.MetaHarnessException
import com.m413w4r3.metaharnessremote.api.MetaHarnessTimeoutException
import org.junit.Assert.assertEquals
import org.junit.Test

class NewRunFailureTest {

    @Test
    fun `a timed out create says the run may exist and never invites a blind retry`() {
        assertEquals(
            "Response timed out.\n" +
                "The run may have been created.\n" +
                "Refresh the runs list before trying again.",
            newRunFailure(MetaHarnessTimeoutException("Response timed out")),
        )
    }

    @Test
    fun `the message of a gateway answer is shown as it is`() {
        val failure = MetaHarnessException(
            statusCode = 409,
            message = "Gateway replied HTTP 409",
            payload = JsonParser.parseString(
                """{"error":"run already exists","message":"run already exists"}"""
            ),
        )

        assertEquals("run already exists", newRunFailure(failure))
    }

    @Test
    fun `a failure without a gateway message falls back to its own`() {
        assertEquals(
            "Gateway replied HTTP 500",
            newRunFailure(MetaHarnessException(500, "Gateway replied HTTP 500")),
        )
        assertEquals(
            "Gateway replied HTTP 500",
            newRunFailure(MetaHarnessException(500, "Gateway replied HTTP 500", JsonPrimitive("boom"))),
        )
    }

    @Test
    fun `a failure without any message is named by its type`() {
        assertEquals("IllegalStateException", newRunFailure(IllegalStateException()))
    }
}
