package com.m413w4r3.metaharnessremote.ui.rundetail

import com.google.gson.JsonParser
import com.m413w4r3.metaharnessremote.api.MetaHarnessException
import com.m413w4r3.metaharnessremote.api.MetaHarnessTimeoutException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

private fun gatewayError(status: Int, body: String): MetaHarnessException =
    MetaHarnessException(
        statusCode = status,
        message = "Gateway replied HTTP $status",
        payload = JsonParser.parseString(body),
    )

class ApprovalFailureTest {

    @Test
    fun `a timeout says the outcome is unknown`() {
        val message = approvalFailure(MetaHarnessTimeoutException("timed out"))

        assertEquals(APPROVAL_TIMEOUT_MESSAGE, message)
        assertTrue(message.contains("may have been recorded"))
    }

    @Test
    fun `a conflict shows the words of the gateway`() {
        val failure = gatewayError(
            409,
            """{"error":"conflict","message":"run is not awaiting plan approval"}""",
        )

        assertEquals("Refused: run is not awaiting plan approval", approvalFailure(failure))
    }

    @Test
    fun `a conflict without a body still says what happened`() {
        val failure = MetaHarnessException(statusCode = 409, message = "Gateway replied HTTP 409")

        assertEquals("Refused: Gateway replied HTTP 409", approvalFailure(failure))
    }

    @Test
    fun `another refusal shows the message of the gateway`() {
        val failure = gatewayError(400, """{"error":"invalid","message":"selected profile is invalid"}""")

        assertEquals("selected profile is invalid", approvalFailure(failure))
    }

    @Test
    fun `a transport failure shows its own description`() {
        assertEquals("connect failed", approvalFailure(IllegalStateException("connect failed")))
        assertEquals("IllegalStateException", approvalFailure(IllegalStateException()))
    }
}
