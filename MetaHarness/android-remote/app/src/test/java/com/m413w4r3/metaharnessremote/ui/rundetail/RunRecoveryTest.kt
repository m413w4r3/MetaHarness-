package com.m413w4r3.metaharnessremote.ui.rundetail

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.m413w4r3.metaharnessremote.api.MetaHarnessException
import com.m413w4r3.metaharnessremote.api.MetaHarnessTimeoutException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

private fun document(json: String): JsonObject = JsonParser.parseString(json).asJsonObject

private const val SCOPE_DELTA = """
    {
      "schema_version": 1,
      "added_paths": ["src/new.py", "tests/test_new.py"],
      "unchanged_paths": ["src/kept.py"]
    }
"""

private fun scopeDocument(
    status: String = "waiting_scope_approval",
    delta: String = SCOPE_DELTA,
    approval: String = """{"recorded": false, "decision": null}""",
    state: String = "{}",
): JsonObject = document(
    """
    {
      "status": "$status",
      "scope_delta": $delta,
      "scope_approval": $approval,
      "state": $state
    }
    """,
)

private fun resumeDocument(
    resume: String = """{"resumable": true, "label": "Retry S01"}""",
): JsonObject = document("""{"status": "failed", "overview": {"resume": $resume}}""")

/** A document whose overview carries no resume field at all. */
private fun documentWithoutResume(): JsonObject =
    document("""{"status": "implementing", "overview": {"current_label": "Implement"}}""")

class RunRecoveryTest {

    @Test
    fun `the scope block needs the waiting status, no decision and a non-empty delta`() {
        assertTrue(awaitingScopeDecision(scopeDocument()))
        assertTrue(scopeApprovalGate(scopeDocument(), null) != null)
        assertTrue(scopeApprovalGate(scopeDocument(), document("""{"scope_approval": true}""")) != null)

        assertFalse(awaitingScopeDecision(scopeDocument(status = "implementing")))
        assertNull(scopeApprovalGate(scopeDocument(status = "implementing"), null))
        assertNull(
            scopeApprovalGate(
                scopeDocument(approval = """{"recorded": true, "decision": "APPROVE"}"""),
                null,
            ),
        )
        assertNull(scopeApprovalGate(scopeDocument(approval = """{"awaiting": false}"""), null))
        assertTrue(
            scopeApprovalGate(scopeDocument(approval = """{"recorded": false, "awaiting": true}"""), null) != null,
        )
    }

    @Test
    fun `a missing or empty scope delta hides the block`() {
        assertNull(scopeApprovalGate(scopeDocument(delta = "null"), null))
        assertNull(scopeApprovalGate(scopeDocument(delta = "{}"), null))
        assertNull(scopeApprovalGate(document("""{"status": "waiting_scope_approval"}"""), null))
    }

    @Test
    fun `only an explicit scope capability hides the block`() {
        assertNull(scopeApprovalGate(scopeDocument(), document("""{"scope_approval": false}""")))
        assertTrue(scopeApprovalGate(scopeDocument(), document("{}")) != null)
        assertTrue(scopeApprovalGate(scopeDocument(), document("""{"scope_approval": "no"}""")) != null)
    }

    @Test
    fun `the added paths come from the delta, else from the recorded one`() {
        val gate = scopeApprovalGate(scopeDocument(), null)!!

        assertEquals(listOf("src/new.py", "tests/test_new.py"), gate.addedPaths)

        val recorded = scopeDocument(
            delta = "null",
            state = """{"scope_delta": {"added_paths": ["src/beside_the_step.py"]}}""",
        )
        assertEquals(
            listOf("src/beside_the_step.py"),
            scopeApprovalGate(recorded, null)!!.addedPaths,
        )

        val empty = scopeDocument(delta = "null", state = """{"scope_delta": {}}""")
        assertNull(scopeApprovalGate(empty, null))
    }

    @Test
    fun `paths of another type are dropped`() {
        val gate = scopeApprovalGate(
            scopeDocument(delta = """{"added_paths": ["ok.py", 7, "  ", "second.py"]}"""),
            null,
        )!!

        assertEquals(listOf("ok.py", "second.py"), gate.addedPaths)
        assertEquals(
            emptyList<String>(),
            scopeApprovalGate(scopeDocument(delta = """{"added_paths": "nope"}"""), null)!!.addedPaths,
        )
    }

    @Test
    fun `the resume block needs a resumable checkpoint`() {
        val gate = resumeGate(resumeDocument(), null)!!

        assertEquals("Retry S01", gate.label)
        assertTrue(resumableRun(resumeDocument()))
        assertNull(resumeGate(resumeDocument(resume = """{"resumable": false}"""), null))
        assertNull(resumeGate(resumeDocument(resume = """{"label": "Retry S01"}"""), null))
        assertNull(resumeGate(document("{}"), null))
        assertNull(resumeGate(documentWithoutResume(), null))
        assertNull(resumeGate(resumeDocument(resume = """{"resumable": "yes"}"""), null))
        assertNull(resumeGate(resumeDocument(resume = """{"resumable": true}"""), null)?.label)
    }

    @Test
    fun `only an explicit resume capability hides the resume block`() {
        assertNull(resumeGate(resumeDocument(), document("""{"resume": false}""")))
        assertTrue(resumeGate(resumeDocument(), document("{}")) != null)
        assertTrue(resumeGate(resumeDocument(), document("""{"resume": "no"}""")) != null)
    }

    @Test
    fun `a scope decision body carries the decision alone`() {
        assertEquals("""{"decision":"APPROVE"}""", scopeDecisionPayload("APPROVE").toString())
        assertEquals("""{"decision":"REJECT"}""", scopeDecisionPayload("REJECT").toString())
        assertEquals(1, scopeDecisionPayload("REJECT").entrySet().size)
    }

    @Test
    fun `a timed out resume says it may have been accepted and is not retried`() {
        assertEquals(
            "The resume request may have been accepted.\nRefresh before retrying.",
            resumeFailure(MetaHarnessTimeoutException("Response timed out")),
        )

        val refused = MetaHarnessException(
            statusCode = 409,
            message = "Gateway replied HTTP 409",
            payload = JsonParser.parseString("""{"message": "run is not resumable"}"""),
        )
        assertEquals("Refused: run is not resumable", resumeFailure(refused))
        assertEquals("Request failed", resumeFailure(MetaHarnessException(null, "Request failed")))
    }
}
