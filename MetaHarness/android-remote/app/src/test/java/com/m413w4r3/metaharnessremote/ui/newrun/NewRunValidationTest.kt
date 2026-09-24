package com.m413w4r3.metaharnessremote.ui.newrun

import com.m413w4r3.metaharnessremote.api.MetaHarnessApi
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NewRunValidationTest {

    @Test
    fun `an empty run id asks the gateway to generate it`() {
        assertEquals(
            NewRunValidation.Valid(spec = "Do it.", runId = null),
            validateNewRun("Do it.", ""),
        )
        assertEquals(
            NewRunValidation.Valid(spec = "Do it.", runId = null),
            validateNewRun("Do it.", "   "),
        )
    }

    @Test
    fun `the run id is trimmed and the spec stays as typed`() {
        assertEquals(
            NewRunValidation.Valid(spec = "  Do it.\n", runId = "run-1"),
            validateNewRun("  Do it.\n", "  run-1  "),
        )
    }

    @Test
    fun `every run id of the charset is accepted`() {
        listOf("r", "9", "run-1", "RUN_2024.09", "a.b-c_d").forEach { runId ->
            assertEquals(
                NewRunValidation.Valid(spec = "Do it.", runId = runId),
                validateNewRun("Do it.", runId),
            )
        }
    }

    @Test
    fun `a spec without text is refused`() {
        listOf("", "   ", "\n\t ").forEach { spec ->
            val refused = refused(validateNewRun(spec, ""))
            assertNotNull(refused.specError)
            assertNull(refused.runIdError)
        }
    }

    @Test
    fun `a spec of exactly the maximum size is accepted, one byte more is not`() {
        accepted(validateNewRun("a".repeat(MetaHarnessApi.MAX_SPEC_BYTES), ""))

        val refused = refused(validateNewRun("a".repeat(MetaHarnessApi.MAX_SPEC_BYTES + 1), ""))
        assertNotNull(refused.specError)
    }

    @Test
    fun `the limit counts utf8 bytes, not characters`() {
        // Two bytes per character: half the characters reach the limit.
        accepted(validateNewRun("é".repeat(MetaHarnessApi.MAX_SPEC_BYTES / 2), ""))

        val refused = refused(validateNewRun("é".repeat(MetaHarnessApi.MAX_SPEC_BYTES / 2 + 1), ""))
        assertNotNull(refused.specError)
    }

    @Test
    fun `a run id outside the charset is refused`() {
        listOf("-run", ".run", "_run", "run/1", "run 1", "run:1", "run?x", "runé", "..").forEach { runId ->
            val refused = refused(validateNewRun("Do it.", runId))
            assertNotNull("$runId must be refused", refused.runIdError)
            assertNull(refused.specError)
        }
    }

    @Test
    fun `the charset is the whole client rule, the gateway owns the rest`() {
        // A run id the regex accepts is sent: the Git-ref rules of the local
        // server (`..`, a trailing dot, `.lock`) are answered by the gateway.
        listOf("run.", "a..b", "run.lock").forEach { runId ->
            assertEquals(
                NewRunValidation.Valid(spec = "Do it.", runId = runId),
                validateNewRun("Do it.", runId),
            )
        }
    }

    @Test
    fun `both fields are reported at once`() {
        val refused = refused(validateNewRun("  ", "bad id"))

        assertNotNull(refused.specError)
        assertNotNull(refused.runIdError)
    }

    @Test
    fun `an invalid run id is refused even when the spec is too large`() {
        val refused = refused(validateNewRun("a".repeat(MetaHarnessApi.MAX_SPEC_BYTES + 1), "-run"))

        assertTrue(refused.specError != null && refused.runIdError != null)
    }

    private fun accepted(result: NewRunValidation): NewRunValidation.Valid =
        result as? NewRunValidation.Valid
            ?: throw AssertionError("expected an accepted form, got $result")

    private fun refused(result: NewRunValidation): NewRunValidation.Invalid =
        result as? NewRunValidation.Invalid
            ?: throw AssertionError("expected a refused form, got $result")
}
