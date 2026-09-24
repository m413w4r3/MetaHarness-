package com.m413w4r3.metaharnessremote.ui.runs

import org.junit.Assert.assertEquals
import org.junit.Test

class RunsViewModelSetupTest {

    @Test
    fun `blank server url requires setup`() {
        assertEquals(
            SetupState.MISSING_SERVER_URL,
            RunsViewModel.resolveSetupState("   ", "remote-token"),
        )
    }

    @Test
    fun `blank remote token requires setup`() {
        assertEquals(
            SetupState.MISSING_REMOTE_TOKEN,
            RunsViewModel.resolveSetupState("https://gateway.example", "  "),
        )
    }

    @Test
    fun `server url and remote token mark setup ready`() {
        assertEquals(
            SetupState.READY,
            RunsViewModel.resolveSetupState("https://gateway.example", "remote-token"),
        )
    }
}
