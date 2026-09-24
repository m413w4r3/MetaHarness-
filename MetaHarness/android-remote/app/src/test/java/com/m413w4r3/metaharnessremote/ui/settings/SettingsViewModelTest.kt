package com.m413w4r3.metaharnessremote.ui.settings

import com.m413w4r3.metaharnessremote.data.ConnectionSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SettingsViewModelTest {

    @Test
    fun `editing credentials does not persist them or update the session`() {
        val settings = FakeSettingsStore("https://old.example.ts.net")
        val tokens = FakeTokenStore("old-token")
        val session = ConnectionSession().apply { remoteToken = "session-token" }
        val viewModel = SettingsViewModel(settings, tokens, session = session)

        viewModel.onServerUrlChange("new.example.ts.net")
        viewModel.onRemoteTokenChange("new-token")

        assertEquals("https://old.example.ts.net", settings.serverUrl)
        assertEquals("old-token", tokens.token)
        assertEquals(0, tokens.saveCount)
        assertEquals("session-token", session.remoteToken)
    }

    @Test
    fun `save persists normalized url and token`() {
        val settings = FakeSettingsStore("https://old.example.ts.net")
        val tokens = FakeTokenStore("old-token")
        val viewModel = SettingsViewModel(settings, tokens, session = ConnectionSession())
        viewModel.onServerUrlChange("  new.example.ts.net/  ")
        viewModel.onRemoteTokenChange("new-token")

        assertTrue(viewModel.saveCredentials())

        assertEquals("https://new.example.ts.net", settings.serverUrl)
        assertEquals("new-token", tokens.token)
        assertEquals(1, tokens.saveCount)
    }

    @Test
    fun `forget clears saved credentials`() {
        val settings = FakeSettingsStore("https://old.example.ts.net")
        val tokens = FakeTokenStore("old-token")
        val session = ConnectionSession().apply { remoteToken = "old-token" }
        val viewModel = SettingsViewModel(settings, tokens, session = session)

        viewModel.forgetCredentials()

        assertEquals("", settings.serverUrl)
        assertEquals(null, tokens.token)
        assertEquals("", viewModel.serverUrl)
        assertEquals("", viewModel.remoteToken)
        assertEquals("", session.remoteToken)
    }

    @Test
    fun `session is updated only when valid credentials are saved`() {
        val session = ConnectionSession().apply { remoteToken = "previous-token" }
        val viewModel = SettingsViewModel(
            FakeSettingsStore("https://old.example.ts.net"),
            FakeTokenStore("old-token"),
            session = session,
        )
        viewModel.onRemoteTokenChange("replacement-token")

        assertEquals("previous-token", session.remoteToken)
        assertTrue(viewModel.saveCredentials())
        assertEquals("replacement-token", session.remoteToken)
    }

    @Test
    fun `an empty replacement token cannot erase the saved token`() {
        val settings = FakeSettingsStore("https://old.example.ts.net")
        val tokens = FakeTokenStore("saved-token")
        val viewModel = SettingsViewModel(settings, tokens, session = ConnectionSession())
        viewModel.onRemoteTokenChange("  ")

        assertFalse(viewModel.saveCredentials())

        assertEquals("saved-token", tokens.token)
        assertEquals(0, tokens.saveCount)
    }

    private class FakeSettingsStore(initialUrl: String) : SettingsStorePort {
        var serverUrl = initialUrl
            private set

        override fun loadServerUrl(): String = serverUrl

        override fun saveServerUrl(serverUrl: String) {
            this.serverUrl = serverUrl
        }
    }

    private class FakeTokenStore(initialToken: String?) : TokenStorePort {
        var token = initialToken
            private set
        var saveCount = 0
            private set

        override fun loadRemoteToken(): String? = token

        override fun saveRemoteToken(token: String) {
            this.token = token
            saveCount += 1
        }

        override fun clearRemoteToken() {
            token = null
        }
    }
}
