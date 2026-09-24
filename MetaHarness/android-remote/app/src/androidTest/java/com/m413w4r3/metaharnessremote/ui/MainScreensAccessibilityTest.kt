package com.m413w4r3.metaharnessremote.ui

import androidx.compose.runtime.Composable
import androidx.compose.ui.test.assertHasClickAction
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.m413w4r3.metaharnessremote.data.ConnectionSession
import com.m413w4r3.metaharnessremote.data.SecureTokenStore
import com.m413w4r3.metaharnessremote.data.ServerSettingsStore
import com.m413w4r3.metaharnessremote.ui.newrun.NewRunScreen
import com.m413w4r3.metaharnessremote.ui.rundetail.RunDetailScreen
import com.m413w4r3.metaharnessremote.ui.runs.RunsScreen
import com.m413w4r3.metaharnessremote.ui.theme.MetaHarnessRemoteTheme
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MainScreensAccessibilityTest {
    @get:Rule
    val composeRule = createComposeRule()

    @Before
    fun clearCredentials() {
        clearStoredCredentials()
    }

    @After
    fun restoreCleanState() {
        clearStoredCredentials()
    }

    @Test
    fun runs_opensSettings_and_settingsControlsAreAccessible() {
        setContent { AppNavigation() }

        composeRule.onNodeWithText("MetaHarness Remote").assertExists()
        composeRule.onNodeWithText("Settings").assertHasClickAction().performClick()

        composeRule.onNodeWithContentDescription("Back").assertHasClickAction()
        composeRule.onNodeWithText("Server URL").assertExists()
        composeRule.onNodeWithText("Remote token").assertExists()

        val editableFields = composeRule.onAllNodes(hasSetTextAction())
        assertEquals("Settings should expose URL and token text fields", 2, editableFields.fetchSemanticsNodes().size)
        editableFields.get(1).performTextInput(TEST_TOKEN)
        composeRule.onNodeWithText(TEST_TOKEN).assertDoesNotExist()
        composeRule.onNodeWithContentDescription("Show token").assertHasClickAction().performClick()
        composeRule.onNodeWithText(TEST_TOKEN).assertExists()
        composeRule.onNodeWithContentDescription("Hide token").assertHasClickAction().performClick()
        composeRule.onNodeWithText(TEST_TOKEN).assertDoesNotExist()

        composeRule.onNodeWithText("Test Connection").assertIsEnabled().assertHasClickAction()
        composeRule.onNodeWithText("Forget credentials").assertHasClickAction().performClick()
        composeRule.onNodeWithText("Forget connection?").assertExists()
        composeRule.onNodeWithText("Cancel").performClick()

        composeRule.onNodeWithContentDescription("Back").performClick()
        composeRule.onNodeWithText("MetaHarness Remote").assertExists()
    }

    @Test
    fun runs_exposesEnabledNewRunAndIdentifiableRefresh() {
        // A non-empty but invalid URL makes the screen ready without contacting a server.
        ServerSettingsStore(InstrumentationRegistry.getInstrumentation().targetContext)
            .saveServerUrl("not a server URL")
        ConnectionSession.shared.remoteToken = TEST_TOKEN
        var openedNewRun = false
        setContent {
            RunsScreen(
                onOpenRun = {},
                onNewRun = { openedNewRun = true },
                onOpenSettings = {},
            )
        }

        composeRule.onNodeWithText("OFFLINE").assertExists()
        composeRule.onNodeWithText("Refresh").assertIsEnabled().assertHasClickAction()
        composeRule.onNodeWithText("+ NEW RUN").assertIsEnabled().assertHasClickAction().performClick()
        composeRule.waitForIdle()
        assertTrue("NEW RUN should invoke its action", openedNewRun)
    }

    @Test
    fun newRun_exposesBackSpecAndCreateRun() {
        setContent { NewRunScreen(onCreated = {}, onBack = {}) }

        composeRule.onNodeWithContentDescription("Back").assertHasClickAction()
        composeRule.onNodeWithText("SPEC").assertExists()
        composeRule.onNodeWithText("CREATE RUN").assertExists()
    }

    @Test
    fun runDetail_exposesBackAndTitle() {
        setContent { RunDetailScreen(runId = "ui-test-run", onBack = {}) }

        composeRule.onNodeWithContentDescription("Back").assertHasClickAction()
        composeRule.onNodeWithText("Run Detail").assertExists()
    }

    private fun setContent(content: @Composable () -> Unit) {
        composeRule.setContent {
            MetaHarnessRemoteTheme(content = content)
        }
    }

    private fun clearStoredCredentials() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        ServerSettingsStore(context).saveServerUrl("")
        SecureTokenStore(context).clearRemoteToken()
        ConnectionSession.shared.remoteToken = ""
    }

    private companion object {
        const val TEST_TOKEN = "instrumented-ui-test-token"
    }
}
