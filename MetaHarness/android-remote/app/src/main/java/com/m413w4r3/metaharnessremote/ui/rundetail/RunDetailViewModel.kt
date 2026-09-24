package com.m413w4r3.metaharnessremote.ui.rundetail

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.m413w4r3.metaharnessremote.api.MetaHarnessApi
import com.m413w4r3.metaharnessremote.data.ConnectionSession
import com.m413w4r3.metaharnessremote.data.ServerSettingsStore
import com.m413w4r3.metaharnessremote.network.GatewayClient
import com.m413w4r3.metaharnessremote.network.GatewayUrls
import kotlinx.coroutines.CancellationException
import okhttp3.OkHttpClient

/** Everything the Run Detail screen renders. */
data class RunDetailUiState(
    /** The id the screen was opened with; shown before the first reply. */
    val runId: String = "",
    /** The last detail read, kept when a later pass fails. */
    val detail: RunDetailView? = null,
    /** The progress tail read so far. */
    val progress: RunProgress = RunProgress(),
    /** One `GET /v1/runs/{runId}` pass has completed, whatever its outcome. */
    val hasLoaded: Boolean = false,
    /** Failure of the last pass, shown next to the last good detail. */
    val error: String? = null,
    /** The run is terminal: polling stopped until the screen is entered again. */
    val pollingStopped: Boolean = false,
)

/**
 * Read-only detail of one run.
 *
 * A pass reads `GET /v1/runs/{runId}` and then
 * `GET /v1/runs/{runId}/progress?offset=N` with the offset kept here, and the
 * caller owns the delay before the next pass: [refreshOnce] returns it, so a
 * screen that is no longer visible stops polling by cancelling its loop.
 * Nothing is stored on the device, and nothing here can change the run.
 */
class RunDetailViewModel(
    private val settingsStore: ServerSettingsStore,
    val runId: String,
    private val session: ConnectionSession = ConnectionSession.shared,
    private val httpClient: OkHttpClient = GatewayClient.defaultHttpClient(),
) : ViewModel() {

    var state by mutableStateOf(RunDetailUiState(runId = runId))
        private set

    /**
     * One detail and progress pass.
     *
     * Returns the delay to wait before the next pass, or null when the run is
     * terminal and polling must stop. A pass in which either read failed keeps
     * the last good detail, reports the error and retries at the slow cadence:
     * a terminal run whose events were not read yet is asked again, and a run
     * that was active when the gateway went away returns to the active cadence
     * as soon as an answer comes back.
     */
    suspend fun refreshOnce(): Long? {
        val baseUrl = when (val normalized = GatewayUrls.normalize(settingsStore.loadServerUrl())) {
            is GatewayUrls.BaseUrl.Invalid -> {
                state = state.copy(hasLoaded = true, error = normalized.reason)
                return RETRY_POLL_INTERVAL_MILLIS
            }

            is GatewayUrls.BaseUrl.Valid -> normalized.value
        }
        val token = session.remoteToken
        if (token.isBlank()) {
            state = state.copy(hasLoaded = true, error = MISSING_TOKEN)
            return RETRY_POLL_INTERVAL_MILLIS
        }

        val api = MetaHarnessApi(baseUrl, token, httpClient)
        val detail = attempt { api.getRun(runId) }
        val progress = attempt { api.progress(runId, state.progress.offset) }

        val read = detail.getOrNull()?.let { runDetailView(it.raw, runId) }
        val shown = read ?: state.detail
        val complete = detail.isSuccess && progress.isSuccess
        val interval = if (complete) runPollIntervalMillis(shown?.status) else RETRY_POLL_INTERVAL_MILLIS

        state = state.copy(
            detail = shown,
            progress = progress.getOrNull()?.let(state.progress::plus) ?: state.progress,
            hasLoaded = true,
            error = listOfNotNull(detail.exceptionOrNull(), progress.exceptionOrNull())
                .firstOrNull()
                ?.let(::describe),
            pollingStopped = interval == null,
        )
        return interval
    }

    /** [CancellationException] belongs to the scope, not to the gateway. */
    private suspend fun <T> attempt(block: suspend () -> T): Result<T> =
        try {
            Result.success(block())
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Exception) {
            Result.failure(failure)
        }

    /** A one-line reason for a failed exchange; the API never puts the token in one. */
    private fun describe(failure: Throwable): String =
        failure.message?.trim()?.takeIf { it.isNotEmpty() } ?: failure.javaClass.simpleName

    companion object {
        private const val MISSING_TOKEN = "Remote token required: set it in Settings"

        fun factory(context: Context, runId: String): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                RunDetailViewModel(ServerSettingsStore(context.applicationContext), runId)
            }
        }
    }
}
