package com.m413w4r3.metaharnessremote.ui.runs

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.m413w4r3.metaharnessremote.api.MetaHarnessApi
import com.m413w4r3.metaharnessremote.data.ConnectionSession
import com.m413w4r3.metaharnessremote.data.ServerSettingsStore
import com.m413w4r3.metaharnessremote.network.GatewayClient
import com.m413w4r3.metaharnessremote.network.GatewayUrls
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient

/** Reachability of the gateway, as last reported by `GET /v1/health`. */
sealed interface GatewayConnection {
    /** No exchange has completed yet. */
    data object Unknown : GatewayConnection
    data object Connected : GatewayConnection
    data class Unreachable(val message: String) : GatewayConnection
}

/** Everything the Runs screen renders. */
data class RunsUiState(
    val connection: GatewayConnection = GatewayConnection.Unknown,
    val sections: List<RunSection> = emptyList(),
    /** The gateway has been queried at least once, whatever the outcome. */
    val hasLoaded: Boolean = false,
    /** One exchange is in flight. */
    val loading: Boolean = false,
    /** Failure of the last `GET /v1/runs`, shown next to the previous list. */
    val error: String? = null,
)

/**
 * Board state of the Runs screen.
 *
 * Each pass asks `GET /v1/health` and `GET /v1/runs` once and keeps the last
 * good list when a pass fails. Nothing is stored: the desktop stays the source
 * of authority, so every screen entry reads the gateway again.
 */
class RunsViewModel(
    private val settingsStore: ServerSettingsStore,
    private val session: ConnectionSession = ConnectionSession.shared,
    private val httpClient: OkHttpClient = GatewayClient.defaultHttpClient(),
) : ViewModel() {

    var state by mutableStateOf(RunsUiState())
        private set

    private var inFlight: Job? = null

    /**
     * One health and runs pass.
     *
     * A pass already in flight absorbs this call, so the poll, the Refresh
     * button and the pull gesture can never queue up behind a slow gateway.
     */
    fun refresh() {
        if (inFlight?.isActive == true) return
        when (val baseUrl = GatewayUrls.normalize(settingsStore.loadServerUrl())) {
            is GatewayUrls.BaseUrl.Invalid -> {
                state = state.copy(
                    connection = GatewayConnection.Unknown,
                    hasLoaded = true,
                    loading = false,
                    error = baseUrl.reason,
                )
            }

            is GatewayUrls.BaseUrl.Valid -> {
                val token = session.remoteToken
                if (token.isBlank()) {
                    state = state.copy(
                        connection = GatewayConnection.Unknown,
                        hasLoaded = true,
                        loading = false,
                        error = MISSING_TOKEN,
                    )
                } else {
                    val api = MetaHarnessApi(baseUrl.value, token, httpClient)
                    state = state.copy(loading = true)
                    inFlight = viewModelScope.launch { load(api) }
                }
            }
        }
    }

    private suspend fun load(api: MetaHarnessApi) {
        val health = attempt { api.health() }
        val runs = attempt { api.listRuns() }
        state = state.copy(
            connection = health.fold(
                onSuccess = { GatewayConnection.Connected },
                onFailure = { GatewayConnection.Unreachable(describe(it)) },
            ),
            sections = runs.getOrNull()?.let(::runSections) ?: state.sections,
            hasLoaded = true,
            loading = false,
            error = runs.exceptionOrNull()?.let(::describe),
        )
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
        /** Poll cadence of the Runs screen while it is visible. */
        const val POLL_INTERVAL_MILLIS = 3_000L

        private const val MISSING_TOKEN = "Remote token required: set it in Settings"

        fun factory(context: Context): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                RunsViewModel(ServerSettingsStore(context.applicationContext))
            }
        }
    }
}
