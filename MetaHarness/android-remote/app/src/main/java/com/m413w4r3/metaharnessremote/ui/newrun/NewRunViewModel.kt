package com.m413w4r3.metaharnessremote.ui.newrun

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
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient

/** Everything the New Run screen renders. */
data class NewRunUiState(
    /** The SPEC as typed; it is sent byte for byte, only its trimmed form is checked. */
    val spec: String = "",
    /** The run id as typed; an empty field asks the gateway to generate one. */
    val runId: String = "",
    val specError: String? = null,
    val runIdError: String? = null,
    /** One `POST /v1/runs` is in flight: a second tap cannot start another. */
    val submitting: Boolean = false,
    /** Failure of the last create, in the words of [newRunFailure]; or null. */
    val error: String? = null,
    /** The run the gateway created: the screen opens its Run Detail. */
    val createdRunId: String? = null,
)

/**
 * Form state of the New Run screen.
 *
 * Exactly one `POST /v1/runs` is sent per tap and nothing is ever retried: a
 * create that timed out may still have created the run, so the screen reports
 * the unknown outcome and leaves the decision to the operator.
 */
class NewRunViewModel(
    private val settingsStore: ServerSettingsStore,
    private val session: ConnectionSession = ConnectionSession.shared,
    private val httpClient: OkHttpClient = GatewayClient.defaultHttpClient(),
) : ViewModel() {

    var state by mutableStateOf(NewRunUiState())
        private set

    /** The SPEC as typed; editing a field drops the error shown for it. */
    fun onSpecChange(value: String) {
        state = state.copy(spec = value, specError = null, error = null)
    }

    fun onRunIdChange(value: String) {
        state = state.copy(runId = value, runIdError = null, error = null)
    }

    /** Validate the form, then send at most one create. */
    fun createRun() {
        if (state.submitting) return
        when (val validation = validateNewRun(state.spec, state.runId)) {
            is NewRunValidation.Invalid -> state = state.copy(
                specError = validation.specError,
                runIdError = validation.runIdError,
                error = null,
            )

            is NewRunValidation.Valid -> submit(validation)
        }
    }

    private fun submit(request: NewRunValidation.Valid) {
        when (val baseUrl = GatewayUrls.normalize(settingsStore.loadServerUrl())) {
            is GatewayUrls.BaseUrl.Invalid -> state = state.copy(error = baseUrl.reason)

            is GatewayUrls.BaseUrl.Valid -> {
                val token = session.remoteToken
                if (token.isBlank()) {
                    state = state.copy(error = MISSING_TOKEN)
                } else {
                    val api = MetaHarnessApi(baseUrl.value, token, httpClient)
                    state = state.copy(submitting = true, error = null)
                    viewModelScope.launch { send(api, request) }
                }
            }
        }
    }

    /**
     * One create, reported as it ends. [CancellationException] belongs to the
     * scope: the screen was left, so its answer is shown nowhere.
     */
    private suspend fun send(api: MetaHarnessApi, request: NewRunValidation.Valid) {
        try {
            val created = api.createRun(request.spec, request.runId)
            state = state.copy(submitting = false, error = null, createdRunId = created.runId)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Exception) {
            state = state.copy(submitting = false, error = newRunFailure(failure))
        }
    }

    companion object {
        private const val MISSING_TOKEN = "Remote token required: set it in Settings"

        fun factory(context: Context): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                NewRunViewModel(ServerSettingsStore(context.applicationContext))
            }
        }
    }
}
