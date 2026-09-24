package com.m413w4r3.metaharnessremote.ui.settings

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.m413w4r3.metaharnessremote.data.ConnectionSession
import com.m413w4r3.metaharnessremote.data.SecureTokenStore
import com.m413w4r3.metaharnessremote.data.ServerSettingsStore
import com.m413w4r3.metaharnessremote.network.GatewayClient
import com.m413w4r3.metaharnessremote.network.GatewayUrls
import com.m413w4r3.metaharnessremote.network.HealthResult
import kotlinx.coroutines.launch

sealed interface ConnectionStatus {
    data object Idle : ConnectionStatus
    data object Testing : ConnectionStatus
    data object Connected : ConnectionStatus
    data class Error(val message: String) : ConnectionStatus
}

/**
 * Connection settings of the app.
 *
 * The typed token is mirrored into [ConnectionSession], where the other screens
 * read it, and handed to [SecureTokenStore], which encrypts it under an
 * Android Keystore key before it reaches the disk.
 */
class SettingsViewModel(
    private val settingsStore: ServerSettingsStore,
    private val tokenStore: SecureTokenStore,
    private val gatewayClient: GatewayClient = GatewayClient(),
    private val session: ConnectionSession = ConnectionSession.shared,
) : ViewModel() {

    /** Persisted across launches. */
    var serverUrl by mutableStateOf(settingsStore.loadServerUrl())
        private set

    /** Reloaded from the encrypted store at launch, then kept in memory only. */
    var remoteToken by mutableStateOf(tokenStore.loadRemoteToken().orEmpty())
        private set

    var status by mutableStateOf<ConnectionStatus>(ConnectionStatus.Idle)
        private set

    init {
        session.remoteToken = remoteToken
    }

    fun onServerUrlChange(value: String) {
        serverUrl = value
        settingsStore.saveServerUrl(value)
        status = ConnectionStatus.Idle
    }

    fun onRemoteTokenChange(value: String) {
        remoteToken = value
        session.remoteToken = value
        tokenStore.saveRemoteToken(value)
        status = ConnectionStatus.Idle
    }

    /** Drops the server URL and the stored token from the device. */
    fun forgetCredentials() {
        serverUrl = ""
        remoteToken = ""
        session.remoteToken = ""
        settingsStore.saveServerUrl("")
        tokenStore.clearRemoteToken()
        status = ConnectionStatus.Idle
    }

    fun testConnection() {
        when (val baseUrl = GatewayUrls.normalize(serverUrl)) {
            is GatewayUrls.BaseUrl.Invalid -> status = ConnectionStatus.Error(baseUrl.reason)
            is GatewayUrls.BaseUrl.Valid -> {
                status = ConnectionStatus.Testing
                viewModelScope.launch {
                    status = when (val result = gatewayClient.health(baseUrl.value, remoteToken)) {
                        is HealthResult.Connected -> ConnectionStatus.Connected
                        is HealthResult.Failed -> ConnectionStatus.Error(result.message)
                    }
                }
            }
        }
    }

    companion object {
        fun factory(context: Context): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val applicationContext = context.applicationContext
                SettingsViewModel(
                    settingsStore = ServerSettingsStore(applicationContext),
                    tokenStore = SecureTokenStore(applicationContext),
                )
            }
        }
    }
}
