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
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

sealed interface ConnectionStatus {
    data object Idle : ConnectionStatus
    data object Testing : ConnectionStatus
    data object Connected : ConnectionStatus
    data class Error(val message: String) : ConnectionStatus
}

internal interface SettingsStorePort {
    fun loadServerUrl(): String
    fun saveServerUrl(serverUrl: String)
}

internal interface TokenStorePort {
    fun loadRemoteToken(): String?
    fun saveRemoteToken(token: String)
    fun clearRemoteToken()
}

/** Settings drafts stay in memory until the operator saves them. */
class SettingsViewModel internal constructor(
    private val settingsStore: SettingsStorePort,
    private val tokenStore: TokenStorePort,
    private val gatewayClient: GatewayClient = GatewayClient(),
    private val session: ConnectionSession = ConnectionSession.shared,
) : ViewModel() {

    /** The editable server address. */
    var serverUrl by mutableStateOf(settingsStore.loadServerUrl())
        private set

    /** The editable token, loaded from the encrypted store at launch. */
    var remoteToken by mutableStateOf(tokenStore.loadRemoteToken().orEmpty())
        private set

    var status by mutableStateOf<ConnectionStatus>(ConnectionStatus.Idle)
        private set

    private var connectionTestJob: Job? = null
    private var connectionTestGeneration: Long = 0

    private fun invalidateConnectionTest() {
        connectionTestGeneration += 1
        connectionTestJob?.cancel()
        connectionTestJob = null
    }

    fun onServerUrlChange(value: String) {
        invalidateConnectionTest()
        serverUrl = value
        status = ConnectionStatus.Idle
    }

    fun onRemoteTokenChange(value: String) {
        invalidateConnectionTest()
        remoteToken = value
        status = ConnectionStatus.Idle
    }

    /** Validates and persists the current draft, then updates the process session. */
    fun saveCredentials(): Boolean {
        invalidateConnectionTest()
        val normalizedUrl = when (val result = GatewayUrls.normalize(serverUrl)) {
            is GatewayUrls.BaseUrl.Invalid -> {
                status = ConnectionStatus.Error(result.reason)
                return false
            }
            is GatewayUrls.BaseUrl.Valid -> result.value
        }
        if (remoteToken.isBlank()) {
            status = ConnectionStatus.Error("Remote token is required")
            return false
        }

        return try {
            // Store the token first so a Keystore failure cannot leave a URL saved
            // alongside a token that failed to encrypt.
            tokenStore.saveRemoteToken(remoteToken)
            settingsStore.saveServerUrl(normalizedUrl)
            serverUrl = normalizedUrl
            session.remoteToken = remoteToken
            status = ConnectionStatus.Idle
            true
        } catch (_: Exception) {
            status = ConnectionStatus.Error("Could not save connection credentials")
            false
        }
    }

    /** Drops the server URL and encrypted token from the device. */
    fun forgetCredentials() {
        invalidateConnectionTest()
        settingsStore.saveServerUrl("")
        tokenStore.clearRemoteToken()
        serverUrl = ""
        remoteToken = ""
        session.remoteToken = ""
        status = ConnectionStatus.Idle
    }

    /** Tests the current in-memory draft without persisting it. */
    fun testConnection() {
        connectionTestJob?.cancel()
        connectionTestJob = null
        val generation = ++connectionTestGeneration

        val baseUrl = when (val result = GatewayUrls.normalize(serverUrl)) {
            is GatewayUrls.BaseUrl.Invalid -> {
                status = ConnectionStatus.Error(result.reason)
                return
            }
            is GatewayUrls.BaseUrl.Valid -> result.value
        }
        if (remoteToken.isBlank()) {
            status = ConnectionStatus.Error("Remote token is required")
            return
        }

        val testedUrl = baseUrl
        val testedToken = remoteToken
        status = ConnectionStatus.Testing
        connectionTestJob = viewModelScope.launch {
            val result = gatewayClient.health(testedUrl, testedToken)
            if (generation != connectionTestGeneration) {
                return@launch
            }

            status = when (result) {
                is HealthResult.Connected -> ConnectionStatus.Connected
                is HealthResult.Failed -> ConnectionStatus.Error(result.message)
            }
            connectionTestJob = null
        }
    }

    companion object {
        fun factory(context: Context): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                val applicationContext = context.applicationContext
                val serverSettings = ServerSettingsStore(applicationContext)
                val secureTokenStore = SecureTokenStore(applicationContext)
                SettingsViewModel(
                    settingsStore = object : SettingsStorePort {
                        override fun loadServerUrl(): String = serverSettings.loadServerUrl()
                        override fun saveServerUrl(serverUrl: String) =
                            serverSettings.saveServerUrl(serverUrl)
                    },
                    tokenStore = object : TokenStorePort {
                        override fun loadRemoteToken(): String? = secureTokenStore.loadRemoteToken()
                        override fun saveRemoteToken(token: String) =
                            secureTokenStore.saveRemoteToken(token)
                        override fun clearRemoteToken() = secureTokenStore.clearRemoteToken()
                    },
                )
            }
        }
    }
}
