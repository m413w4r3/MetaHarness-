package com.m413w4r3.metaharnessremote

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.m413w4r3.metaharnessremote.data.ConnectionSession
import com.m413w4r3.metaharnessremote.data.SecureTokenStore
import com.m413w4r3.metaharnessremote.ui.AppNavigation
import com.m413w4r3.metaharnessremote.ui.theme.MetaHarnessRemoteTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Prime the in-memory session from the encrypted store, so a restart
        // keeps the operator connected without a trip to the Settings screen.
        ConnectionSession.shared.remoteToken =
            SecureTokenStore(applicationContext).loadRemoteToken().orEmpty()
        setContent {
            MetaHarnessRemoteTheme {
                AppNavigation()
            }
        }
    }
}
