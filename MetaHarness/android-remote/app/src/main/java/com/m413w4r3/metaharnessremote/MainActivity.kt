package com.m413w4r3.metaharnessremote

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.m413w4r3.metaharnessremote.ui.AppNavigation
import com.m413w4r3.metaharnessremote.ui.theme.MetaHarnessRemoteTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MetaHarnessRemoteTheme {
                AppNavigation()
            }
        }
    }
}
