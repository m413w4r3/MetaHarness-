package com.m413w4r3.metaharnessremote.ui

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.m413w4r3.metaharnessremote.ui.settings.SettingsScreen

object Routes {
    const val SETTINGS = "settings"
}

@Composable
fun AppNavigation(navController: NavHostController = rememberNavController()) {
    NavHost(navController = navController, startDestination = Routes.SETTINGS) {
        composable(Routes.SETTINGS) { SettingsScreen() }
    }
}
