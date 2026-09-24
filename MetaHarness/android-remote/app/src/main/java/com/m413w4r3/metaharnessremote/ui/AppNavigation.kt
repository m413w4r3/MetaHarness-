package com.m413w4r3.metaharnessremote.ui

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.m413w4r3.metaharnessremote.ui.runs.RunsScreen
import com.m413w4r3.metaharnessremote.ui.settings.SettingsScreen

object Routes {
    const val RUNS = "runs"
    const val SETTINGS = "settings"
}

@Composable
fun AppNavigation(navController: NavHostController = rememberNavController()) {
    NavHost(navController = navController, startDestination = Routes.RUNS) {
        composable(Routes.RUNS) {
            RunsScreen(
                onOpenRun = { _ -> /* the Run Detail screen arrives with a later prompt */ },
                onNewRun = { /* the New Run screen arrives with a later prompt */ },
                onOpenSettings = { navController.navigate(Routes.SETTINGS) },
            )
        }
        composable(Routes.SETTINGS) { SettingsScreen() }
    }
}
