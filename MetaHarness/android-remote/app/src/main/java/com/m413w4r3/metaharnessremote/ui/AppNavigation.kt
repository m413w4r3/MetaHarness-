package com.m413w4r3.metaharnessremote.ui

import android.net.Uri
import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.m413w4r3.metaharnessremote.ui.newrun.NewRunScreen
import com.m413w4r3.metaharnessremote.ui.rundetail.RunDetailScreen
import com.m413w4r3.metaharnessremote.ui.runs.RunsScreen
import com.m413w4r3.metaharnessremote.ui.settings.SettingsScreen

object Routes {
    const val RUNS = "runs"
    const val SETTINGS = "settings"
    const val NEW_RUN = "new-run"

    const val RUN_ID = "runId"
    const val RUN_DETAIL = "run/{$RUN_ID}"

    /** The Run Detail route of one run id; the id is escaped for the URL. */
    fun runDetail(runId: String): String = "run/${Uri.encode(runId)}"
}

@Composable
fun AppNavigation(navController: NavHostController = rememberNavController()) {
    NavHost(navController = navController, startDestination = Routes.RUNS) {
        composable(Routes.RUNS) {
            RunsScreen(
                onOpenRun = { runId -> navController.navigate(Routes.runDetail(runId)) },
                onNewRun = { navController.navigate(Routes.NEW_RUN) },
                onOpenSettings = { navController.navigate(Routes.SETTINGS) },
            )
        }
        composable(Routes.NEW_RUN) {
            NewRunScreen(
                onCreated = { runId ->
                    navController.navigate(Routes.runDetail(runId)) {
                        // The form is consumed by the run it created: going back
                        // from the run returns to the board, never to a filled-in
                        // form that could create a second run.
                        popUpTo(Routes.NEW_RUN) { inclusive = true }
                    }
                },
            )
        }
        composable(
            route = Routes.RUN_DETAIL,
            arguments = listOf(navArgument(Routes.RUN_ID) { type = NavType.StringType }),
        ) { entry ->
            RunDetailScreen(runId = entry.arguments?.getString(Routes.RUN_ID).orEmpty())
        }
        composable(Routes.SETTINGS) { SettingsScreen() }
    }
}
