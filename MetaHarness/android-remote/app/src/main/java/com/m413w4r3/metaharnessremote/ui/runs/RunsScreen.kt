package com.m413w4r3.metaharnessremote.ui.runs

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Badge
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.repeatOnLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.delay

/**
 * Main screen: the run board of the remote gateway.
 *
 * `GET /v1/health` and `GET /v1/runs` are read on entry and then polled every
 * [RunsViewModel.POLL_INTERVAL_MILLIS] while this screen is resumed, so a screen
 * that is not visible costs nothing. Pulling down reloads immediately.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RunsScreen(
    onOpenRun: (String) -> Unit,
    onNewRun: () -> Unit,
    onOpenSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val viewModel: RunsViewModel = viewModel(factory = RunsViewModel.factory(context))
    val lifecycleOwner = LocalLifecycleOwner.current
    val state = viewModel.state

    LaunchedEffect(lifecycleOwner, viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
            while (true) {
                viewModel.refresh()
                if (viewModel.state.setupState != SetupState.READY) break
                delay(RunsViewModel.POLL_INTERVAL_MILLIS)
            }
        }
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = { Text("MetaHarness Remote") },
                actions = {
                    TextButton(onClick = onOpenSettings) { Text("Settings") }
                },
            )
        },
    ) { innerPadding ->
        if (state.setupState != SetupState.READY) {
            Box(
                modifier = Modifier.fillMaxSize().padding(innerPadding),
                contentAlignment = Alignment.Center,
            ) {
                SetupCard(
                    setupState = state.setupState,
                    onOpenSettings = onOpenSettings,
                    modifier = Modifier.fillMaxWidth().padding(16.dp),
                )
            }
        } else {
            PullToRefreshBox(
                isRefreshing = state.loading && state.hasLoaded,
                onRefresh = viewModel::refresh,
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding),
            ) {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    item(key = "header") {
                        Header(
                            state = state,
                            onNewRun = onNewRun,
                            onRefresh = viewModel::refresh,
                        )
                    }
                    state.error?.let { message -> item(key = "error") { Note(message, isError = true) } }
                    if (!state.hasLoaded) item(key = "loading") { Note("Loading runs…", isError = false) }
                    if (state.hasLoaded && state.sections.isEmpty() && state.error == null) {
                        item(key = "empty") { Note("No runs yet.", isError = false) }
                    }
                    state.sections.forEach { section ->
                        item(key = "section:${section.category.name}") { SectionHeader(section) }
                        items(section.runs, key = { "run:${it.runId}" }) { card ->
                            RunCardView(card = card, onOpen = { onOpenRun(card.runId) })
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SetupCard(
    setupState: SetupState,
    onOpenSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val message = when (setupState) {
        SetupState.READY -> return
        SetupState.MISSING_SERVER_URL -> "Set the HTTPS address of your MetaHarness remote gateway."
        SetupState.MISSING_REMOTE_TOKEN -> "Enter the remote access token for your gateway."
    }
    Card(modifier = modifier) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("Connect MetaHarness", style = MaterialTheme.typography.titleLarge)
            Text(message, style = MaterialTheme.typography.bodyMedium)
            Button(onClick = onOpenSettings) { Text("OPEN SETTINGS") }
        }
    }
}

@Composable
private fun Header(
    state: RunsUiState,
    onNewRun: () -> Unit,
    onRefresh: () -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        ConnectionLine(state)
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = onNewRun) { Text("+ NEW RUN") }
            OutlinedButton(onClick = onRefresh, enabled = !state.loading) { Text("Refresh") }
        }
    }
}

@Composable
private fun ConnectionLine(state: RunsUiState) {
    val connected = state.connection == GatewayConnection.Connected
    val color = when {
        !state.hasLoaded -> MaterialTheme.colorScheme.outline
        connected -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.error
    }
    val label = when {
        !state.hasLoaded -> "Connecting…"
        connected -> "Connected"
        else -> "Not connected"
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text(text = "●", color = color, style = MaterialTheme.typography.bodyMedium)
        Text(text = label, color = color, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun SectionHeader(section: RunSection) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            text = section.category.label,
            style = MaterialTheme.typography.titleSmall,
            color = sectionColor(section.category),
            modifier = Modifier.weight(1f),
        )
        Text(
            text = section.runs.size.toString(),
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun RunCardView(card: RunCard, onOpen: () -> Unit) {
    Card(onClick = onOpen, modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = card.runId,
                    style = MaterialTheme.typography.titleSmall,
                    fontFamily = FontFamily.Monospace,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f),
                )
                if (card.category == RunCategory.ACTION_REQUIRED) {
                    Badge {
                        Text(
                            text = RunCategory.ACTION_REQUIRED.label,
                            style = MaterialTheme.typography.labelSmall,
                        )
                    }
                }
            }
            Text(
                text = card.planTitle ?: DASH,
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = runLine(card),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            card.failure?.let { failure ->
                Text(
                    text = failure,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

@Composable
private fun Note(message: String, isError: Boolean) {
    Text(
        text = message,
        style = MaterialTheme.typography.bodySmall,
        color = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

/** `status · short commit · updatedAt`, with the desktop's placeholder. */
private fun runLine(card: RunCard): String =
    listOf(card.status ?: DASH, card.commitSha ?: DASH, card.updatedAt ?: DASH).joinToString(" · ")

@Composable
private fun sectionColor(category: RunCategory): Color = when (category) {
    RunCategory.ACTION_REQUIRED, RunCategory.FAILED -> MaterialTheme.colorScheme.error
    RunCategory.ACTIVE -> MaterialTheme.colorScheme.primary
    RunCategory.COMPLETED, RunCategory.OTHER -> MaterialTheme.colorScheme.onSurfaceVariant
}

private const val DASH = "—"
