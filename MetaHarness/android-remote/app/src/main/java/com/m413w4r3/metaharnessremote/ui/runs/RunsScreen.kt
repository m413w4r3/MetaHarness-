package com.m413w4r3.metaharnessremote.ui.runs

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Badge
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
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
import java.util.Locale

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
                    if (state.connection == GatewayConnection.Connected && state.sections.isEmpty() && state.error == null) {
                        item(key = "empty") { EmptyRunsState(onNewRun) }
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
        ConnectionStatusCard(state = state, onRetry = onRefresh)
        BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
            if (maxWidth < 340.dp) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onNewRun, modifier = Modifier.fillMaxWidth()) { Text("+ NEW RUN") }
                    OutlinedButton(
                        onClick = onRefresh,
                        enabled = !state.loading,
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("Refresh") }
                }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = onNewRun, modifier = Modifier.weight(1f)) { Text("+ NEW RUN") }
                    OutlinedButton(
                        onClick = onRefresh,
                        enabled = !state.loading,
                        modifier = Modifier.weight(1f),
                    ) { Text("Refresh") }
                }
            }
        }
    }
}

@Composable
private fun ConnectionStatusCard(state: RunsUiState, onRetry: () -> Unit) {
    val connected = state.connection == GatewayConnection.Connected
    val offlineReason = when (val connection = state.connection) {
        is GatewayConnection.Unreachable -> connection.message
        GatewayConnection.Unknown -> state.error.takeIf { state.hasLoaded }
        GatewayConnection.Connected -> null
    }
    val label = when {
        connected -> "CONNECTED"
        offlineReason != null -> "OFFLINE"
        else -> "CONNECTING"
    }
    val color = when {
        connected -> MaterialTheme.colorScheme.primary
        offlineReason != null -> MaterialTheme.colorScheme.error
        else -> MaterialTheme.colorScheme.outline
    }
    val containerColor = when {
        connected -> MaterialTheme.colorScheme.primaryContainer
        offlineReason != null -> MaterialTheme.colorScheme.errorContainer
        else -> MaterialTheme.colorScheme.surfaceContainerLow
    }
    val description = when {
        connected -> "MetaHarness gateway reachable"
        offlineReason != null -> offlineReason.shortReason()
        else -> "Checking MetaHarness gateway…"
    }
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = containerColor),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(start = 14.dp, end = 6.dp, top = 6.dp, bottom = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(text = "●", color = color, style = MaterialTheme.typography.bodyMedium)
                    Text(text = label, color = color, style = MaterialTheme.typography.labelLarge)
                }
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            TextButton(onClick = onRetry, enabled = !state.loading) { Text("RETRY") }
        }
    }
}

private fun String.shortReason(maxLength: Int = 96): String =
    trim().let { if (it.length <= maxLength) it else it.take(maxLength - 1).trimEnd() + "…" }

@Composable
private fun SectionHeader(section: RunSection) {
    if (section.category == RunCategory.ACTION_REQUIRED) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = section.category.label,
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.weight(1f),
                )
                Badge { Text(section.runs.size.toString()) }
            }
        }
    } else {
        Row(
            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
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
}

@Composable
private fun RunCardView(card: RunCard, onOpen: () -> Unit) {
    val actionRequired = card.category == RunCategory.ACTION_REQUIRED
    Card(
        onClick = onOpen,
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (actionRequired) MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.48f)
            else MaterialTheme.colorScheme.surfaceContainerLow,
        ),
    ) {
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
                StatusBadge(card)
            }
            Text(
                text = card.planTitle ?: DASH,
                style = MaterialTheme.typography.bodyMedium,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = metadataLine(card),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            card.failure?.let { failure ->
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.small,
                    color = MaterialTheme.colorScheme.errorContainer,
                ) {
                    Text(
                        text = failure,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        maxLines = 3,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }
        }
    }
}

@Composable
private fun StatusBadge(card: RunCard) {
    val label = when (card.status?.lowercase(Locale.ROOT)) {
        "awaiting_plan_approval" -> "PLAN APPROVAL"
        "waiting_scope_approval" -> "SCOPE APPROVAL"
        "blocked" -> "BLOCKED"
        else -> card.status?.uppercase(Locale.ROOT)?.replace('_', ' ') ?: DASH
    }
    val actionRequired = card.category == RunCategory.ACTION_REQUIRED
    val foreground = when {
        actionRequired || card.category == RunCategory.FAILED -> MaterialTheme.colorScheme.error
        card.category == RunCategory.ACTIVE -> MaterialTheme.colorScheme.primary
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }
    val background = when {
        actionRequired || card.category == RunCategory.FAILED -> MaterialTheme.colorScheme.errorContainer
        card.category == RunCategory.ACTIVE -> MaterialTheme.colorScheme.primaryContainer
        else -> MaterialTheme.colorScheme.surfaceVariant
    }
    Surface(
        modifier = Modifier.widthIn(max = 148.dp),
        shape = MaterialTheme.shapes.small,
        color = background,
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp),
            style = MaterialTheme.typography.labelSmall,
            color = foreground,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun EmptyRunsState(onNewRun: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().heightIn(min = 252.dp).padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text("No runs yet", style = MaterialTheme.typography.headlineSmall)
            Text(
                "Create your first MetaHarness run from this device.",
                modifier = Modifier.padding(top = 8.dp),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Button(onClick = onNewRun, modifier = Modifier.padding(top = 20.dp)) {
                Text("+ NEW RUN")
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

/** `updated time · short SHA`, with the desktop's placeholder. */
private fun metadataLine(card: RunCard): String =
    listOf(card.updatedAt ?: DASH, card.commitSha ?: DASH).joinToString(" · ")

@Composable
private fun sectionColor(category: RunCategory): Color = when (category) {
    RunCategory.ACTION_REQUIRED, RunCategory.FAILED -> MaterialTheme.colorScheme.error
    RunCategory.ACTIVE -> MaterialTheme.colorScheme.primary
    RunCategory.COMPLETED, RunCategory.OTHER -> MaterialTheme.colorScheme.onSurfaceVariant
}

private const val DASH = "—"
