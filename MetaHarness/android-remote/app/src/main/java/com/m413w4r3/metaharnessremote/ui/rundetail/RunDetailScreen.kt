package com.m413w4r3.metaharnessremote.ui.rundetail

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListScope
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
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
import com.m413w4r3.metaharnessremote.ui.runs.RunCategory
import kotlinx.coroutines.delay

/**
 * Run Detail: one run, read-only.
 *
 * `GET /v1/runs/{runId}` and `GET /v1/runs/{runId}/progress?offset=N` are read
 * on entry and then on the cadence the run itself asks for
 * ([runPollIntervalMillis]) while this screen is resumed: every two seconds
 * while the run is active, every five seconds while it waits for the operator,
 * and not at all once it is completed or failed. A screen that is not visible
 * polls nothing, because leaving [Lifecycle.State.RESUMED] cancels the loop.
 *
 * Progress is incremental: the offset starts at 0 and then follows the
 * `next_offset` of every answer, so each pass adds only the events that became
 * visible since the previous one.
 */
@Composable
fun RunDetailScreen(runId: String, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val viewModel: RunDetailViewModel =
        viewModel(key = runId, factory = RunDetailViewModel.factory(context, runId))
    val lifecycleOwner = LocalLifecycleOwner.current
    val state = viewModel.state

    // A pass returns the delay it wants next, or null for a terminal run.
    LaunchedEffect(lifecycleOwner, viewModel) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
            while (true) {
                val interval = viewModel.refreshOnce() ?: break
                delay(interval)
            }
        }
    }

    LazyColumn(
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item(key = "header") { Header(state.runId) }
        state.error?.let { message -> item(key = "error") { Note(message, isError = true) } }
        if (!state.hasLoaded) item(key = "loading") { Note("Loading run…", isError = false) }
        state.detail?.let { detail -> detailItems(detail, state) }
    }
}

/** The sections of one loaded run, in the order the screen reads them. */
private fun LazyListScope.detailItems(
    detail: RunDetailView,
    state: RunDetailUiState,
) {
    item(key = "state") { StateCard(detail, state.pollingStopped) }
    detail.failure?.let { failure -> item(key = "failure") { Failure(failure) } }
    if (detail.overview.isNotEmpty()) {
        item(key = "overview-title") { SectionTitle("OVERVIEW") }
        detail.overview.forEach { line ->
            item(key = "overview:${line.label}") { Field(line.label, line.value) }
        }
    }
    detail.plan?.let { plan ->
        item(key = "plan-title") { SectionTitle("PLAN") }
        item(key = "plan") { Body(plan) }
    }
    item(key = "steps-title") { SectionTitle("IMPLEMENTATION STEPS") }
    if (detail.steps.isEmpty()) {
        item(key = "steps-empty") { Note("No implementation steps.", isError = false) }
    } else {
        items(detail.steps, key = { "step:${it.id}" }) { step -> StepCard(step) }
    }
    item(key = "progress-title") {
        SectionTitle("PROGRESS · offset ${state.progress.offset}")
    }
    if (state.progress.events.isEmpty()) {
        item(key = "progress-empty") { Note("No progress events yet.", isError = false) }
    } else {
        // Newest first: a polled log is read from its end, without scrolling.
        items(state.progress.events.asReversed()) { event -> Body(event, mono = true) }
    }
}

@Composable
private fun Header(runId: String) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(text = "Run Detail", style = MaterialTheme.typography.headlineSmall)
        Text(
            text = runId.ifBlank { DASH },
            style = MaterialTheme.typography.titleMedium,
            fontFamily = FontFamily.Monospace,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}

@Composable
private fun StateCard(detail: RunDetailView, pollingStopped: Boolean) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Field("STATUS", detail.status ?: DASH, color = statusColor(detail.status))
            detail.planTitle?.let { title -> Field("PLAN TITLE", title) }
            Field("CYCLE", detail.cycle?.toString() ?: DASH)
            Field("CANDIDATE SHA", detail.candidateSha ?: DASH, mono = true)
            Field("UPDATED", detail.updatedAt ?: DASH)
            if (pollingStopped) {
                Note("Polling stopped: the run is ${detail.status ?: DASH}.", false)
            }
        }
    }
}

@Composable
private fun StepCard(step: RunStep) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = step.id,
                    style = MaterialTheme.typography.titleSmall,
                    fontFamily = FontFamily.Monospace,
                )
                Text(
                    text = step.executionClass ?: DASH,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            step.title?.let { title -> Text(text = title, style = MaterialTheme.typography.bodyMedium) }
            step.profile?.let { profile ->
                Text(
                    text = "${step.profileLabel} · $profile",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontFamily = FontFamily.Monospace,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
    }
}

@Composable
private fun SectionTitle(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.titleSmall,
        color = MaterialTheme.colorScheme.primary,
    )
}

@Composable
private fun Field(
    label: String,
    value: String,
    mono: Boolean = false,
    color: Color = Color.Unspecified,
) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontFamily = if (mono) FontFamily.Monospace else null,
            color = color,
        )
    }
}

@Composable
private fun Body(text: String, mono: Boolean = false) {
    Text(
        text = text,
        style = MaterialTheme.typography.bodySmall,
        fontFamily = if (mono) FontFamily.Monospace else null,
    )
}

@Composable
private fun Failure(failure: String) {
    Text(
        text = failure,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.error,
    )
}

@Composable
private fun Note(message: String, isError: Boolean) {
    Text(
        text = message,
        style = MaterialTheme.typography.bodySmall,
        color = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@Composable
private fun statusColor(status: String?): Color = when (RunCategory.of(status)) {
    RunCategory.ACTION_REQUIRED, RunCategory.FAILED -> MaterialTheme.colorScheme.error
    RunCategory.ACTIVE -> MaterialTheme.colorScheme.primary
    RunCategory.COMPLETED, RunCategory.OTHER -> MaterialTheme.colorScheme.onSurface
}

private const val DASH = "—"
