package com.m413w4r3.metaharnessremote.ui.newrun

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.m413w4r3.metaharnessremote.api.MetaHarnessApi

/**
 * New Run screen: one SPEC, an optional run id and one `CREATE RUN`.
 *
 * A tap validates the form and sends exactly one `POST /v1/runs`; a create
 * that succeeds hands its run id to [onCreated], so the screen that follows
 * opens the run. The button is disabled while the call is in flight, so a
 * double tap cannot create two runs.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewRunScreen(
    onCreated: (String) -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val viewModel: NewRunViewModel = viewModel(factory = NewRunViewModel.factory(context))
    val state = viewModel.state
    val specBytes = state.spec.toByteArray(Charsets.UTF_8).size
    val trimmedRunId = state.runId.trim()
    val runIdIsInvalid = trimmedRunId.isNotEmpty() &&
        !MetaHarnessApi.RUN_ID_PATTERN.matches(trimmedRunId)

    // Keyed on the run id alone: a recomposed callback must not navigate twice.
    state.createdRunId?.let { runId ->
        LaunchedEffect(runId) { onCreated(runId) }
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = { Text("New Run") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                        )
                    }
                },
            )
        },
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .imePadding()
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                OutlinedTextField(
                    value = state.spec,
                    onValueChange = viewModel::onSpecChange,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("SPEC") },
                    minLines = SPEC_MIN_LINES,
                    maxLines = SPEC_MAX_LINES,
                    isError = state.specError != null,
                )
                Text(
                    text = "$specBytes / ${MetaHarnessApi.MAX_SPEC_BYTES} bytes",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (specBytes > MetaHarnessApi.MAX_SPEC_BYTES) {
                        MaterialTheme.colorScheme.error
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
                FieldError(state.specError)
            }
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                OutlinedTextField(
                    value = state.runId,
                    onValueChange = viewModel::onRunIdChange,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Run ID optional") },
                    singleLine = true,
                    isError = state.runIdError != null,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Ascii,
                        imeAction = ImeAction.Done,
                    ),
                )
                Text(
                    text = "Letters, numbers, '.', '_' and '-'",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (runIdIsInvalid) {
                        MaterialTheme.colorScheme.error
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
                FieldError(state.runIdError)
            }
            state.error?.let { message ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                    ),
                ) {
                    Text(
                        text = message,
                        modifier = Modifier.padding(16.dp),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onErrorContainer,
                    )
                }
            }
            Button(
                onClick = {
                    if (!state.submitting && state.createdRunId == null) {
                        viewModel.createRun()
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = !state.submitting && state.createdRunId == null,
            ) {
                if (state.submitting) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                        )
                        Text("CREATING…")
                    }
                } else {
                    Text("CREATE RUN")
                }
            }
        }
    }
}

@Composable
private fun FieldError(message: String?) {
    if (message == null) return
    Text(
        text = message,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.error,
    )
}

private const val SPEC_MIN_LINES = 6
private const val SPEC_MAX_LINES = 12
