package com.m413w4r3.metaharnessremote.ui.newrun

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
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
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            OutlinedTextField(
                value = state.spec,
                onValueChange = viewModel::onSpecChange,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("SPEC") },
                minLines = SPEC_MIN_LINES,
                maxLines = SPEC_MAX_LINES,
                isError = state.specError != null,
            )
            FieldError(state.specError)
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
            FieldError(state.runIdError)
            Button(
                onClick = viewModel::createRun,
                enabled = !state.submitting && state.createdRunId == null,
            ) {
                Text("CREATE RUN")
            }
            if (state.submitting) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Text(text = "Creating the run…", style = MaterialTheme.typography.bodyMedium)
                }
            }
            state.error?.let { message ->
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                )
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
