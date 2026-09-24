package com.m413w4r3.metaharnessremote.ui.rundetail

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListScope
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
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
 * Run Detail: one run, and the plan decision it may wait for.
 *
 * `GET /v1/runs/{runId}` and `GET /v1/runs/{runId}/progress?offset=N` are read
 * on entry and then on the cadence the run itself asks for
 * ([runPollIntervalMillis]) while this screen is resumed: every two seconds
 * while the run is active, every five seconds while it waits for the operator,
 * and not at all once it is completed or failed — a resume answer starts the
 * loop again, because the run may be live again. A screen that is not visible
 * polls nothing, because leaving [Lifecycle.State.RESUMED] cancels the loop.
 *
 * Progress is incremental: the offset starts at 0 and then follows the
 * `next_offset` of every answer, so each pass adds only the events that became
 * visible since the previous one.
 *
 * While the run waits for a plan decision, a `PLAN APPROVAL` block appears above
 * the plan: it offers the profiles the decision must name, and sends one
 * `POST /v1/runs/{runId}/approval` per decision — never two at a time. While it
 * waits for a repair scope, a `SCOPE APPROVAL` block shows the paths the delta
 * adds and sends one `POST /v1/runs/{runId}/scope-approval` per decision; while
 * the run publishes a resumable checkpoint, a `RESUME` block sends one
 * `POST /v1/runs/{runId}/resume`. While the run offers a plan recovery, a
 * `RECOVER PLAN` block collects a replacement and sends one
 * `POST /v1/runs/{runId}/recover-plan` once the operator confirms it. Every
 * irreversible or replayable action is confirmed first or reported as possibly
 * applied, and nothing is ever retried.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RunDetailScreen(
    runId: String,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val viewModel: RunDetailViewModel =
        viewModel(key = runId, factory = RunDetailViewModel.factory(context, runId))
    val lifecycleOwner = LocalLifecycleOwner.current
    val state = viewModel.state
    val approvalActions = remember(viewModel) {
        RunApprovalActions(
            selectFinalReviewer = viewModel::selectFinalReviewer,
            selectSemanticReviser = viewModel::selectSemanticReviser,
            selectCheckRepair = viewModel::selectCheckRepair,
            selectStepProfile = viewModel::selectStepProfile,
            approve = viewModel::approve,
            askReject = viewModel::askReject,
        )
    }
    val scopeActions = remember(viewModel) {
        RunScopeActions(approve = viewModel::approveScope, askReject = viewModel::askScopeReject)
    }
    val recoveryActions = remember(viewModel) {
        RunRecoveryActions(updatePlan = viewModel::updateReplacementPlan, askReplace = viewModel::askRecoverPlan)
    }
    val resumeActions = remember(viewModel) { RunResumeActions(resume = viewModel::resume) }

    // A pass returns the delay it wants next, or null for a terminal run. A
    // resume answer starts the loop again: the run may be live again.
    LaunchedEffect(lifecycleOwner, viewModel, state.resumeAnswers) {
        lifecycleOwner.lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) {
            while (true) {
                val interval = viewModel.refreshOnce() ?: break
                delay(interval)
            }
        }
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = { Text("Run Detail") },
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
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            item(key = "header") { Header(state.runId) }
            state.error?.let { message -> item(key = "error") { Note(message, isError = true) } }
            if (!state.hasLoaded) item(key = "loading") { Note("Loading run…", isError = false) }
            state.detail?.let { detail ->
                detailItems(detail, state, approvalActions, scopeActions, recoveryActions, resumeActions)
            }
        }
    }

    // The dialog sits outside the list: a lazy item may be disposed while the
    // operator reads the plan, and the confirmation must survive that.
    if (state.approval.confirmingReject) {
        RejectPlanDialog(
            submitting = state.approval.submitting,
            onConfirm = viewModel::confirmReject,
            onDismiss = viewModel::dismissReject,
        )
    }
    if (state.scope.confirmingReject) {
        RejectScopeDialog(
            submitting = state.scope.submitting,
            onConfirm = viewModel::confirmScopeReject,
            onDismiss = viewModel::dismissScopeReject,
        )
    }
    if (state.recovery.confirming) {
        RecoverPlanDialog(
            submitting = state.recovery.submitting,
            onConfirm = viewModel::confirmRecoverPlan,
            onDismiss = viewModel::dismissRecoverPlan,
        )
    }
}

/** The sections of one loaded run, in the order the screen reads them. */
private fun LazyListScope.detailItems(
    detail: RunDetailView,
    state: RunDetailUiState,
    actions: RunApprovalActions,
    scopeActions: RunScopeActions,
    recoveryActions: RunRecoveryActions,
    resumeActions: RunResumeActions,
) {
    item(key = "state") { StateCard(detail, state.pollingStopped) }
    if (state.approval.gate != null) item(key = "approval") { ApprovalCard(state.approval, actions) }
    state.scope.gate?.let { gate ->
        item(key = "scope") { ScopeApprovalCard(gate, state.scope, scopeActions) }
    }
    state.recovery.gate?.let { gate ->
        item(key = "recovery") { RecoverPlanCard(gate, state.recovery, recoveryActions) }
    }
    state.resume.gate?.let { gate ->
        item(key = "resume") { ResumeCard(gate, state.resume, resumeActions) }
    }
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
    Text(
        text = runId.ifBlank { DASH },
        style = MaterialTheme.typography.titleMedium,
        fontFamily = FontFamily.Monospace,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
    )
}

/**
 * What the approval block does; the screen owns the ViewModel that implements
 * it, so the card stays a function of the state it renders.
 */
data class RunApprovalActions(
    val selectFinalReviewer: (String) -> Unit,
    val selectSemanticReviser: (String) -> Unit,
    val selectCheckRepair: (String) -> Unit,
    val selectStepProfile: (stepId: String, profileId: String) -> Unit,
    val approve: () -> Unit,
    val askReject: () -> Unit,
)

/**
 * What the scope approval block does; the screen owns the ViewModel that
 * implements it, so the card stays a function of the state it renders.
 */
data class RunScopeActions(
    val approve: () -> Unit,
    val askReject: () -> Unit,
)

/**
 * What the resume block does; the screen owns the ViewModel that implements it,
 * so the card stays a function of the state it renders.
 */
data class RunResumeActions(val resume: () -> Unit)

/**
 * What the plan recovery block does; the screen owns the ViewModel that
 * implements it, so the card stays a function of the state it renders.
 */
data class RunRecoveryActions(
    val updatePlan: (String) -> Unit,
    val askReplace: () -> Unit,
)

/**
 * The plan approval of one run: the profiles the decision must name, and the
 * two decisions.
 *
 * The block only reaches this composable when the run document holds the four
 * conditions of the gate, so everything here is about the decision itself: it
 * starts from the profiles the run routed, keeps what the operator changes, and
 * only enables APPROVE once every step and every role the run uses holds a
 * compatible profile.
 */
@Composable
private fun ApprovalCard(approval: RunApprovalUiState, actions: RunApprovalActions) {
    val gate = approval.gate ?: return
    val profiles = approval.profiles
    val selection = approval.selection
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            SectionTitle("PLAN APPROVAL")
            Text(
                text = "The run waits for a decision on this plan.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            approval.error?.let { message -> Note(message, isError = true) }
            approval.profilesError?.let { message -> Note(message, isError = true) }
            when {
                profiles == null || selection == null ->
                    Note("Loading model profiles…", isError = false)

                profiles.options.isEmpty() ->
                    Note("No model profile is available.", isError = true)

                else -> ApprovalForm(gate, profiles, selection, approval.submitting, actions)
            }
            if (approval.submitting) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Text(
                        text = "Sending the decision…",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

/** The three role profiles, the step implementers and the two decisions. */
@Composable
private fun ApprovalForm(
    gate: ApprovalGate,
    profiles: ApprovalProfiles,
    selection: ApprovalSelection,
    submitting: Boolean,
    actions: RunApprovalActions,
) {
    ProfilePicker(
        label = "Final reviewer",
        options = profiles.withRole(REVIEWER_ROLE),
        selected = selection.finalReviewerProfile,
        enabled = !submitting,
        onSelect = actions.selectFinalReviewer,
    )
    ProfilePicker(
        label = "Semantic reviser",
        options = profiles.withRole(REVISER_ROLE),
        selected = selection.semanticReviserProfile,
        enabled = !submitting && gate.semanticRevisionEnabled,
        disabledNote = "This run does not enable semantic revision",
        onSelect = actions.selectSemanticReviser,
    )
    ProfilePicker(
        label = "Check repair",
        options = profiles.withRole(REPAIR_ROLE),
        selected = selection.checkRepairProfile,
        enabled = !submitting && gate.checkRepairEnabled,
        disabledNote = "This run has no correction budget",
        onSelect = actions.selectCheckRepair,
    )
    SectionTitle("STEP PROFILES")
    if (gate.steps.isEmpty()) {
        Note("The plan holds no step to select an implementer for.", isError = true)
    }
    gate.steps.forEach { step ->
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(
                text = listOfNotNull(step.id, step.title).joinToString(" · "),
                style = MaterialTheme.typography.bodyMedium,
            )
            ProfilePicker(
                label = step.executionClass,
                options = profiles.implementers(step),
                selected = selection.stepProfiles[step.id].orEmpty(),
                enabled = !submitting,
                onSelect = { profileId -> actions.selectStepProfile(step.id, profileId) },
            )
        }
    }
    // The payload is built here only to know whether APPROVE may be tapped; the
    // ViewModel builds the one it sends from the same function.
    val approveEnabled = !submitting && approvalPayload(selection, gate, profiles) != null
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(
            onClick = actions.askReject,
            enabled = !submitting,
            colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
        ) {
            Text("REJECT PLAN")
        }
        Button(onClick = actions.approve, enabled = approveEnabled) {
            Text("APPROVE & CONTINUE")
        }
    }
}

/**
 * One profile of the form, read as a button that opens the list of the profiles
 * the run may use for that role or step.
 */
@Composable
private fun ProfilePicker(
    label: String,
    options: List<ProfileOption>,
    selected: String,
    enabled: Boolean,
    disabledNote: String? = null,
    onSelect: (String) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        when {
            !enabled -> Text(
                text = disabledNote ?: selected.ifEmpty { DASH },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            options.isEmpty() -> Text(
                text = "No compatible profile",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
            )

            else -> Box {
                TextButton(onClick = { expanded = true }) {
                    Text(
                        text = selected.ifEmpty { "Select a profile" },
                        fontFamily = FontFamily.Monospace,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    options.forEach { option ->
                        DropdownMenuItem(
                            text = { Text(text = option.id, fontFamily = FontFamily.Monospace) },
                            onClick = {
                                expanded = false
                                onSelect(option.id)
                            },
                        )
                    }
                }
            }
        }
    }
}

/** The irreversibility of a rejection is stated before it is sent, once. */
@Composable
private fun RejectPlanDialog(
    submitting: Boolean,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = { if (!submitting) onDismiss() },
        title = { Text("Reject this plan?") },
        text = {
            Text(
                "This action is irreversible: the run records the rejection and " +
                    "does not implement the plan.",
            )
        },
        confirmButton = {
            TextButton(
                onClick = onConfirm,
                enabled = !submitting,
                colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error),
            ) {
                Text("REJECT PLAN")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, enabled = !submitting) { Text("Cancel") }
        },
    )
}

/**
 * The repair scope of one run: the paths the requested delta adds, and the two
 * decisions.
 *
 * The block only reaches this composable when the run document holds the gate,
 * so it never offers to edit the delta: approval is bound to the exact scope
 * the run recorded, and the rejection is confirmed because it is irreversible.
 */
@Composable
private fun ScopeApprovalCard(gate: ScopeGate, scope: RunScopeUiState, actions: RunScopeActions) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            SectionTitle("SCOPE APPROVAL")
            Text(
                text = "The repair asks for additional mutable scope beyond the automatic limit.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            scope.error?.let { message -> Note(message, isError = true) }
            if (gate.addedPaths.isEmpty()) {
                Note("The scope delta adds no path.", isError = false)
            } else {
                Text(
                    text = "ADDED PATHS · ${gate.addedPaths.size}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                gate.addedPaths.forEach { path -> Body(path, mono = true) }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(
                    onClick = actions.askReject,
                    enabled = !scope.submitting,
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
                ) {
                    Text("REJECT SCOPE")
                }
                Button(onClick = actions.approve, enabled = !scope.submitting) {
                    Text("APPROVE SCOPE")
                }
            }
            if (scope.submitting) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Text(
                        text = "Sending the decision…",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

/**
 * The resume of one run: the checkpoint the run publishes, and the one action
 * that continues it.
 *
 * A resume that failed says so and is never sent again by the screen: a timeout
 * may have been accepted, so the refreshed run is what tells the operator where
 * it stands.
 */
@Composable
private fun ResumeCard(gate: ResumeGate, resume: RunResumeUiState, actions: RunResumeActions) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            SectionTitle("RESUME")
            Text(
                text = "The run holds a checkpoint it can continue from.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            resume.error?.let { message -> Note(message, isError = true) }
            Button(onClick = actions.resume, enabled = !resume.submitting) {
                Text(gate.label ?: "RESUME RUN")
            }
            if (resume.submitting) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Text(
                        text = "Sending the resume request…",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

/** The irreversibility of a scope rejection is stated before it is sent, once. */
@Composable
private fun RejectScopeDialog(
    submitting: Boolean,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = { if (!submitting) onDismiss() },
        title = { Text("Reject this scope?") },
        text = {
            Text(
                "This action is irreversible: the run records the rejection and " +
                    "fails without repairing the scope.",
            )
        },
        confirmButton = {
            TextButton(
                onClick = onConfirm,
                enabled = !submitting,
                colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error),
            ) {
                Text("REJECT SCOPE")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, enabled = !submitting) { Text("Cancel") }
        },
    )
}

/**
 * The plan recovery of one run: why the run offers it, the plan it rejected,
 * and the replacement the operator writes.
 *
 * The block only reaches this composable when the run is eligible and the
 * capability is not explicitly refused, so everything here is about the
 * replacement itself: it is counted in UTF-8 bytes against the bound the
 * gateway published, and it leaves the phone only once the operator confirmed
 * it — the validator that decides is MetaHarness, not this screen.
 */
@Composable
private fun RecoverPlanCard(
    gate: PlanRecoveryGate,
    recovery: RunRecoveryUiState,
    actions: RunRecoveryActions,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            SectionTitle("RECOVER PLAN")
            Text(
                text = "The run holds no executable plan: publish a corrected META PLAN v2 for it.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            recovery.error?.let { message -> Note(message, isError = true) }
            gate.reason?.let { reason -> Field("REASON", reason) }
            gate.rejectedPlan?.let { rejected ->
                var showRejected by remember { mutableStateOf(false) }
                OutlinedButton(onClick = { showRejected = !showRejected }) {
                    Text(if (showRejected) "HIDE REJECTED PLAN" else "SHOW REJECTED PLAN")
                }
                if (showRejected) Body(rejected, mono = true)
            }
            OutlinedTextField(
                value = recovery.replacementPlan,
                onValueChange = actions.updatePlan,
                enabled = !recovery.submitting,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Replacement META PLAN v2") },
                minLines = 12,
            )
            val overLimit = gate.maxBytes != null && recovery.replacementBytes > gate.maxBytes
            Text(
                text = "${recovery.replacementBytes} / ${gate.maxBytes ?: DASH} bytes UTF-8",
                style = MaterialTheme.typography.labelSmall,
                color = if (overLimit) {
                    MaterialTheme.colorScheme.error
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
            )
            Button(
                onClick = actions.askReplace,
                enabled = !recovery.submitting && replacementPlanAccepted(
                    recovery.replacementPlan,
                    recovery.replacementBytes,
                    gate.maxBytes,
                ),
            ) {
                Text("REPLACE PLAN")
            }
            if (recovery.submitting) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                    Text(
                        text = "Sending the replacement plan…",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

/** The replacement is stated before it is sent, and only then is it sent. */
@Composable
private fun RecoverPlanDialog(
    submitting: Boolean,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = { if (!submitting) onDismiss() },
        title = { Text("Replace the rejected plan?") },
        text = { Text(RECOVER_CONFIRMATION_MESSAGE) },
        confirmButton = {
            TextButton(onClick = onConfirm, enabled = !submitting) {
                Text("REPLACE PLAN & CONTINUE")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, enabled = !submitting) { Text("Keep editing") }
        },
    )
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
