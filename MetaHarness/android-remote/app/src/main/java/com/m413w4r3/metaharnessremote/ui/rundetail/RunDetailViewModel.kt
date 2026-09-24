package com.m413w4r3.metaharnessremote.ui.rundetail

import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.google.gson.JsonObject
import com.m413w4r3.metaharnessremote.api.MetaHarnessApi
import com.m413w4r3.metaharnessremote.data.ConnectionSession
import com.m413w4r3.metaharnessremote.data.ServerSettingsStore
import com.m413w4r3.metaharnessremote.network.GatewayClient
import com.m413w4r3.metaharnessremote.network.GatewayUrls
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient

/** Everything the Run Detail screen renders. */
data class RunDetailUiState(
    /** The id the screen was opened with; shown before the first reply. */
    val runId: String = "",
    /** The last detail read, kept when a later pass fails. */
    val detail: RunDetailView? = null,
    /** The progress tail read so far. */
    val progress: RunProgress = RunProgress(),
    /** One `GET /v1/runs/{runId}` pass has completed, whatever its outcome. */
    val hasLoaded: Boolean = false,
    /** Failure of the last pass, shown next to the last good detail. */
    val error: String? = null,
    /**
     * The run is terminal: polling stopped, until the screen is entered again
     * or a resume is answered.
     */
    val pollingStopped: Boolean = false,
    /** The plan approval block: what it shows and what it is doing. */
    val approval: RunApprovalUiState = RunApprovalUiState(),
    /** The scope approval block: what it shows and what it is doing. */
    val scope: RunScopeUiState = RunScopeUiState(),
    /** The plan recovery block: what it shows and what it is doing. */
    val recovery: RunRecoveryUiState = RunRecoveryUiState(),
    /** The resume block: what it shows and what it is doing. */
    val resume: RunResumeUiState = RunResumeUiState(),
    /**
     * How many resume requests have been answered. A resumed run is live
     * again, so the polling loop restarts when this changes; a terminal run
     * nobody resumed is still left alone.
     */
    val resumeAnswers: Int = 0,
)

/** Everything the plan approval block of the Run Detail screen renders. */
data class RunApprovalUiState(
    /** The gate the last run document exposed, or null when there is no block. */
    val gate: ApprovalGate? = null,
    /** The model profiles the form offers; null until they are read. */
    val profiles: ApprovalProfiles? = null,
    /** Failure of the last profile read, shown in place of the form. */
    val profilesError: String? = null,
    /** The operator's choices; null until the form is initialised. */
    val selection: ApprovalSelection? = null,
    /** The capabilities document of `GET /v1/config`, read once per screen. */
    val capabilities: JsonObject? = null,
    /** One `POST /v1/runs/{runId}/approval` is in flight: no second one starts. */
    val submitting: Boolean = false,
    /** Failure of the last decision, shown above the form. */
    val error: String? = null,
    /** The irreversible rejection waits for its confirmation. */
    val confirmingReject: Boolean = false,
)

/** Everything the scope approval block of the Run Detail screen renders. */
data class RunScopeUiState(
    /** The gate the last run document exposed, or null when there is no block. */
    val gate: ScopeGate? = null,
    /** One `POST /v1/runs/{runId}/scope-approval` is in flight: no second one starts. */
    val submitting: Boolean = false,
    /** Failure of the last decision, shown above the buttons. */
    val error: String? = null,
    /** The irreversible rejection waits for its confirmation. */
    val confirmingReject: Boolean = false,
)

/** Everything the resume block of the Run Detail screen renders. */
data class RunResumeUiState(
    /** The gate the last run document exposed, or null when there is no block. */
    val gate: ResumeGate? = null,
    /** One `POST /v1/runs/{runId}/resume` is in flight: no second one starts. */
    val submitting: Boolean = false,
    /** Failure of the last resume, shown above the button. */
    val error: String? = null,
)

/** Everything the plan recovery block of the Run Detail screen renders. */
data class RunRecoveryUiState(
    /** The gate the last run document exposed, or null when there is no block. */
    val gate: PlanRecoveryGate? = null,
    /** The replacement the operator typed, kept across passes. */
    val replacementPlan: String = "",
    /** Its UTF-8 size, counted when it was typed. */
    val replacementBytes: Long = 0L,
    /** One `POST /v1/runs/{runId}/recover-plan` is in flight: no second one starts. */
    val submitting: Boolean = false,
    /** Failure of the last replacement, shown above the form. */
    val error: String? = null,
    /** The replacement waits for the confirmation before it is sent. */
    val confirming: Boolean = false,
)

/**
 * One run: its detail, its progress, and the decisions it may wait for.
 *
 * A pass reads `GET /v1/runs/{runId}` and then
 * `GET /v1/runs/{runId}/progress?offset=N` with the offset kept here, and the
 * caller owns the delay before the next pass: [refreshOnce] returns it, so a
 * screen that is no longer visible stops polling by cancelling its loop.
 * Nothing is stored on the device.
 *
 * While the run waits for a plan decision, the pass also reads the capabilities
 * of `GET /v1/config` and the profiles of `GET /v1/model-profiles`, once per
 * screen entry, and keeps the operator's choices across passes. [approve] and
 * [confirmReject] then send at most one `POST /v1/runs/{runId}/approval` each —
 * never two at a time — and every answer, a refusal and a timeout included, is
 * followed by one more [refreshOnce]. [approveScope], [confirmScopeReject],
 * [confirmRecoverPlan] and [resume] follow the same rule for their own route.
 * Nothing is ever retried.
 */
class RunDetailViewModel(
    private val settingsStore: ServerSettingsStore,
    val runId: String,
    private val session: ConnectionSession = ConnectionSession.shared,
    private val httpClient: OkHttpClient = GatewayClient.defaultHttpClient(),
) : ViewModel() {

    var state by mutableStateOf(RunDetailUiState(runId = runId))
        private set

    /**
     * One detail and progress pass.
     *
     * Returns the delay to wait before the next pass, or null when the run is
     * terminal and polling must stop. A pass in which either read failed keeps
     * the last good detail, reports the error and retries at the slow cadence:
     * a terminal run whose events were not read yet is asked again, and a run
     * that was active when the gateway went away returns to the active cadence
     * as soon as an answer comes back.
     */
    suspend fun refreshOnce(): Long? {
        val baseUrl = when (val normalized = GatewayUrls.normalize(settingsStore.loadServerUrl())) {
            is GatewayUrls.BaseUrl.Invalid -> {
                state = state.copy(hasLoaded = true, error = normalized.reason)
                return RETRY_POLL_INTERVAL_MILLIS
            }

            is GatewayUrls.BaseUrl.Valid -> normalized.value
        }
        val token = session.remoteToken
        if (token.isBlank()) {
            state = state.copy(hasLoaded = true, error = MISSING_TOKEN)
            return RETRY_POLL_INTERVAL_MILLIS
        }

        val api = MetaHarnessApi(baseUrl, token, httpClient)
        val detail = attempt { api.getRun(runId) }
        val progress = attempt { api.progress(runId, state.progress.offset) }

        val document = detail.getOrNull()?.raw
        val read = document?.let { runDetailView(it, runId) }
        val shown = read ?: state.detail
        val gates = gatePass(api, document)
        val complete = detail.isSuccess && progress.isSuccess
        val interval = if (complete) runPollIntervalMillis(shown?.status) else RETRY_POLL_INTERVAL_MILLIS

        state = state.copy(
            detail = shown,
            progress = progress.getOrNull()?.let(state.progress::plus) ?: state.progress,
            hasLoaded = true,
            error = listOfNotNull(detail.exceptionOrNull(), progress.exceptionOrNull())
                .firstOrNull()
                ?.let(::describe),
            pollingStopped = interval == null,
            approval = gates.approval,
            scope = gates.scope,
            recovery = gates.recovery,
            resume = gates.resume,
        )
        return interval
    }

    /**
     * The operator blocks of one pass: the plan decision, the scope decision
     * and the resume.
     *
     * The run document decides whether there is a block at all; while any of
     * them is a candidate, the pass also reads the capabilities of
     * `GET /v1/config` — and, for the plan decision, the profiles of
     * `GET /v1/model-profiles` — once per screen entry, and keeps the operator's
     * choices. A pass that read no run document leaves the blocks as they were.
     */
    private suspend fun gatePass(api: MetaHarnessApi, document: JsonObject?): GatePass {
        if (document == null) return GatePass(state.approval, state.scope, state.recovery, state.resume)
        var approval = state.approval
        if (needsCapabilities(document) && approval.capabilities == null) {
            // A capabilities document that cannot be read never hides the block:
            // an explicit `false` capability does.
            approval = approval.copy(capabilities = attempt { api.config() }.getOrNull())
        }
        val capabilities = approvalCapabilities(document, approval.capabilities)
        if (awaitingPlanDecision(document) && planApprovalAvailable(capabilities) && approval.profiles == null) {
            val read = attempt { api.modelProfiles() }
            val failure = read.exceptionOrNull()
            approval = if (failure == null) {
                approval.copy(profiles = approvalProfiles(read.getOrThrow()), profilesError = null)
            } else {
                approval.copy(profilesError = describe(failure))
            }
        }
        val gate = planApprovalGate(document, capabilities)
        val selection = when {
            gate == null -> null
            approval.selection != null -> approval.selection
            approval.profiles != null -> defaultApprovalSelection(document, gate, approval.profiles)
            else -> null
        }
        return GatePass(
            approval = approval.copy(gate = gate, selection = selection),
            scope = scopePass(document, capabilities),
            recovery = recoveryPass(document, capabilities),
            resume = resumePass(document, capabilities),
        )
    }

    /** True when at least one block needs the capabilities of `GET /v1/config`. */
    private fun needsCapabilities(document: JsonObject): Boolean =
        awaitingPlanDecision(document) ||
            awaitingScopeDecision(document) ||
            resumableRun(document) ||
            planRecoverable(document)

    /** The scope block of one pass: the gate of the last run document. */
    private fun scopePass(document: JsonObject, capabilities: JsonObject?): RunScopeUiState {
        val gate = scopeApprovalGate(document, capabilities)
        val current = state.scope
        return current.copy(gate = gate, confirmingReject = current.confirmingReject && gate != null)
    }

    /** The resume block of one pass: the gate of the last run document. */
    private fun resumePass(document: JsonObject, capabilities: JsonObject?): RunResumeUiState =
        state.resume.copy(gate = resumeGate(document, capabilities))

    /**
     * The plan recovery block of one pass: the gate of the last run document,
     * and the replacement the operator typed, which every pass keeps.
     */
    private fun recoveryPass(document: JsonObject, capabilities: JsonObject?): RunRecoveryUiState {
        val gate = planRecoveryGate(document, capabilities)
        val current = state.recovery
        return current.copy(gate = gate, confirming = current.confirming && gate != null)
    }

    /** The operator chose the reviewer of the plan. */
    fun selectFinalReviewer(profileId: String) {
        changeSelection { it.copy(finalReviewerProfile = profileId) }
    }

    /** The operator chose the profile that revises the plan after validation. */
    fun selectSemanticReviser(profileId: String) {
        changeSelection { it.copy(semanticReviserProfile = profileId) }
    }

    /** The operator chose the profile that repairs a failed check. */
    fun selectCheckRepair(profileId: String) {
        changeSelection { it.copy(checkRepairProfile = profileId) }
    }

    /** The operator chose the implementer of one plan step. */
    fun selectStepProfile(stepId: String, profileId: String) {
        changeSelection { it.copy(stepProfiles = it.stepProfiles + (stepId to profileId)) }
    }

    /**
     * The operator tapped APPROVE: at most one decision leaves this screen, and
     * only when every step and role of the form holds a compatible profile.
     */
    fun approve() {
        val approval = state.approval
        val gate = approval.gate ?: return
        val profiles = approval.profiles ?: return
        val selection = approval.selection ?: return
        val payload = approvalPayload(selection, gate, profiles) ?: return
        decide(payload)
    }

    /** The operator tapped REJECT: the irreversible decision is asked first. */
    fun askReject() {
        if (state.approval.submitting) return
        state = state.copy(approval = state.approval.copy(confirmingReject = true, error = null))
    }

    /** The operator closed the rejection dialog without deciding. */
    fun dismissReject() {
        state = state.copy(approval = state.approval.copy(confirmingReject = false))
    }

    /** The operator confirmed the irreversible rejection: one `REJECT` is sent. */
    fun confirmReject() {
        decide(rejectionPayload())
    }

    /** One `POST /v1/runs/{runId}/approval`; nothing is sent while one is in flight. */
    private fun decide(payload: JsonObject) {
        val approval = state.approval
        if (approval.submitting) return
        when (val baseUrl = GatewayUrls.normalize(settingsStore.loadServerUrl())) {
            is GatewayUrls.BaseUrl.Invalid ->
                state = state.copy(approval = approval.copy(error = baseUrl.reason, confirmingReject = false))

            is GatewayUrls.BaseUrl.Valid -> {
                val token = session.remoteToken
                if (token.isBlank()) {
                    state = state.copy(approval = approval.copy(error = MISSING_TOKEN, confirmingReject = false))
                } else {
                    state = state.copy(
                        approval = approval.copy(submitting = true, error = null, confirmingReject = false),
                    )
                    viewModelScope.launch { send(MetaHarnessApi(baseUrl.value, token, httpClient), payload) }
                }
            }
        }
    }

    /**
     * One decision, reported as it ends.
     *
     * The run is read again whatever the answer was: an accepted decision moved
     * it, and a refusal means another decision is already recorded. The decision
     * itself is never sent twice. [CancellationException] belongs to the scope:
     * the screen was left, so its answer is shown nowhere.
     */
    private suspend fun send(api: MetaHarnessApi, payload: JsonObject) {
        val outcome = attempt { api.approveRun(runId, payload) }
        state = state.copy(
            approval = state.approval.copy(
                submitting = false,
                error = outcome.exceptionOrNull()?.let(::approvalFailure),
            ),
        )
        refreshOnce()
    }

    /** The operator approved the requested repair scope: one `APPROVE` is sent. */
    fun approveScope() {
        decideScope(APPROVE)
    }

    /** The operator tapped REJECT: the irreversible decision is asked first. */
    fun askScopeReject() {
        if (state.scope.submitting) return
        state = state.copy(scope = state.scope.copy(confirmingReject = true, error = null))
    }

    /** The operator closed the scope rejection dialog without deciding. */
    fun dismissScopeReject() {
        state = state.copy(scope = state.scope.copy(confirmingReject = false))
    }

    /** The operator confirmed the irreversible rejection: one `REJECT` is sent. */
    fun confirmScopeReject() {
        decideScope(REJECT)
    }

    /**
     * One `POST /v1/runs/{runId}/scope-approval`; nothing is sent while one is
     * in flight. The decision is the operator's, and the exact delta stays the
     * one the run recorded.
     */
    private fun decideScope(decision: String) {
        val scope = state.scope
        if (scope.submitting) return
        when (val baseUrl = GatewayUrls.normalize(settingsStore.loadServerUrl())) {
            is GatewayUrls.BaseUrl.Invalid ->
                state = state.copy(scope = scope.copy(error = baseUrl.reason, confirmingReject = false))

            is GatewayUrls.BaseUrl.Valid -> {
                val token = session.remoteToken
                if (token.isBlank()) {
                    state = state.copy(scope = scope.copy(error = MISSING_TOKEN, confirmingReject = false))
                } else {
                    state = state.copy(
                        scope = scope.copy(submitting = true, error = null, confirmingReject = false),
                    )
                    viewModelScope.launch {
                        sendScope(MetaHarnessApi(baseUrl.value, token, httpClient), decision)
                    }
                }
            }
        }
    }

    /**
     * One scope decision, reported as it ends.
     *
     * The run is read again whatever the answer was: an accepted decision moved
     * it, and a refusal means another decision is already recorded. The decision
     * itself is never sent twice. [CancellationException] belongs to the scope:
     * the screen was left, so its answer is shown nowhere.
     */
    private suspend fun sendScope(api: MetaHarnessApi, decision: String) {
        // A scope decision fails exactly like a plan decision: a refusal, a
        // conflict or a timeout, and the refreshed run is what speaks next.
        val outcome = attempt { api.approveScope(runId, decision) }
        state = state.copy(
            scope = state.scope.copy(
                submitting = false,
                error = outcome.exceptionOrNull()?.let(::approvalFailure),
            ),
        )
        refreshOnce()
    }

    /**
     * The operator asked to resume the run: one empty `POST` is sent, and
     * nothing is sent while one is in flight. A resume that timed out may have
     * been accepted, so it is never sent again by the screen.
     */
    fun resume() {
        val resume = state.resume
        if (resume.submitting) return
        when (val baseUrl = GatewayUrls.normalize(settingsStore.loadServerUrl())) {
            is GatewayUrls.BaseUrl.Invalid ->
                state = state.copy(resume = resume.copy(error = baseUrl.reason))

            is GatewayUrls.BaseUrl.Valid -> {
                val token = session.remoteToken
                if (token.isBlank()) {
                    state = state.copy(resume = resume.copy(error = MISSING_TOKEN))
                } else {
                    state = state.copy(resume = resume.copy(submitting = true, error = null))
                    viewModelScope.launch { sendResume(MetaHarnessApi(baseUrl.value, token, httpClient)) }
                }
            }
        }
    }

    /**
     * One resume, reported as it ends.
     *
     * The run is read again whatever the answer was — a timeout may have been
     * accepted, and the refreshed run, not a second resume, tells what happened.
     */
    private suspend fun sendResume(api: MetaHarnessApi) {
        val outcome = attempt { api.resumeRun(runId) }
        state = state.copy(
            resume = state.resume.copy(
                submitting = false,
                error = outcome.exceptionOrNull()?.let(::resumeFailure),
            ),
            resumeAnswers = state.resumeAnswers + 1,
        )
        refreshOnce()
    }

    /**
     * The operator typed in the replacement field. Nothing is sent: the byte
     * count follows the text, the previous failure is dropped, and a
     * confirmation already given no longer covers what the field now holds.
     */
    fun updateReplacementPlan(plan: String) {
        val recovery = state.recovery
        if (recovery.submitting) return
        state = state.copy(
            recovery = recovery.copy(
                replacementPlan = plan,
                replacementBytes = replacementPlanBytes(plan),
                error = null,
                confirming = false,
            ),
        )
    }

    /**
     * The operator tapped REPLACE PLAN: the confirmation is asked first, and
     * only for a replacement that holds text and fits the published bound.
     */
    fun askRecoverPlan() {
        val recovery = state.recovery
        if (recovery.submitting || !recoveryReady()) return
        state = state.copy(recovery = recovery.copy(confirming = true, error = null))
    }

    /** The operator closed the confirmation without sending anything. */
    fun dismissRecoverPlan() {
        state = state.copy(recovery = state.recovery.copy(confirming = false))
    }

    /**
     * The operator confirmed the replacement: exactly one
     * `POST /v1/runs/{runId}/recover-plan` is sent, and nothing is sent while
     * one is in flight.
     */
    fun confirmRecoverPlan() {
        val recovery = state.recovery
        if (recovery.submitting || !recoveryReady()) return
        when (val baseUrl = GatewayUrls.normalize(settingsStore.loadServerUrl())) {
            is GatewayUrls.BaseUrl.Invalid ->
                state = state.copy(recovery = recovery.copy(error = baseUrl.reason, confirming = false))

            is GatewayUrls.BaseUrl.Valid -> {
                val token = session.remoteToken
                if (token.isBlank()) {
                    state = state.copy(recovery = recovery.copy(error = MISSING_TOKEN, confirming = false))
                } else {
                    state = state.copy(
                        recovery = recovery.copy(submitting = true, error = null, confirming = false),
                    )
                    viewModelScope.launch {
                        sendRecovery(
                            MetaHarnessApi(baseUrl.value, token, httpClient),
                            recovery.replacementPlan,
                        )
                    }
                }
            }
        }
    }

    /**
     * One replacement plan, reported as it ends.
     *
     * The run is read again whatever the answer was — a timeout may have been
     * accepted, and the refreshed run, not a second replacement, tells what
     * happened. The plan itself is never sent twice.
     */
    private suspend fun sendRecovery(api: MetaHarnessApi, plan: String) {
        val outcome = attempt { api.recoverPlan(runId, plan) }
        state = state.copy(
            recovery = state.recovery.copy(
                submitting = false,
                error = outcome.exceptionOrNull()?.let(::recoverFailure),
            ),
        )
        refreshOnce()
    }

    /** True when the replacement the operator typed may be sent. */
    private fun recoveryReady(): Boolean {
        val recovery = state.recovery
        val maxBytes = recovery.gate?.maxBytes ?: return false
        return replacementPlanAccepted(recovery.replacementPlan, recovery.replacementBytes, maxBytes)
    }

    private fun changeSelection(change: (ApprovalSelection) -> ApprovalSelection) {
        val approval = state.approval
        val selection = approval.selection ?: return
        state = state.copy(approval = approval.copy(selection = change(selection), error = null))
    }

    /** [CancellationException] belongs to the scope, not to the gateway. */
    private suspend fun <T> attempt(block: suspend () -> T): Result<T> =
        try {
            Result.success(block())
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (failure: Exception) {
            Result.failure(failure)
        }

    /** A one-line reason for a failed exchange; the API never puts the token in one. */
    private fun describe(failure: Throwable): String =
        failure.message?.trim()?.takeIf { it.isNotEmpty() } ?: failure.javaClass.simpleName

    /** What one pass found for the four operator blocks. */
    private data class GatePass(
        val approval: RunApprovalUiState,
        val scope: RunScopeUiState,
        val recovery: RunRecoveryUiState,
        val resume: RunResumeUiState,
    )

    companion object {
        private const val MISSING_TOKEN = "Remote token required: set it in Settings"

        fun factory(context: Context, runId: String): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                RunDetailViewModel(ServerSettingsStore(context.applicationContext), runId)
            }
        }
    }
}
