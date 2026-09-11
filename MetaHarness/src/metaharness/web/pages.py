"""Small server-rendered HTML pages for the local MetaHarness UI.

Every artifact value (plan, review, checks, paths, failures) is untrusted:
server-side it is escaped with :func:`html.escape`; client-side the polling
script only writes it through ``textContent`` or nodes created explicitly,
never ``innerHTML``.  The inline ``<style>``/``<script>`` blocks carry the
per-response CSP nonce set by the server.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

from ..models import HarnessConfig
from ..profiles import profiles_for_config, safe_profile_metadata


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _json_script(value: Any) -> str:
    # Run IDs and tokens are validated/generated values. Escaping the two HTML
    # delimiters also keeps this safe if the helper is reused for other data.
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _page(title: str, body: str, script: str = "", *, nonce: str | None = None) -> str:
    nonce_attribute = f' nonce="{_e(nonce)}"' if nonce else ""
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="referrer" content="no-referrer">
  <title>{_e(title)} · MetaHarness</title>
  <style{nonce_attribute}>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 1180px; margin: 0 auto; padding: 1rem; line-height: 1.4; }}
    a {{ color: #6aa9ff; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem; }}
    th, td {{ text-align: left; border-bottom: 1px solid #7776; padding: .55rem; vertical-align: top; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; padding: 1rem; border: 1px solid #7776; border-radius: .4rem; background: #7772; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .7rem; }}
    .card {{ border: 1px solid #7776; border-radius: .4rem; padding: .8rem; }}
    .timeline {{ list-style: none; padding: 0; max-width: 32rem; }}
    .timeline li {{ border-left: 3px solid #7778; padding: .35rem .8rem; margin: 0; }}
    .timeline li.current {{ border-color: #4cae4c; font-weight: 700; }}
    .timeline li.done {{ opacity: .72; }}
    button {{ padding: .55rem .8rem; margin-right: .5rem; cursor: pointer; }}
    button:disabled {{ cursor: default; opacity: .55; }}
    .danger {{ color: #ff8d8d; }}
    .muted {{ opacity: .7; }}
    dt {{ font-weight: 700; }} dd {{ margin: 0 0 .5rem; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
{body}
{(f"<script{nonce_attribute}>" + script + "</script>") if script else ""}
</body>
</html>
"""


def _failure(value: Any) -> str:
    if not value:
        return '<span class="muted">—</span>'
    if isinstance(value, dict):
        reason = _e(value.get("reason"))
        detail = value.get("detail")
        suffix = f" — {_e(detail)}" if detail is not None else ""
        return f'<span class="danger">{reason}{suffix}</span>'
    return f'<span class="danger">{_e(value)}</span>'


def render_index(runs: list[dict[str, Any]], *, nonce: str | None = None) -> str:
    """Render the run list.  It has no mutation, hence no mutation token."""

    rows = []
    for run in runs:
        run_id = run.get("run_id")
        rows.append(
            "<tr>"
            f'<td><a href="/runs/{_e(run_id)}">{_e(run_id)}</a></td>'
            f"<td>{_e(run.get('status'))}</td>"
            f"<td>{_e(run.get('updated_at'))}</td>"
            f"<td>{_e(run.get('plan_title'))}</td>"
            f"<td>{_e(run.get('commit_sha'))}</td>"
            f"<td>{_failure(run.get('failure'))}</td>"
            "</tr>"
        )
    table = "".join(rows) or '<tr><td colspan="6" class="muted">Aucun run.</td></tr>'
    script = """
const tableBody = document.querySelector('#runs-body');
function cell(value) { const node = document.createElement('td'); node.textContent = value == null ? '—' : String(value); return node; }
function refreshRuns() {
  fetch('/api/runs', { cache: 'no-store' }).then(response => response.json()).then(payload => {
    while (tableBody.firstChild) tableBody.removeChild(tableBody.firstChild);
    for (const run of (payload.runs || [])) {
      const row = document.createElement('tr');
      const linkCell = document.createElement('td');
      const link = document.createElement('a');
      link.href = '/runs/' + encodeURIComponent(String(run.run_id));
      link.textContent = run.run_id == null ? '—' : String(run.run_id);
      linkCell.appendChild(link); row.appendChild(linkCell);
      for (const key of ['status', 'updated_at', 'plan_title', 'commit_sha', 'failure']) row.appendChild(cell(key === 'failure' && run[key] && typeof run[key] === 'object' ? run[key].reason : run[key]));
      tableBody.appendChild(row);
    }
  }).catch(() => {});
}
setInterval(refreshRuns, 2000);
"""
    body = f"""
<header><h1>MetaHarness</h1><p class="muted">Observation locale des runs</p></header>
<p><a href="/new">NEW RUN</a></p>
<table>
  <thead><tr><th>RUN ID</th><th>STATUS</th><th>UPDATED</th><th>PLAN TITLE</th><th>COMMIT</th><th>FAILURE</th></tr></thead>
  <tbody id="runs-body">{table}</tbody>
</table>
"""
    return _page("Runs", body, script, nonce=nonce)


def render_new_run(
    config: HarnessConfig,
    token: str,
    *,
    nonce: str | None = None,
) -> str:
    profiles = [
        safe_profile_metadata(profile)
        for profile in profiles_for_config(config).values()
        if any(role.value == "planner" for role in profile.roles)
    ]
    planner_options = "".join(
        f'<option value="{_e(profile["id"])}"{" selected" if profile["id"] == config.ui.default_planner_profile else ""}>{_e(profile["display_name"])}</option>'
        for profile in profiles
    )
    script = f"""
const META_TOKEN = {_json_script(token)};
const form = document.getElementById('new-run-form');
const button = document.getElementById('create-run');
const message = document.getElementById('create-run-message');
form.addEventListener('submit', async (event) => {{
  event.preventDefault();
  button.disabled = true;
  message.textContent = '';
  try {{
    const response = await fetch('/api/runs', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json', 'X-MetaHarness-Token': META_TOKEN }},
      body: JSON.stringify({{ spec: document.getElementById('spec').value, run_id: document.getElementById('run-id').value || null, planner_profile: document.getElementById('planner-profile').value }})
    }});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || 'Unable to create run.');
    window.location.assign(payload.location);
  }} catch (error) {{
    message.textContent = error instanceof Error ? error.message : 'Unable to create run.';
    button.disabled = false;
  }}
}});
"""
    body = f"""
<main>
<p><a href="/">← Tous les runs</a></p>
<h1>New Run</h1>
<dl>
  <dt>Repository</dt><dd><input type="text" value="{_e(config.repo)}" readonly></dd>
  <dt>Base ref</dt><dd><input type="text" value="{_e(config.base_ref)}" readonly></dd>
</dl>
<form id="new-run-form">
  <label for="spec">SPEC</label><br>
  <textarea id="spec" rows="20" cols="100" required></textarea><br>
  <label for="run-id">Run ID (optional)</label><br>
  <input id="run-id" type="text" autocomplete="off"><br>
  <label for="planner-profile">Planner</label><br>
  <select id="planner-profile">{planner_options}</select><br><br>
  <button id="create-run" type="submit">CREATE RUN</button>
  <span id="create-run-message" class="danger" role="status"></span>
</form>
</main>
"""
    return _page("New Run", body, script, nonce=nonce)


_TIMELINE = (
    ("created", "CREATED"),
    ("planning", "PLANNING"),
    ("awaiting_plan_approval", "AWAITING PLAN APPROVAL"),
    ("worktree_ready", "WORKTREE"),
    ("implementing", "IMPLEMENTING"),
    ("validating", "VALIDATING"),
    ("reviewing", "REVIEWING"),
    ("committed", "COMMITTED"),
)
_ORDER = {value: index for index, (value, _label) in enumerate(_TIMELINE)}
_TERMINAL_LABELS = {
    "blocked": "BLOCKED",
    "plan_rejected": "REJECTED",
    "failed": "FAILED",
    "interrupted": "INTERRUPTED",
}
# Statuses after which the run state can no longer change.
TERMINAL_STATUSES = frozenset({"committed", *_TERMINAL_LABELS})
AWAITING_APPROVAL_STATUS = "awaiting_plan_approval"

# Elements the run-page polling script updates from ``GET /api/runs/<id>``.
RUN_PAGE_DYNAMIC_IDS = (
    "run-status",
    "run-updated",
    "approval-actions",
    "approve",
    "reject",
    "approval-message",
    "model-selection-warning",
    "base-sha",
    "branch",
    "worktree",
    "commit-sha",
    "failure",
    "timeline",
    "plan-contract",
    "plan-raw",
    "checks",
    "review",
    "reviewer-raw-state",
    "reviewer-raw",
    "spec",
)


def _timeline_items(status: Any) -> str:
    current = str(status or "")
    terminal_label = _TERMINAL_LABELS.get(current)
    current_order = _ORDER.get(current, -1)
    items = []
    for value, label in _TIMELINE:
        classes = []
        if value == current:
            classes.append("current")
        elif current_order >= 0 and _ORDER.get(value, 99) < current_order:
            classes.append("done")
        items.append(f'<li class="{" ".join(classes)}">{label}</li>')
    if terminal_label:
        items.append(f'<li class="current danger">{terminal_label}</li>')
    return "".join(items)


def _check_cards(checks: Any) -> str:
    if not checks:
        return '<p class="muted">Aucun check.</p>'
    if not isinstance(checks, list):
        return f"<pre>{_e(checks)}</pre>"
    cards = []
    for check in checks:
        if not isinstance(check, dict):
            cards.append(f"<pre>{_e(check)}</pre>")
            continue
        cards.append(
            '<article class="card">'
            f"<strong>{_e(check.get('name'))}</strong>"
            f"<dl><dt>exit_code</dt><dd>{_e(check.get('exit_code'))}</dd>"
            f"<dt>timed_out</dt><dd>{_e(check.get('timed_out'))}</dd>"
            f"<dt>duration</dt><dd>{_e(check.get('duration_seconds', check.get('duration')))}</dd>"
            f"<dt>workspace_mutated</dt><dd>{_e(check.get('workspace_mutated'))}</dd></dl>"
            f"<p>stdout tail</p><pre>{_e(check.get('stdout_tail'))}</pre>"
            f"<p>stderr tail</p><pre>{_e(check.get('stderr_tail'))}</pre>"
            "</article>"
        )
    return '<div class="grid">' + "".join(cards) + "</div>"


def _review(review: Any) -> str:
    if not review:
        return '<p class="muted">Aucune review.</p>'
    if not isinstance(review, dict):
        return f"<pre>{_e(review)}</pre>"
    def value(underscore: str, spaced: str | None = None) -> Any:
        return review.get(underscore, review.get(spaced)) if spaced else review.get(underscore)

    fields = (
        ("verdict", value("verdict")),
        ("route", value("route")),
        ("summary", value("summary")),
        ("findings", value("findings")),
        ("required fixes", value("required_fixes", "required fixes")),
        ("missing tests", value("missing_tests", "missing tests")),
        ("residual risks", value("residual_risks", "residual risks")),
    )
    return "<dl>" + "".join(f"<dt>{_e(label)}</dt><dd>{_e(value)}</dd>" for label, value in fields) + "</dl>"


def _reviewer_raw_state(available: bool) -> str:
    return "reviewer.raw.md disponible" if available else "reviewer.raw.md non disponible"


_RUN_SCRIPT = """
const page = document.getElementById('run-page');
let decisionSent = false;
let reloadRequested = false;
let statePoll = null;
const rendered = {};
function byId(id) { return document.getElementById(id); }
function display(value) { if (value == null) return ''; return typeof value === 'object' ? JSON.stringify(value) : String(value); }
function setText(id, value) { const node = byId(id); if (node) node.textContent = value == null || value === '' ? '—' : display(value); }
function setPre(id, value) { const node = byId(id); if (node) node.textContent = display(value); }
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
function el(tag, text, className) { const node = document.createElement(tag); if (text !== undefined) node.textContent = display(text); if (className) node.className = className; return node; }
function changed(key, value) { const signature = JSON.stringify(value === undefined ? null : value); if (rendered[key] === signature) return false; rendered[key] = signature; return true; }
function renderFailure(failure) {
  const node = byId('failure'); clear(node);
  if (!failure) { node.appendChild(el('span', '—', 'muted')); return; }
  const text = typeof failure === 'object' ? display(failure.reason) + (failure.detail != null ? ' — ' + display(failure.detail) : '') : display(failure);
  node.appendChild(el('span', text, 'danger'));
}
function renderTimeline(status) {
  const list = byId('timeline'); clear(list);
  const order = TIMELINE.findIndex(item => item[0] === status);
  TIMELINE.forEach((item, index) => {
    const entry = el('li', item[1]);
    if (item[0] === status) entry.className = 'current'; else if (order >= 0 && index < order) entry.className = 'done';
    list.appendChild(entry);
  });
  if (Object.prototype.hasOwnProperty.call(TERMINAL_LABELS, status)) list.appendChild(el('li', TERMINAL_LABELS[status], 'current danger'));
}
function renderChecks(checks) {
  const root = byId('checks'); clear(root);
  if (!checks || (Array.isArray(checks) && checks.length === 0)) { root.appendChild(el('p', 'Aucun check.', 'muted')); return; }
  if (!Array.isArray(checks)) { root.appendChild(el('pre', checks)); return; }
  const grid = el('div', undefined, 'grid');
  for (const check of checks) {
    if (!check || typeof check !== 'object') { grid.appendChild(el('pre', check)); continue; }
    const card = el('article', undefined, 'card');
    card.appendChild(el('strong', check.name));
    const list = document.createElement('dl');
    const duration = check.duration_seconds !== undefined ? check.duration_seconds : check.duration;
    for (const [label, value] of [['exit_code', check.exit_code], ['timed_out', check.timed_out], ['duration', duration], ['workspace_mutated', check.workspace_mutated]]) {
      list.appendChild(el('dt', label)); list.appendChild(el('dd', value));
    }
    card.appendChild(list);
    card.appendChild(el('p', 'stdout tail')); card.appendChild(el('pre', check.stdout_tail));
    card.appendChild(el('p', 'stderr tail')); card.appendChild(el('pre', check.stderr_tail));
    grid.appendChild(card);
  }
  root.appendChild(grid);
}
function renderReview(review) {
  const root = byId('review'); clear(root);
  if (!review) { root.appendChild(el('p', 'Aucune review.', 'muted')); return; }
  if (typeof review !== 'object' || Array.isArray(review)) { root.appendChild(el('pre', review)); return; }
  const pick = (a, b) => review[a] !== undefined ? review[a] : review[b];
  const list = document.createElement('dl');
  for (const [label, value] of [['verdict', review.verdict], ['route', review.route], ['summary', review.summary], ['findings', review.findings], ['required fixes', pick('required_fixes', 'required fixes')], ['missing tests', pick('missing_tests', 'missing tests')], ['residual risks', pick('residual_risks', 'residual risks')]]) {
    list.appendChild(el('dt', label)); list.appendChild(el('dd', value));
  }
  root.appendChild(list);
}
function updateApproval(status) {
  const awaiting = status === AWAITING_STATUS;
  if (awaiting && META_TOKEN === null) {
    // This page was rendered before the approval gate, without the mutation
    // token: reload once so the server renders the decision controls.
    if (!reloadRequested) { reloadRequested = true; window.location.reload(); }
    return;
  }
  byId('approval-actions').hidden = !awaiting;
  const disabled = !awaiting || decisionSent || META_TOKEN === null;
  byId('approve').disabled = disabled; byId('reject').disabled = disabled;
}
function renderSelectionWarning(run) {
  const node = byId('model-selection-warning'); if (!node) return;
  const execution = run && run.state && run.state.execution ? run.state.execution : {};
  const modes = [execution.planner, execution.implementer, execution.reviewer].filter(Boolean).map(item => item.selection_mode);
  node.hidden = !modes.includes('external-ui');
  node.textContent = node.hidden ? '' : 'Model selection is external. MetaHarness cannot force or verify the actual model selected in the provider UI.';
}
function applyRun(run) {
  if (!run || typeof run !== 'object') return;
  const state = run.state && typeof run.state === 'object' ? run.state : {};
  const status = state.status !== undefined ? state.status : run.status;
  page.dataset.status = display(status);
  page.dataset.updatedAt = display(run.updated_at);
  setText('run-status', status);
  setText('run-updated', run.updated_at);
  setText('base-sha', state.base_sha);
  setText('branch', state.branch);
  setText('worktree', state.worktree);
  setText('commit-sha', state.commit_sha !== undefined ? state.commit_sha : run.commit_sha);
  if (changed('failure', run.failure)) renderFailure(run.failure);
  if (changed('status', status)) renderTimeline(status);
  updateApproval(status);
  renderSelectionWarning(run);
  const plan = run.plan && typeof run.plan === 'object' ? run.plan : {};
  if (changed('contract', plan.contract)) setPre('plan-contract', plan.contract);
  if (changed('raw', plan.raw)) setPre('plan-raw', plan.raw);
  if (changed('spec', run.spec)) setPre('spec', run.spec);
  if (changed('checks', run.checks)) renderChecks(run.checks);
  if (changed('review', run.review)) renderReview(run.review);
  byId('reviewer-raw-state').textContent = run.reviewer_raw_available ? 'reviewer.raw.md disponible' : 'reviewer.raw.md non disponible';
  if (changed('reviewer_raw', run.reviewer_raw)) setPre('reviewer-raw', run.reviewer_raw);
  if (TERMINAL_STATUSES.includes(status) && statePoll !== null) { clearInterval(statePoll); statePoll = null; }
}
function pollRun() {
  fetch('/api/runs/' + encodeURIComponent(RUN_ID), { cache: 'no-store' })
    .then(response => response.ok ? response.json() : null)
    .then(applyRun)
    .catch(() => {});
}
let progressOffset = 0;
function addProgress(value) { const item = document.createElement('li'); item.textContent = value; byId('progress-events').appendChild(item); }
function pollProgress() {
  fetch('/api/runs/' + encodeURIComponent(RUN_ID) + '/progress?offset=' + progressOffset, { cache: 'no-store' }).then(response => response.json()).then(payload => {
    progressOffset = Number(payload.next_offset || progressOffset);
    for (const value of (payload.events || [])) addProgress(value);
  }).catch(() => {});
}
function decide(decision) {
  if (META_TOKEN === null) return;
  byId('approve').disabled = true; byId('reject').disabled = true;
  const body = decision === 'APPROVE' ? { decision, implementer_profile: byId('implementer-profile').value, reviewer_profile: byId('reviewer-profile').value } : { decision };
  fetch('/api/runs/' + encodeURIComponent(RUN_ID) + '/approval', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-MetaHarness-Token': META_TOKEN }, body: JSON.stringify(body) })
    .then(response => response.json().then(payload => ({ ok: response.ok, payload })))
    .then(result => { if (result.ok) decisionSent = true; byId('approval-message').textContent = result.ok ? ' Décision enregistrée.' : ' ' + display(result.payload.message || 'Erreur.'); pollRun(); })
    .catch(() => { byId('approval-message').textContent = ' Erreur réseau.'; });
}
byId('approve').addEventListener('click', () => decide('APPROVE'));
byId('reject').addEventListener('click', () => decide('REJECT'));
pollProgress(); setInterval(pollProgress, 1000);
if (!TERMINAL_STATUSES.includes(page.dataset.status)) { statePoll = setInterval(pollRun, STATE_POLL_MS); }
"""
STATE_POLL_MS = 2000


def render_run(
    run: dict[str, Any],
    token: str | None = None,
    *,
    config: HarnessConfig | None = None,
    nonce: str | None = None,
) -> str:
    """Render one run.

    The mutation token is embedded only while the run awaits plan approval:
    no other page can send a decision.  A page rendered earlier reloads once
    when polling observes the approval gate.
    """

    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
    status = state.get("status", run.get("status"))
    run_id = run.get("run_id")
    can_decide = status == AWAITING_APPROVAL_STATUS and bool(token)
    page_token = token if can_decide else None
    disabled = "" if can_decide else " disabled"
    hidden = "" if can_decide else " hidden"
    reviewer_raw = run.get("reviewer_raw")
    execution = state.get("execution") if isinstance(state.get("execution"), dict) else {}
    external_selection = any(
        isinstance(execution.get(role), dict)
        and execution[role].get("selection_mode") == "external-ui"
        for role in ("planner", "implementer", "reviewer")
    )
    warning_text = (
        "Model selection is external. MetaHarness cannot force or verify the actual model selected in the provider UI."
        if external_selection else ""
    )
    profile_metadata = {
        role: [
            safe_profile_metadata(profile)
            for profile in profiles_for_config(config).values()
            if any(item.value == role for item in profile.roles)
        ]
        for role in ("implementer", "reviewer")
    } if config is not None else {"implementer": [], "reviewer": []}
    defaults = {
        "implementer": config.ui.default_implementer_profile if config else None,
        "reviewer": config.ui.default_reviewer_profile if config else None,
    }
    def options(role: str) -> str:
        return "".join(
            f'<option value="{_e(item["id"])}"{" selected" if item["id"] == defaults[role] else ""}>{_e(item["display_name"])}</option>'
            for item in profile_metadata[role]
        )
    implementer_options = options("implementer")
    reviewer_options = options("reviewer")
    script = (
        f"const META_TOKEN = {_json_script(page_token)};\n"
        f"const RUN_ID = {_json_script(run_id)};\n"
        f"const TIMELINE = {_json_script([list(item) for item in _TIMELINE])};\n"
        f"const TERMINAL_LABELS = {_json_script(_TERMINAL_LABELS)};\n"
        f"const TERMINAL_STATUSES = {_json_script(sorted(TERMINAL_STATUSES))};\n"
        f"const AWAITING_STATUS = {_json_script(AWAITING_APPROVAL_STATUS)};\n"
        f"const STATE_POLL_MS = {STATE_POLL_MS};\n"
        + _RUN_SCRIPT
    )
    body = f"""
<main id="run-page" data-run-id="{_e(run_id)}" data-status="{_e(status)}" data-updated-at="{_e(run.get('updated_at'))}">
<p><a href="/">← Tous les runs</a></p>
<header><h1>Run {_e(run_id)}</h1><p>Status : <strong id="run-status">{_e(status)}</strong> · <span class="muted">mis à jour <span id="run-updated">{_e(run.get('updated_at'))}</span></span></p></header>
<div id="approval-actions"{hidden}><label for="implementer-profile">Implementer</label> <select id="implementer-profile">{implementer_options}</select> <label for="reviewer-profile">Reviewer</label> <select id="reviewer-profile">{reviewer_options}</select> <button id="approve" type="button"{disabled}>APPROVE PLAN</button><button id="reject" type="button"{disabled}>REJECT PLAN</button><span id="approval-message"></span></div>
<p id="model-selection-warning" class="danger"{"" if external_selection else " hidden"}>{_e(warning_text)}</p>
<section><h2>Header</h2><div class="grid">
  <div class="card"><strong>Base SHA</strong><br><span id="base-sha">{_e(state.get('base_sha'))}</span></div>
  <div class="card"><strong>Branch</strong><br><span id="branch">{_e(state.get('branch'))}</span></div>
  <div class="card"><strong>Worktree</strong><br><span id="worktree">{_e(state.get('worktree'))}</span></div>
  <div class="card"><strong>Commit SHA</strong><br><span id="commit-sha">{_e(state.get('commit_sha'))}</span></div>
  <div class="card"><strong>Failure</strong><br><span id="failure">{_failure(state.get('failure'))}</span></div>
</div></section>
<section><h2>Timeline</h2><ul class="timeline" id="timeline">{_timeline_items(status)}</ul></section>
<section><h2>Plan</h2><h3>Canonical implementation contract</h3><pre id="plan-contract">{_e(plan.get('contract'))}</pre><h3>planner.raw.md</h3><pre id="plan-raw">{_e(plan.get('raw'))}</pre></section>
<section><details><summary>SPEC</summary><pre id="spec">{_e(run.get('spec'))}</pre></details></section>
<section><h2>Progress Codex</h2><ul id="progress-events"></ul></section>
<section><h2>Checks</h2><div id="checks">{_check_cards(run.get('checks'))}</div></section>
<section><h2>Review</h2><div id="review">{_review(run.get('review'))}</div></section>
<section><h2>Reviewer raw</h2><p id="reviewer-raw-state" class="muted">{_reviewer_raw_state(reviewer_raw is not None)}</p><details><summary>Déplier reviewer.raw.md</summary><pre id="reviewer-raw">{_e(reviewer_raw)}</pre></details></section>
</main>
"""
    return _page(f"Run {run_id}", body, script, nonce=nonce)


__all__ = [
    "RUN_PAGE_DYNAMIC_IDS",
    "STATE_POLL_MS",
    "TERMINAL_STATUSES",
    "render_index",
    "render_new_run",
    "render_run",
]
