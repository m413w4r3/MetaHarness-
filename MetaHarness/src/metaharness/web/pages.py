"""Small server-rendered HTML pages for the local MetaHarness UI."""

from __future__ import annotations

import json
from html import escape
from typing import Any


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


def _page(title: str, body: str, script: str = "") -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)} · MetaHarness</title>
  <style>
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
{("<script>" + script + "</script>") if script else ""}
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


def render_index(runs: list[dict[str, Any]], token: str) -> str:
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
    script = f"""
const META_TOKEN = {_json_script(token)};
const tableBody = document.querySelector('#runs-body');
function cell(value) {{ const node = document.createElement('td'); node.textContent = value == null ? '—' : String(value); return node; }}
function refreshRuns() {{
  fetch('/api/runs').then(response => response.json()).then(payload => {{
    while (tableBody.firstChild) tableBody.removeChild(tableBody.firstChild);
    for (const run of (payload.runs || [])) {{
      const row = document.createElement('tr');
      const linkCell = document.createElement('td');
      const link = document.createElement('a');
      link.href = '/runs/' + encodeURIComponent(String(run.run_id));
      link.textContent = run.run_id == null ? '—' : String(run.run_id);
      linkCell.appendChild(link); row.appendChild(linkCell);
      for (const key of ['status', 'updated_at', 'plan_title', 'commit_sha', 'failure']) row.appendChild(cell(key === 'failure' && run[key] && typeof run[key] === 'object' ? run[key].reason : run[key]));
      tableBody.appendChild(row);
    }}
  }}).catch(() => {{}});
}}
setInterval(refreshRuns, 2000);
"""
    body = f"""
<header><h1>MetaHarness</h1><p class="muted">Observation locale des runs</p></header>
<table>
  <thead><tr><th>RUN ID</th><th>STATUS</th><th>UPDATED</th><th>PLAN TITLE</th><th>COMMIT</th><th>FAILURE</th></tr></thead>
  <tbody id="runs-body">{table}</tbody>
</table>
"""
    return _page("Runs", body, script)


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


def _timeline(status: Any) -> str:
    current = str(status or "")
    terminal_label = {
        "blocked": "BLOCKED",
        "plan_rejected": "REJECTED",
        "failed": "FAILED",
        "interrupted": "INTERRUPTED",
    }.get(current)
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
    return "<ul class=" + '"timeline">' + "".join(items) + "</ul>"


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


def render_run(run: dict[str, Any], token: str) -> str:
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
    status = state.get("status", run.get("status"))
    run_id = run.get("run_id")
    can_decide = status == "awaiting_plan_approval"
    buttons = (
        '<div id="approval-actions"><button id="approve" type="button">APPROVE PLAN</button>'
        '<button id="reject" type="button">REJECT PLAN</button><span id="approval-message"></span></div>'
        if can_decide
        else ""
    )
    script = f"""
const META_TOKEN = {_json_script(token)};
const RUN_ID = {_json_script(run_id)};
let progressOffset = 0;
function addProgress(value) {{ const item = document.createElement('li'); item.textContent = value; document.querySelector('#progress-events').appendChild(item); }}
function pollProgress() {{
  fetch('/api/runs/' + encodeURIComponent(RUN_ID) + '/progress?offset=' + progressOffset).then(response => response.json()).then(payload => {{
    progressOffset = Number(payload.next_offset || progressOffset);
    for (const value of (payload.events || [])) addProgress(value);
  }}).catch(() => {{}});
}}
function decide(decision) {{
  const approve = document.querySelector('#approve'); const reject = document.querySelector('#reject');
  if (approve) approve.disabled = true; if (reject) reject.disabled = true;
  fetch('/api/runs/' + encodeURIComponent(RUN_ID) + '/approval', {{ method: 'POST', headers: {{ 'Content-Type': 'application/json', 'X-MetaHarness-Token': META_TOKEN }}, body: JSON.stringify({{ decision }}) }})
    .then(response => response.json().then(payload => ({{ ok: response.ok, payload }})))
    .then(result => {{ document.querySelector('#approval-message').textContent = result.ok ? ' Décision enregistrée.' : ' ' + (result.payload.message || 'Erreur.'); }})
    .catch(() => {{ document.querySelector('#approval-message').textContent = ' Erreur réseau.'; }});
}}
const approveButton = document.querySelector('#approve'); if (approveButton) approveButton.addEventListener('click', () => decide('APPROVE'));
const rejectButton = document.querySelector('#reject'); if (rejectButton) rejectButton.addEventListener('click', () => decide('REJECT'));
pollProgress(); setInterval(pollProgress, 1000);
"""
    body = f"""
<p><a href="/">← Tous les runs</a></p>
<header><h1>Run {_e(run_id)}</h1><p>Status : <strong>{_e(status)}</strong></p></header>
{buttons}
<section><h2>Header</h2><div class="grid">
  <div class="card"><strong>Base SHA</strong><br>{_e(state.get('base_sha'))}</div>
  <div class="card"><strong>Branch</strong><br>{_e(state.get('branch'))}</div>
  <div class="card"><strong>Worktree</strong><br>{_e(state.get('worktree'))}</div>
  <div class="card"><strong>Commit SHA</strong><br>{_e(state.get('commit_sha'))}</div>
  <div class="card"><strong>Failure</strong><br>{_failure(state.get('failure'))}</div>
</div></section>
<section><h2>Timeline</h2>{_timeline(status)}</section>
<section><h2>Plan</h2><h3>Canonical implementation contract</h3><pre>{_e(plan.get('contract'))}</pre><h3>planner.raw.md</h3><pre>{_e(plan.get('raw'))}</pre></section>
<section><h2>Progress Codex</h2><ul id="progress-events"></ul></section>
<section><h2>Checks</h2>{_check_cards(run.get('checks'))}</section>
<section><h2>Review</h2>{_review(run.get('review'))}</section>
<section><h2>Reviewer raw</h2><details><summary>Déplier reviewer.raw.md</summary><pre>{_e(run.get('reviewer_raw'))}</pre></details></section>
"""
    return _page(f"Run {run_id}", body, script)


__all__ = ["render_index", "render_run"]
