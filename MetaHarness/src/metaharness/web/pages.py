"""Server-rendered pages for the local MetaHarness UI.

Every page works without JavaScript.  A running run page additionally
loads the static ``/static/run.js`` for targeted live updates; there is
no full-page refresh and no inline script.
"""

from __future__ import annotations

from html import escape
from typing import Any

import json

from ..models import ExecutionModePolicy, HarnessConfig
from ..profiles import profiles_for_config, safe_profile_metadata
from ..run_options import RunOptions
from .api import LIVE_STOP_STATUSES, context_level, publish_target


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _page(
    title: str,
    body: str,
    *,
    nonce: str | None = None,
    refresh_seconds: int | None = None,
    script: bool = False,
) -> str:
    nonce_attribute = f' nonce="{_e(nonce)}"' if nonce else ""
    script_tag = '<script src="/static/run.js" defer></script>' if script else ""
    refresh = (
        f'<meta http-equiv="refresh" content="{refresh_seconds}">'
        if refresh_seconds is not None
        else ""
    )
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="referrer" content="no-referrer">
  {refresh}
  <title>{_e(title)} · MetaHarness</title>
  <style{nonce_attribute}>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 1240px; margin: 0 auto; padding: 1rem; line-height: 1.45; }}
    a {{ color: #6aa9ff; }}
    header.sticky {{ position: sticky; top: 0; z-index: 2; padding: .8rem 0; background: Canvas; border-bottom: 1px solid #7776; }}
    h1, h2, h3 {{ line-height: 1.2; }} h2 {{ margin-top: 2rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1rem 0 2rem; }}
    th, td {{ text-align: left; border-bottom: 1px solid #7776; padding: .55rem; vertical-align: top; }}
    pre, code, .mono {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; max-height: 32rem; overflow: auto; padding: 1rem; border: 1px solid #7776; border-radius: .4rem; background: #7772; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: .8rem; }}
    .card {{ border: 1px solid #7776; border-radius: .5rem; padding: .85rem; }}
    .card.pass {{ border-color: #4cae4c; }} .card.fail, .danger {{ color: #ff8d8d; }} .muted {{ opacity: .72; }}
    .badge {{ display: inline-block; border: 1px solid #7778; border-radius: 999px; padding: .12rem .55rem; font-size: .9rem; font-weight: 700; }}
    .badge.failed {{ border-color: #d66; color: #ff8d8d; }} .badge.success {{ border-color: #4cae4c; color: #71d471; }}
    .timeline {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .35rem; list-style: none; padding: 0; }}
    .timeline li {{ border: 1px solid #7776; border-radius: .35rem; padding: .4rem .55rem; }} .timeline li.current {{ border-color: #4cae4c; font-weight: 700; }} .timeline li.done {{ opacity: .72; }}
    label {{ display: block; font-weight: 700; margin-top: .8rem; }} input, textarea, select {{ box-sizing: border-box; max-width: 100%; padding: .45rem; margin-top: .25rem; }} textarea {{ width: min(100%, 72rem); }}
    button {{ padding: .6rem .9rem; margin: .8rem .5rem 0 0; cursor: pointer; font-weight: 700; }} button.approve {{ border-color: #4cae4c; }} button.reject {{ border-color: #d66; }}
    dl {{ margin: .4rem 0; }} dt {{ font-weight: 700; }} dd {{ margin: 0 0 .55rem; overflow-wrap: anywhere; }} summary {{ cursor: pointer; font-weight: 700; }}
    .run-card {{ border: 1px solid #7776; border-radius: .6rem; padding: .8rem 1rem; }}
    .run-card h1 {{ margin: 0 0 .5rem; font-size: 1.15rem; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .5rem 1rem; }}
    .label {{ margin: 0; font-size: .72rem; letter-spacing: .08em; font-weight: 700; opacity: .7; }}
    .value {{ margin: .1rem 0 .3rem; font-weight: 700; }} .small {{ font-size: .82rem; }}
    dl.tokens {{ display: grid; grid-template-columns: auto 1fr; gap: 0 .5rem; margin: 0; font-size: .82rem; }} dl.tokens dd {{ margin: 0; }}
    .failure-card {{ margin-top: .6rem; }} .failure-title {{ font-size: 1.1rem; margin: .2rem 0; }}
    button.resume {{ border-color: #4cae4c; text-transform: uppercase; }}
    ol.pipeline-list {{ list-style: none; padding: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .35rem; }}
    ol.pipeline-list li {{ border: 1px solid #7776; border-radius: .35rem; padding: .35rem .55rem; }}
    .symbol {{ display: inline-block; width: 1.2rem; font-weight: 700; }}
    .state-complete .symbol {{ color: #71d471; }} .state-running {{ border-color: #6aa9ff !important; font-weight: 700; }}
    .state-failed {{ border-color: #d66 !important; color: #ff8d8d; }} .state-resumable {{ border-color: #e0a030 !important; }}
    .state-skipped {{ opacity: .55; }} .context.warning {{ color: #e0a030; }} .context.severe {{ color: #ff8d8d; }}
  </style>
  {script_tag}
</head>
<body>
{body}
</body>
</html>
"""


_TIMELINE = (
    ("created", "CREATED"), ("planning", "PLANNING"),
    ("awaiting_plan_approval", "AWAITING PLAN APPROVAL"), ("worktree_ready", "WORKTREE"),
    ("preparing", "PREPARING"), ("implementing", "IMPLEMENTING"),
    ("pre_revision_validating", "PRE-REVISION VALIDATING"),
    ("revising", "REVISING"), ("revalidating", "REVALIDATING"), ("reviewing", "REVIEWING"),
    ("approved", "APPROVED"), ("publishing", "PUBLISHING"), ("published", "PUBLISHED"),
)
_ORDER = {value: index for index, (value, _label) in enumerate(_TIMELINE)}
_TERMINAL_LABELS = {"blocked": "BLOCKED", "plan_rejected": "REJECTED", "failed": "FAILED", "interrupted": "INTERRUPTED"}
TERMINAL_STATUSES = frozenset({"committed", "published", *_TERMINAL_LABELS})
AWAITING_APPROVAL_STATUS = "awaiting_plan_approval"
# Elements updated in place by /static/run.js (textContent/classList/hidden).
RUN_PAGE_DYNAMIC_IDS: tuple[str, ...] = (
    "live-status", "live-updated", "live-current", "live-next", "live-failure",
    "live-tokens-planner", "live-tokens-luna", "live-tokens-claude", "live-tokens-reviewer",
    "live-events", "refresh-details",
)
STATE_POLL_MS = 2000


def refresh_seconds_for_run(run: dict[str, Any]) -> int | None:
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    status = str(state.get("status", run.get("status", "")))
    if status in TERMINAL_STATUSES:
        return None
    if status == AWAITING_APPROVAL_STATUS:
        approval = run.get("approval") if isinstance(run.get("approval"), dict) else {}
        return 1 if approval.get("recorded") else None
    if status == "approved":
        return 1
    return 2


def _failure(value: Any) -> str:
    if not value:
        return '<span class="muted">—</span>'
    if isinstance(value, dict):
        suffix = f" — {_e(value.get('detail'))}" if value.get("detail") is not None else ""
        return f'<span class="danger">{_e(value.get("reason"))}{suffix}</span>'
    return f'<span class="danger">{_e(value)}</span>'


def _status_badge(status: Any) -> str:
    value = str(status or "—")
    style = "failed" if value in {"failed", "blocked", "plan_rejected", "interrupted"} else "success" if value in {"committed", "approved", "published"} else ""
    return f'<span class="badge {style}">{_e(value)}</span>'


def render_index(runs: list[dict[str, Any]], *, nonce: str | None = None) -> str:
    rows = []
    for run in runs:
        run_id = run.get("run_id")
        rows.append(
            "<tr>"
            f'<td><a href="/runs/{_e(run_id)}">{_e(run_id)}</a></td>'
            f"<td>{_status_badge(run.get('status'))}</td><td>{_e(run.get('updated_at'))}</td>"
            f"<td>{_e(run.get('plan_title'))}</td><td class=\"mono\">{_e(run.get('commit_sha'))}</td>"
            f"<td>{_failure(run.get('failure'))}</td></tr>"
        )
    table = "".join(rows) or '<tr><td colspan="6" class="muted">Aucun run.</td></tr>'
    body = f'''<header><h1>MetaHarness</h1><p class="muted">Observation locale des runs</p></header>
<p><a href="/new">NEW RUN</a></p>
<table><thead><tr><th>RUN ID</th><th>STATUS</th><th>UPDATED</th><th>PLAN TITLE</th><th>COMMIT</th><th>FAILURE</th></tr></thead><tbody>{table}</tbody></table>'''
    return _page("Runs", body, nonce=nonce, refresh_seconds=5)


def render_new_run(config: HarnessConfig, token: str, *, nonce: str | None = None) -> str:
    defaults = RunOptions.from_config(config)
    options = {
        "planner": _new_profile_options(config, "planner", defaults.planner_profile),
        "implementer": _new_profile_options(config, "implementer", defaults.default_implementer_profile),
        "reviewer": _new_profile_options(config, "reviewer", defaults.reviewer_profile),
        "reviser": _new_profile_options(config, "reviser", defaults.reviser_profile, optional=True),
        "repair": _new_profile_options(config, "repair", defaults.repair_profile, optional=True),
    }
    claude = "enabled" if defaults.claude_revision_enabled else "disabled"
    body = f'''<main><p><a href="/">← Tous les runs</a></p><h1>New Run</h1>
<dl><dt>Repository</dt><dd class="mono">{_e(config.repo)}</dd><dt>Base ref</dt><dd class="mono">{_e(config.base_ref)}</dd>
<dt>Execution policy</dt><dd>{_e(_execution_policy_label(config))}</dd>
<dt>Publication target</dt><dd>{_e(publish_target(config, {})[1])}</dd></dl>
<form action="/runs" method="post" accept-charset="UTF-8"><input type="hidden" name="_token" value="{_e(token)}">
<label for="spec">SPEC</label><textarea id="spec" name="spec" rows="20" required></textarea>
<label for="run-id">Run ID (optional)</label><input id="run-id" name="run_id" type="text" autocomplete="off" value="">
<section class="card"><h2>RUN OPTIONS</h2>
<h3>Planner</h3><select id="planner-profile" name="planner_profile" required>{options["planner"]}</select>
<label for="implementer-profile">Default implementer</label><select id="implementer-profile" name="default_implementer_profile" required>{options["implementer"]}</select>
<label for="reviewer-profile">Final reviewer</label><select id="reviewer-profile" name="reviewer_profile" required>{options["reviewer"]}</select>
<label for="claude-revision">Claude revision</label><select id="claude-revision" name="claude_revision_enabled" required><option value="enabled"{" selected" if claude == "enabled" else ""}>enabled</option><option value="disabled"{" selected" if claude == "disabled" else ""}>disabled</option></select>
<label for="reviser-profile">Claude reviser profile</label><select id="reviser-profile" name="reviser_profile" aria-describedby="reviser-help">{options["reviser"]}</select><p id="reviser-help" class="muted">Ce choix est validé côté serveur même si Claude est désactivé.</p>
<label for="repair-cycles">Automatic repair cycles</label><select id="repair-cycles" name="repair_cycles" required><option value="0"{" selected" if defaults.repair_cycles == 0 else ""}>0</option><option value="1"{" selected" if defaults.repair_cycles == 1 else ""}>1</option></select>
<label for="repair-profile">Repair implementer profile</label><select id="repair-profile" name="repair_profile">{options["repair"]}</select>
<label for="decomposition">Decomposition</label><select id="decomposition" name="decomposition" required><option value="balanced"{" selected" if defaults.decomposition == "balanced" else ""}>balanced</option><option value="aggressive"{" selected" if defaults.decomposition == "aggressive" else ""}>aggressive</option></select>
<label for="execution-mode-policy">Execution mode</label><select id="execution-mode-policy" name="execution_mode_policy" required><option value="auto"{" selected" if defaults.execution_mode_policy == "auto" else ""}>auto</option><option value="require-staged"{" selected" if defaults.execution_mode_policy == "require-staged" else ""}>require-staged</option></select>
<label for="single-limit">SINGLE mutable paths</label><input id="single-limit" name="single_step_max_mutable_paths" type="number" min="1" step="1" value="{_e(defaults.single_step_max_mutable_paths)}" required>
<label for="staged-limit">STAGED mutable paths</label><input id="staged-limit" name="staged_step_max_mutable_paths" type="number" min="1" step="1" value="{_e(defaults.staged_step_max_mutable_paths)}" required>
</section><br><button type="submit">CREATE RUN</button></form></main>'''
    return _page("New Run", body, nonce=nonce)


def _new_profile_options(
    config: HarnessConfig, role: str, selected: Any, *, optional: bool = False,
) -> str:
    rows = []
    if optional and selected is None:
        rows.append('<option value="" selected>not configured</option>')
    for profile in profiles_for_config(config).values():
        item = safe_profile_metadata(profile)
        if role in item["roles"]:
            rows.append(
                f'<option value="{_e(item["id"])}"{" selected" if item["id"] == selected else ""}>'
                f'{_e(item["display_name"])}</option>'
            )
    return "".join(rows)


def _execution_policy_label(config: HarnessConfig) -> str:
    if config.planning.execution_mode_policy == ExecutionModePolicy.REQUIRE_STAGED.value:
        return "STAGED required"
    return "auto (planner chooses SINGLE or STAGED)"


def _timeline_items(status: Any) -> str:
    current = str(status or "")
    current_order = _ORDER.get(current, -1)
    if current == "committed":
        # Historic non-publishing runs end immediately after APPROVED.
        current_order = _ORDER["approved"] + 1
    items = []
    for value, label in _TIMELINE:
        classes = "current" if value == current else "done" if current_order >= 0 and _ORDER.get(value, 99) < current_order else ""
        items.append(f'<li class="{classes}">{label}</li>')
    if current == "committed":
        items.append('<li class="current">COMMITTED</li>')
    if current in _TERMINAL_LABELS:
        items.append(f'<li class="current danger">{_TERMINAL_LABELS[current]}</li>')
    return "".join(items)


def _failure_reason(run: dict[str, Any]) -> str:
    failure = run.get("failure")
    return str(failure.get("reason", "")) if isinstance(failure, dict) else str(failure or "")


def _publish_section(state: dict[str, Any]) -> str:
    publish = state.get("publish") if isinstance(state.get("publish"), dict) else {}
    if not publish:
        return ""
    if publish.get("mode") == "fast-forward-base":
        return _fast_forward_publish_section(state, publish)
    web_url = publish.get("web_url")
    link = (
        f'<a href="{_e(web_url)}" rel="noopener noreferrer">branch</a>'
        if isinstance(web_url, str) and web_url.startswith("https://")
        else _e(publish.get("branch"))
    )
    return (
        '<section><h2>PUBLISH</h2><p><strong>'
        f'{_e(str(state.get("status", "")).upper())}</strong></p>'
        f'<dl><dt>commit</dt><dd class="mono">{_e(publish.get("commit_sha") or state.get("commit_sha"))}</dd>'
        f'<dt>remote</dt><dd>{_e(publish.get("remote"))}</dd>'
        f'<dt>branch</dt><dd class="mono">{link}</dd></dl></section>'
    )


def _fast_forward_publish_section(state: dict[str, Any], publish: dict[str, Any]) -> str:
    target = f'{publish.get("remote")}/{publish.get("target")}'
    commit = publish.get("commit_sha") or state.get("commit_sha")
    web_url = publish.get("web_url")
    link = (
        f' · <a href="{_e(web_url)}" rel="noopener noreferrer">commit</a>'
        if isinstance(web_url, str) and web_url.startswith("https://") else ""
    )
    if publish.get("status") == "pushed":
        headline = f'<p><strong>Published to {_e(target)}</strong></p><p class="mono">{_e(commit)}{link}</p>'
    else:
        local = (
            f'<p class="danger">Local {_e(publish.get("target"))} already points to '
            f'<span class="mono">{_e(commit)}</span>; {_e(target)} was not updated.</p>'
            if publish.get("local_base_updated") else ""
        )
        headline = f'<p class="danger"><strong>Not published to {_e(target)}</strong> ({_e(publish.get("status"))})</p>{local}'
    notices = "".join(
        f'<p class="muted">The checkout <span class="mono">{_e(path)}</span> has '
        f'{_e(publish.get("target"))} checked out: MetaHarness did not touch its index or files. '
        f'Synchronize it with <span class="mono">git -C {_e(path)} read-tree -m -u '
        f'{_e(publish.get("base_sha"))} {_e(commit)}</span> (refuses to overwrite local changes).</p>'
        for path in (publish.get("base_checked_out_in") or []) if isinstance(path, str)
    )
    run_branch = publish.get("run_branch")
    branch_line = (
        f'<p class="muted">Isolated run branch (kept locally, not pushed): '
        f'<span class="mono">{_e(run_branch)}</span></p>' if run_branch else ""
    )
    return f'<section class="publish"><h2>PUBLISH</h2>{headline}{branch_line}{notices}</section>'


def _section_open(run: dict[str, Any], names: tuple[str, ...]) -> str:
    return " open" if any(_failure_reason(run).startswith(name) for name in names) else ""


def _agent_auth_failure_notice(run: dict[str, Any], config: HarnessConfig | None) -> str:
    if _failure_reason(run) != "CODEX_AUTH_FAILURE":
        return ""
    configured_home = (
        str(config.codex_runtime.home.expanduser().resolve())
        if config is not None
        else "<configured home>"
    )
    return (
        '<div class="card fail"><p><strong>Codex authentication failed.</strong></p>'
        "<p>Run once:</p>"
        f'<pre>CODEX_HOME="{_e(configured_home)}" codex login</pre></div>'
    )


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
        passed = check.get("exit_code") == 0 and not check.get("timed_out") and not check.get("workspace_mutated")
        status = "PASS" if passed else "FAIL"
        duration = check.get("duration_seconds", check.get("duration"))
        cards.append(f'<article class="card {"pass" if passed else "fail"}"><h3>{_e(check.get("name"))} — {status}</h3><p><strong>{status}</strong> · exit code {_e(check.get("exit_code"))} · duration {_e(duration)} · mutation {_e(check.get("workspace_mutated"))}</p><details><summary>stdout / stderr</summary><p>stdout</p><pre>{_e(check.get("stdout_tail"))}</pre><p>stderr</p><pre>{_e(check.get("stderr_tail"))}</pre></details></article>')
    return '<div class="grid">' + "".join(cards) + "</div>"


def _review(review: Any) -> str:
    if not review:
        return '<p class="muted">Aucune review.</p>'
    if not isinstance(review, dict):
        return f"<pre>{_e(review)}</pre>"
    fields = (("verdict", review.get("verdict")), ("route", review.get("route")), ("summary", review.get("summary")), ("findings", review.get("findings")), ("required fixes", review.get("required_fixes", review.get("required fixes"))), ("missing tests", review.get("missing_tests", review.get("missing tests"))), ("residual risks", review.get("residual_risks", review.get("residual risks"))))
    return "<dl>" + "".join(f"<dt>{_e(label)}</dt><dd>{_e(value)}</dd>" for label, value in fields) + "</dl>"


def _execution_card(state: dict[str, Any], config: HarnessConfig | None) -> str:
    execution = state.get("execution") if isinstance(state.get("execution"), dict) else {}
    metadata = {}
    if config is not None:
        metadata = {profile.id: safe_profile_metadata(profile) for profile in profiles_for_config(config).values()}
    cards = []
    for role, title in (("planner", "Planner"), ("implementer", "Implementer"), ("reviser", "Reviser"), ("repair_implementer", "Repair implementer"), ("reviewer", "Reviewer")):
        selected = execution.get(role) if isinstance(execution.get(role), dict) else {}
        profile_id = selected.get("profile_id")
        profile = metadata.get(profile_id, {})
        model = selected.get("model", profile.get("model_label")); mode = selected.get("selection_mode", profile.get("selection_mode")); effort = selected.get("effort", profile.get("effort"))
        warning = '<p class="danger">external-ui: le modèle est sélectionné dans le fournisseur externe.</p>' if mode == "external-ui" else ""
        extra = f'<dt>effort</dt><dd>{_e(effort)}</dd>' if role == "implementer" else ""
        cards.append(f'<article class="card"><h3>{title}</h3><dl><dt>profile</dt><dd>{_e(profile_id or "—")}</dd><dt>model</dt><dd>{_e(model or "—")}</dd><dt>selection mode</dt><dd>{_e(mode or "—")}</dd>{extra}</dl>{warning}</article>')
    return '<div class="grid">' + "".join(cards) + "</div>"


def _setup_cards(results: Any) -> str:
    if not results:
        return '<p class="muted">Aucune commande de setup.</p>'
    cards = []
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict):
            continue
        passed = result.get("exit_code") == 0 and not result.get("timed_out")
        cards.append(f'<article class="card {"pass" if passed else "fail"}"><h3>{_e(result.get("name"))} — {"PASS" if passed else "FAIL"}</h3><p>duration {_e(result.get("duration_seconds"))} · exit code {_e(result.get("exit_code"))}</p><details><summary>stdout / stderr</summary><pre>{_e(result.get("stdout_tail"))}</pre><pre>{_e(result.get("stderr_tail"))}</pre></details></article>')
    return '<div class="grid">' + "".join(cards) + "</div>"


def _profile_options(
    config: HarnessConfig | None, role: str, selected: Any, *, driver: str | None = None,
) -> str:
    if config is None:
        return ""
    result = []
    for profile in profiles_for_config(config).values():
        item = safe_profile_metadata(profile)
        if role in item["roles"] and (driver is None or item["driver"] == driver):
            result.append(f'<option value="{_e(item["id"])}"{" selected" if item["id"] == selected else ""}>{_e(item["display_name"])}</option>')
    return "".join(result)


_HIGH_WORKER_INPUT_TOKENS = 100_000
_HIGH_CONTEXT_WARNING = '<span class="danger">High worker context usage</span>'
_STEP_ICONS = {"completed": "✓", "failed": "✗", "interrupted": "✗", "running": "▶"}


def _artifact_map(artifacts: Any) -> dict[Any, dict[str, Any]]:
    if not isinstance(artifacts, list):
        return {}
    return {item.get("id"): item for item in artifacts if isinstance(item, dict)}


def _contract_block(artifact: dict[str, Any]) -> str:
    """The exact stored contract bytes, flagged if they are not the hashed ones."""

    contract = artifact.get("contract")
    if contract is None:
        return '<p class="danger">Contract artifact is missing.</p>'
    warning = "" if artifact.get("contract_matches_bundle") else (
        '<p class="danger">This contract does not match implementation_bundle.json.</p>'
    )
    return (
        f'<details open><summary>Exact implementation contract</summary>{warning}'
        f'<pre class="contract">{_e(contract)}</pre></details>'
    )


def _v2_approval_form(
    run_id: Any, token: str, state: dict[str, Any], config: HarnessConfig | None,
    artifacts: Any = None,
) -> str:
    planner = state.get("planner") if isinstance(state.get("planner"), dict) else {}
    steps = planner.get("steps") if isinstance(planner.get("steps"), list) else state.get("steps", [])
    artifact_map = _artifact_map(artifacts)
    metadata = (
        {profile.id: safe_profile_metadata(profile) for profile in profiles_for_config(config).values()}
        if config is not None else {}
    )
    options = state.get("run_options") if isinstance(state.get("run_options"), dict) else {}
    if not options and config is not None:
        options = RunOptions.from_config(config).to_dict()
    requested_profiles = options.get("profiles") if isinstance(options.get("profiles"), dict) else {}
    rows: list[str] = []
    overview: list[str] = []
    for item in steps:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        step_id = item["id"]
        selected = requested_profiles.get("default_implementer_profile") or item.get("recommended_profile") or item.get("profile_id")
        meta = metadata.get(selected, {})
        overview.append(
            f'<li><span class="mono">{_e(step_id)}</span> → recommended '
            f'{_profile_triplet(selected, meta.get("model_label"), meta.get("effort"))}</li>'
        )
        rows.append(
            f'<section class="card"><h3>{_e(step_id)} — {_e(item.get("title"))}</h3>'
            f'<p>Recommended implementer: <span class="mono">{_e(selected)}</span></p>'
            f'<label for="step-profile-{_e(step_id)}">Implementer (selected)</label>'
            f'<select id="step-profile-{_e(step_id)}" name="step_profile__{_e(step_id)}" required>'
            f'{_profile_options(config, "implementer", selected)}</select>'
            f'{_contract_block(artifact_map.get(step_id, {}))}</section>'
        )
    reviewer = requested_profiles.get("reviewer_profile") or planner.get("reviewer_recommendation")
    # Revision is shown (and selectable) only when revision.enabled: all four
    # execution families are then visible before APPROVE.  The repair
    # implementer is its own configured default, never derived from the
    # reviser, and only Codex profiles are offered for it.
    cycle_profiles = ""
    pipeline = options.get("pipeline") if isinstance(options.get("pipeline"), dict) else {}
    cycle_enabled = bool(pipeline.get("claude_revision_enabled")) or pipeline.get("repair_cycles") == 1
    if config is not None and cycle_enabled:
        reviser = requested_profiles.get("reviser_profile")
        repair = requested_profiles.get("repair_profile")
        reviser_meta = metadata.get(reviser, {})
        repair_meta = metadata.get(repair, {})
        cycle_profiles = (
            '<section class="card revision-profile"><h3>Reviser</h3>'
            f'<p>Claude <span class="mono">{_e(reviser_meta.get("model_label"))}</span> · '
            f'recommended {_profile_triplet(reviser, reviser_meta.get("model_label"), reviser_meta.get("effort"))}</p>'
            '<label for="reviser-profile">Reviser (selected)</label>'
            f'<select id="reviser-profile" name="reviser_profile" required>{_profile_options(config, "reviser", reviser, driver="claude-code")}</select></section>'
            '<section class="card repair-profile"><h3>Repair implementer</h3>'
            f'<p>Codex <span class="mono">{_e(repair_meta.get("model_label"))}</span> · '
            f'recommended {_profile_triplet(repair, repair_meta.get("model_label"), repair_meta.get("effort"))}</p>'
            '<label for="repair-profile">Repair implementer (selected)</label>'
            f'<select id="repair-profile" name="repair_profile" required>{_profile_options(config, "repair", repair, driver="codex")}</select></section>'
        )
    execution = state.get("execution") if isinstance(state.get("execution"), dict) else {}
    planner_selected = execution.get("planner") if isinstance(execution.get("planner"), dict) else {}
    reviewer_meta = metadata.get(reviewer, {})
    return f'''<section class="card"><h2>Execution plan</h2>
<p>Execution mode: <strong>{_e(planner.get("execution_mode"))}</strong></p>
<p>Steps: {_e(len(rows))}</p>
{_approval_targets(config)}
<form action="/runs/{_e(run_id)}/approval" method="post"><input type="hidden" name="_token" value="{_e(token)}"><input type="hidden" name="decision" value="APPROVE">
<h3>Planner</h3><p class="mono">{_e(planner_selected.get("profile_id") or "—")} / {_e(planner_selected.get("model") or "—")}</p>
<h3>Initial implementation</h3><ul class="plan-steps">{"".join(overview)}</ul>
{"".join(rows)}{cycle_profiles}<section class="card reviewer-profile"><h3>Reviewer</h3><p>recommended {_profile_triplet(reviewer, reviewer_meta.get("model_label"), reviewer_meta.get("selection_mode"))}</p><label for="reviewer-profile">Reviewer (selected)</label><select id="reviewer-profile" name="reviewer_profile" required>{_profile_options(config, "reviewer", reviewer)}</select></section><br><button class="approve" type="submit">APPROVE PLAN</button></form>
<form action="/runs/{_e(run_id)}/approval" method="post"><input type="hidden" name="_token" value="{_e(token)}"><input type="hidden" name="decision" value="REJECT"><button class="reject" type="submit">REJECT PLAN</button></form></section>'''


def _approval_targets(config: HarnessConfig | None) -> str:
    if config is None:
        return ""
    return (
        f'<p>Execution policy: <strong>{_e(_execution_policy_label(config))}</strong></p>'
        f'<p>Publication target: <strong>{_e(publish_target(config, {})[1])}</strong></p>'
    )


def _kilo(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "0"
    return f"{value // 1000}k" if value >= 1000 else str(value)


_CONTEXT_LABELS = {"warning": "High worker context usage", "severe": "SEVERE CONTEXT USAGE"}


def _context_line(step_id: Any, cycle: Any, usage: dict[str, Any], level: str) -> str:
    if level not in _CONTEXT_LABELS:
        return ""
    return (
        f'<p class="context {level}">Luna {_e(step_id)} · {_kilo(usage.get("input_tokens"))} input · '
        f'{_kilo(usage.get("cached_input_tokens"))} cached · <strong>{_CONTEXT_LABELS[level]}</strong>'
        f' · <a href="#diag-c{_e(cycle)}-{_e(step_id)}">Voir diagnostic</a></p>'
    )


def _step_card(item: dict[str, Any], artifact: dict[str, Any]) -> str:
    status = str(item.get("status", "waiting"))
    icon = _STEP_ICONS.get(status, "…")
    usage = artifact.get("usage") if isinstance(artifact.get("usage"), dict) else item.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    level = artifact.get("context_level") or context_level(usage)
    warning = f" · {_e(_CONTEXT_LABELS[level])}" if level in _CONTEXT_LABELS else (
        f" · {_HIGH_CONTEXT_WARNING}" if artifact.get("high_context") else ""
    )
    cycle = artifact.get("cycle", item.get("cycle", 1))
    diagnostics = artifact.get("token_diagnostics")
    diagnostic_block = (
        f'<details id="diag-c{_e(cycle)}-{_e(item.get("id"))}" class="token-diagnostic">'
        f'<summary>Token diagnostic</summary><pre>{_e(json.dumps(diagnostics, indent=2))}</pre></details>'
        if isinstance(diagnostics, dict) else ""
    )
    events = artifact.get("events") if isinstance(artifact.get("events"), list) else []
    event_items = "".join(f"<li>{_e(event)}</li>" for event in events) or '<li class="muted">No event yet.</li>'
    reason = artifact.get("failure_reason")
    reason_line = f'<p class="danger">failure: {_e(reason)}</p>' if reason else ""
    return (
        f'<details class="card step {_e(status)}"{" open" if status in {"running", "failed"} else ""}>'
        f'<summary>{_e(item.get("id"))} {icon} — {_e(item.get("title"))} · '
        f'{_e(usage.get("input_tokens", 0))} input / {_e(usage.get("output_tokens", 0))} output{warning}</summary>'
        f'<p>profile: <span class="mono">{_e(item.get("profile_id"))}</span></p>'
        f'<p>status: {_e(status)}</p>{reason_line}'
        f'{_context_line(item.get("id"), cycle, usage, level)}'
        f'<h4>Recent events</h4><ul class="events">{event_items}</ul>'
        f'<details><summary>contract</summary><pre>{_e(artifact.get("contract"))}</pre></details>'
        f'<details><summary>final report</summary><pre>{_e(artifact.get("final"))}</pre></details>'
        f'<details><summary>stderr</summary><pre>{_e(artifact.get("stderr"))}</pre></details>'
        f'{diagnostic_block}</details>'
    )


def _v2_steps(state: dict[str, Any], artifacts: Any = None) -> str:
    steps = state.get("steps") if isinstance(state.get("steps"), list) else []
    if not steps:
        return '<p class="muted">No staged steps.</p>'
    artifact_map = _artifact_map(artifacts)
    return "".join(
        _step_card(item, artifact_map.get(item.get("id"), {}))
        for item in steps if isinstance(item, dict)
    )


# Statuses during which a cycle phase is the current one (auto-opened).
_REVISION_PHASES = frozenset({"pre_revision_validating", "revising"})
_CHECK_PHASES = frozenset({"validating", "revalidating"})
_REVIEW_PHASES = frozenset({"reviewing"})


def _checks_verdict(checks: Any) -> str:
    if not isinstance(checks, dict):
        return "—"
    gate = checks.get("gate") if isinstance(checks.get("gate"), dict) else {}
    if isinstance(gate.get("passed"), bool):
        return "PASS" if gate["passed"] else "FAIL"
    items = checks.get("checks") if isinstance(checks.get("checks"), list) else []
    if not items:
        return "—"
    required = [item for item in items if isinstance(item, dict) and item.get("required", True)]
    passed = all(
        item.get("exit_code") == 0 and not item.get("timed_out") and not item.get("workspace_mutated")
        for item in required
    )
    return "PASS" if passed else "FAIL"


def _cycle_revision_block(revision: Any, open_attr: str) -> str:
    if not isinstance(revision, dict):
        return f'<details class="card revision"{open_attr}><summary>Claude revision</summary><p class="muted">Not started.</p></details>'
    report = revision.get("report") if isinstance(revision.get("report"), dict) else {}
    events = revision.get("events") if isinstance(revision.get("events"), list) else []
    event_items = "".join(f"<li>{_e(event)}</li>" for event in events) or '<li class="muted">No event yet.</li>'
    changed = report.get("changed_paths") if isinstance(report.get("changed_paths"), list) else []
    return (
        f'<details class="card revision"{open_attr}><summary>Claude revision · '
        f'{_e(report.get("status") or "running")} · {_usage_pair(revision.get("usage"))}</summary>'
        f'<p>profile: <span class="mono">{_e(report.get("profile_id"))}</span></p>'
        f'<p>changed paths: <span class="mono">{_e(", ".join(str(path) for path in changed) or "—")}</span></p>'
        f'<h4>Recent events</h4><ul class="events">{event_items}</ul>'
        f'<details><summary>revision report</summary><pre>{_e(revision.get("final"))}</pre></details>'
        f'<details><summary>pre-revision checks</summary><pre>{_e(revision.get("pre_checks"))}</pre></details></details>'
    )


def _cycle_checks_block(checks: Any, open_attr: str) -> str:
    if not isinstance(checks, dict):
        return f'<details class="card checks"{open_attr}><summary>Checks</summary><p class="muted">Not run.</p></details>'
    changed = checks.get("changed_files") if isinstance(checks.get("changed_files"), list) else []
    return (
        f'<details class="card checks"{open_attr}><summary>Checks · {_checks_verdict(checks)}</summary>'
        f'{_check_cards(checks.get("checks"))}'
        f'<p>Changed files</p><ul>{"".join(f"<li class=mono>{_e(path)}</li>" for path in changed) or "<li class=muted>—</li>"}</ul>'
        f'<details><summary>Diff</summary><pre>{_e(checks.get("diff_tail"))}</pre></details></details>'
    )


def _cycle_review_block(number: Any, review: Any, open_attr: str) -> str:
    title = f"Reviewer #{_e(number)}"
    if not isinstance(review, dict):
        return f'<details class="card review"{open_attr}><summary>{title}</summary><p class="muted">Not reviewed.</p></details>'
    result = review.get("review") if isinstance(review.get("review"), dict) else {}
    verdict = f'{result.get("verdict") or "—"} / {result.get("route") or "—"}'
    return (
        f'<details class="card review"{open_attr}><summary>{title} · {_e(verdict)}</summary>'
        f'{_review(result)}<details><summary>reviewer.raw.md</summary><pre>{_e(review.get("raw"))}</pre></details></details>'
    )


def _cycle_sections(run: dict[str, Any]) -> str:
    """CYCLE 1 — INITIAL and (when it exists) CYCLE 2 — REPAIR, never mixed."""

    cycles = run.get("cycle_artifacts") if isinstance(run.get("cycle_artifacts"), list) else []
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    status = str(state.get("status", ""))
    current = run.get("cycle") if run.get("cycle") in (1, 2) else 1
    terminal = status in TERMINAL_STATUSES
    sections: list[str] = []
    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue
        number = cycle.get("number")
        kind = str(cycle.get("kind", "")).upper()
        active = number == current
        steps = cycle.get("steps") if isinstance(cycle.get("steps"), list) else []
        cards = "".join(
            _step_card(item, item) for item in steps if isinstance(item, dict)
        ) or '<p class="muted">No staged steps.</p>'

        def phase_open(phases: frozenset[str]) -> str:
            return " open" if active and not terminal and status in phases else ""

        failure = cycle.get("failure")
        header_note = f' · <span class="danger">{_e(failure)}</span>' if failure else ""
        sections.append(
            f'<details class="card cycle cycle-{_e(number)}"{" open" if active or terminal and number == len(cycles) else ""}>'
            f'<summary>CYCLE {_e(number)} — {_e(kind)} · {_e(cycle.get("status") or "—")}{header_note}</summary>'
            f'{cards}'
            f'{_cycle_revision_block(cycle.get("revision"), phase_open(_REVISION_PHASES))}'
            f'{_cycle_checks_block(cycle.get("checks"), phase_open(_CHECK_PHASES))}'
            f'{_cycle_review_block(number, cycle.get("review"), phase_open(_REVIEW_PHASES))}'
            '</details>'
        )
    return "".join(sections)


def _final_summary(run: dict[str, Any]) -> str:
    """One line matching the FINAL cycle's checks and reviewer."""

    cycles = run.get("cycle_artifacts") if isinstance(run.get("cycle_artifacts"), list) else []
    if not cycles or not isinstance(cycles[-1], dict):
        return ""
    final = cycles[-1]
    review = final.get("review") if isinstance(final.get("review"), dict) else {}
    result = review.get("review") if isinstance(review.get("review"), dict) else {}
    verdict = f'{result.get("verdict") or "—"} / {result.get("route") or "—"}' if result else "—"
    return (
        f'<p class="final-summary">Final cycle C0{_e(final.get("number"))} ({_e(final.get("kind"))}) · '
        f'checks {_checks_verdict(final.get("checks"))} · reviewer #{_e(final.get("number"))} {_e(verdict)}</p>'
    )


def _usage_pair(usage: Any) -> str:
    usage = usage if isinstance(usage, dict) else {}
    return f'{_e(usage.get("input_tokens", 0))} input / {_e(usage.get("output_tokens", 0))} output'


def _usage_detail(usage: Any) -> str:
    usage = usage if isinstance(usage, dict) else {}
    return (
        f'cached input {_e(usage.get("cached_input_tokens", 0))} · '
        f'cache write {_e(usage.get("cache_write_input_tokens", 0))} · '
        f'reasoning {_e(usage.get("reasoning_output_tokens", 0))} · '
        f'total {_e(usage.get("total_tokens", 0))}'
    )


def _usage_section(run: dict[str, Any]) -> str:
    """Token counters per phase, as persisted; no cost is derived."""

    usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
    implementer = usage.get("implementer") if isinstance(usage.get("implementer"), dict) else {}
    if "luna_c01" in usage:
        phase_rows = (
            ("Planner initial", usage.get("planner")),
            ("Luna C01", usage.get("luna_c01")),
            ("Claude C01", usage.get("claude_c01")),
            ("Reviewer C01", usage.get("reviewer_c01")),
            ("Repair planner C02", usage.get("repair_planner_c02")),
            ("Luna C02", usage.get("luna_c02")),
            ("Claude C02", usage.get("claude_c02")),
            ("Reviewer C02", usage.get("reviewer_c02")),
            ("Grand total", usage.get("grand_total")),
        )
    else:
        phase_rows = (
            ("Planner", usage.get("planner")),
            ("Luna", implementer.get("total")),
            ("Claude revision", usage.get("reviser")),
            ("Reviewer", usage.get("reviewer")),
        )
    rows = "".join(
        f'<tr><th>{label}</th><td>{_usage_pair(value)}</td><td class="muted">{_usage_detail(value)}</td></tr>'
        for label, value in phase_rows
    )
    step_rows = []
    for step in implementer.get("steps") if isinstance(implementer.get("steps"), list) else []:
        if not isinstance(step, dict):
            continue
        step_usage = step.get("usage") if isinstance(step.get("usage"), dict) else {}
        level = context_level(step_usage)
        flag = (
            f' · <span class="context {level}">{_CONTEXT_LABELS[level]}</span>'
            f' · <a href="#diag-c{_e(step.get("cycle", 1))}-{_e(step.get("id"))}">Voir diagnostic</a>'
            if level in _CONTEXT_LABELS else ""
        )
        step_rows.append(
            f'<tr><td class="mono">{_e(step.get("id"))}</td><td>{_usage_pair(step_usage)}</td>'
            f'<td class="muted">{_usage_detail(step_usage)}{flag}</td></tr>'
        )
    steps_table = (
        '<table class="usage-steps"><thead><tr><th>Luna step</th><th>tokens</th><th>detail</th></tr></thead>'
        f'<tbody>{"".join(step_rows)}</tbody></table>'
        if step_rows else ""
    )
    return (
        '<section class="usage"><h2>TOKEN USAGE</h2>'
        f'<table class="usage-phases"><tbody>{rows}</tbody></table>{steps_table}</section>'
    )


def _profile_triplet(profile_id: Any, model: Any, effort: Any) -> str:
    if not profile_id:
        return '<span class="muted">—</span>'
    return f'<span class="mono">{_e(profile_id)} / {_e(model or "—")} / {_e(effort or "—")}</span>'


def _execution_card_v2(
    state: dict[str, Any], run: dict[str, Any], config: HarnessConfig | None,
) -> str:
    """Planner, reviewer and per-step implementers; recommended vs approved."""

    metadata: dict[str, Any] = {}
    if config is not None:
        metadata = {profile.id: safe_profile_metadata(profile) for profile in profiles_for_config(config).values()}
    execution = state.get("execution") if isinstance(state.get("execution"), dict) else {}
    planner_state = state.get("planner") if isinstance(state.get("planner"), dict) else {}
    selection = run.get("execution_selection")
    approved = selection if isinstance(selection, dict) and selection.get("schema_version") in (3, 4) else {}
    planner = execution.get("planner") if isinstance(execution.get("planner"), dict) else {}
    planner_mode = planner.get("selection_mode")
    planner_warning = '<p class="danger">external-ui: le modèle est sélectionné dans le fournisseur externe.</p>' if planner_mode == "external-ui" else ""
    planner_card = (
        f'<article class="card"><h3>Planner</h3><dl><dt>profile</dt><dd>{_e(planner.get("profile_id") or "—")}</dd>'
        f'<dt>model</dt><dd>{_e(planner.get("model") or "—")}</dd><dt>selection mode</dt><dd>{_e(planner_mode or "—")}</dd></dl>{planner_warning}</article>'
    )
    reviewer_recommended = planner_state.get("reviewer_recommendation")
    reviewer_approved = approved.get("reviewer") if isinstance(approved.get("reviewer"), dict) else {}
    recommended_meta = metadata.get(reviewer_recommended, {})
    reviewer_card = (
        f'<article class="card"><h3>Reviewer</h3><dl>'
        f'<dt>recommended</dt><dd>{_profile_triplet(reviewer_recommended, recommended_meta.get("model_label"), recommended_meta.get("selection_mode"))}</dd>'
        f'<dt>approved</dt><dd>{_profile_triplet(reviewer_approved.get("profile_id"), reviewer_approved.get("model"), reviewer_approved.get("selection_mode")) if reviewer_approved else "<span class=muted>pending approval</span>"}</dd>'
        f'</dl></article>'
    )
    reviser_approved = approved.get("reviser") if isinstance(approved.get("reviser"), dict) else execution.get("reviser", {})
    repair_approved = approved.get("repair_implementer") if isinstance(approved.get("repair_implementer"), dict) else execution.get("repair_implementer", {})
    reviser_approved = reviser_approved if isinstance(reviser_approved, dict) else {}
    repair_approved = repair_approved if isinstance(repair_approved, dict) else {}
    show_cycle = bool(
        reviser_approved or repair_approved or (config is not None and config.revision.enabled)
    )
    pending = "<span class=muted>pending approval</span>"
    cycle_cards = (
        f'<article class="card"><h3>Reviser</h3><dl><dt>approved</dt><dd>{_profile_triplet(reviser_approved.get("profile_id"), reviser_approved.get("model"), reviser_approved.get("effort")) if reviser_approved else pending}</dd></dl></article>'
        f'<article class="card"><h3>Repair implementer</h3><dl><dt>approved</dt><dd>{_profile_triplet(repair_approved.get("profile_id"), repair_approved.get("model"), repair_approved.get("effort")) if repair_approved else pending}</dd></dl></article>'
        if show_cycle else ""
    )
    approved_steps = {
        item.get("step_id"): item.get("implementer")
        for item in (approved.get("steps") if isinstance(approved.get("steps"), list) else [])
        if isinstance(item, dict) and isinstance(item.get("implementer"), dict)
    }
    recommended_steps = planner_state.get("steps") if isinstance(planner_state.get("steps"), list) else []
    rows = []
    for item in recommended_steps:
        if not isinstance(item, dict):
            continue
        step_id = item.get("id")
        recommended = item.get("recommended_profile")
        meta = metadata.get(recommended, {})
        chosen = approved_steps.get(step_id)
        rows.append(
            f'<tr><td class="mono">{_e(step_id)}</td>'
            f'<td>recommended {_profile_triplet(recommended, meta.get("model_label"), meta.get("effort"))}</td>'
            f'<td>approved {_profile_triplet(chosen.get("profile_id"), chosen.get("model"), chosen.get("effort")) if chosen else "<span class=muted>pending approval</span>"}</td></tr>'
        )
    steps_table = (
        '<h3>Step implementers</h3><table class="step-implementers"><tbody>'
        + ("".join(rows) or '<tr><td class="muted">No step.</td></tr>')
        + "</tbody></table>"
    )
    return f'<div class="grid">{planner_card}{cycle_cards}{reviewer_card}</div>{steps_table}'


def run_page_polls(run: dict[str, Any]) -> bool:
    """Whether the run page loads ``/static/run.js`` (running statuses only)."""

    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    status = str(state.get("status", run.get("status", "")) or "")
    return bool(status) and status not in LIVE_STOP_STATUSES


_PIPELINE_SYMBOLS = {
    "complete": "✓", "running": "▶", "failed": "✗", "waiting": "·", "resumable": "↻", "skipped": "–",
}
_FAILURE_MESSAGES = {
    "CLAUDE_FAILED": "Claude invocation failed",
    "CLAUDE_AUTH_FAILURE": "Claude authentication failed",
    "CLAUDE_TIMEOUT": "Claude revision timed out",
    "CLAUDE_COMMITTED": "Claude created a commit",
    "CODEX_AUTH_FAILURE": "Codex authentication failed",
    "AGENT_TIMEOUT": "Luna step timed out",
    "AGENT_FAILED": "Luna step failed",
    "AGENT_NO_CHANGE": "Luna step changed nothing",
    "STEP_WRITE_SET_VIOLATION": "Luna step changed an unauthorized path",
    "REVIEWER_TRANSPORT_FAILURE": "Reviewer could not be reached",
    "REVIEWER_OUTPUT_INVALID": "Reviewer answer is invalid",
    "LLM_FAILURE": "Model call failed",
    "PUSH_FAILED": "Publication push failed",
    "BASE_MOVED_SINCE_RUN": "Base branch moved since the run started",
    "RESUME_INTEGRITY_FAILURE": "Resume refused: the run no longer matches its checkpoint",
    "RESUME_REQUIRES_OPERATOR": "Resume requires an operator",
    "REVIEW_LOOP_EXHAUSTED": "Automatic correction budget exhausted",
    "INTERRUPTED": "Run interrupted",
}


def _short_id(run_id: Any) -> str:
    text = str(run_id or "")
    tail = text.rsplit("-", 1)[-1]
    return tail if 0 < len(tail) < len(text) else text[:16]


def _pipeline_section(overview: dict[str, Any]) -> str:
    items = overview.get("pipeline") if isinstance(overview.get("pipeline"), list) else []
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        state = item.get("state") if item.get("state") in _PIPELINE_SYMBOLS else "waiting"
        rows.append(
            f'<li class="state-{state}" data-pipeline-key="{_e(item.get("key"))}">'
            f'<span class="symbol">{_PIPELINE_SYMBOLS[state]}</span> {_e(item.get("label"))}</li>'
        )
    return (
        '<section class="pipeline"><h2>EXECUTION PIPELINE</h2>'
        f'<ol class="pipeline-list">{"".join(rows)}</ol>'
        '<p class="muted small">✓ complete · ▶ running · ✗ failed · · waiting · ↻ resumable</p></section>'
    )


def _run_configuration(state: dict[str, Any], config: HarnessConfig | None = None) -> str:
    """Render the safe requested snapshot and any changed final profiles."""

    snapshot = state.get("run_options") if isinstance(state.get("run_options"), dict) else {}
    if not snapshot and config is not None:
        snapshot = RunOptions.from_config(config).to_dict()
    planning = snapshot.get("planning") if isinstance(snapshot.get("planning"), dict) else {}
    pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
    requested = snapshot.get("profiles") if isinstance(snapshot.get("profiles"), dict) else {}
    execution = state.get("execution") if isinstance(state.get("execution"), dict) else {}
    final: dict[str, Any] = {}
    planner = execution.get("planner") if isinstance(execution.get("planner"), dict) else {}
    if planner.get("profile_id"):
        final["planner_profile"] = planner["profile_id"]
    steps = execution.get("steps") if isinstance(execution.get("steps"), list) else []
    if steps and isinstance(steps[0], dict) and isinstance(steps[0].get("implementer"), dict):
        final["default_implementer_profile"] = steps[0]["implementer"].get("profile_id")
    for key, role in (("reviewer_profile", "reviewer"), ("reviser_profile", "reviser"), ("repair_profile", "repair_implementer")):
        item = execution.get(role)
        if isinstance(item, dict):
            final[key] = item.get("profile_id")
    rows = [
        ("decomposition", planning.get("decomposition")),
        ("execution mode", planning.get("execution_mode_policy")),
        ("SINGLE mutable limit", planning.get("single_step_max_mutable_paths")),
        ("STAGED mutable limit", planning.get("staged_step_max_mutable_paths")),
        ("Claude revision", "on" if pipeline.get("claude_revision_enabled") else "off"),
        ("repair cycles", pipeline.get("repair_cycles")),
    ]
    for key, label in (("planner_profile", "planner"), ("default_implementer_profile", "implementer"), ("reviewer_profile", "reviewer"), ("reviser_profile", "reviser"), ("repair_profile", "repair")):
        rows.append((f"requested {label}", requested.get(key)))
        if key in final and final.get(key) != requested.get(key):
            rows.append((f"effective/final {label}", final.get(key)))
    return '<section class="card"><h2>RUN CONFIGURATION</h2><dl>' + "".join(
        f'<dt>{_e(label)}</dt><dd class="mono">{_e(value if value is not None else "—")}</dd>'
        for label, value in rows
    ) + '</dl></section>'


def _failure_card(run: dict[str, Any], token: str | None, overview: dict[str, Any]) -> str:
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    status = str(state.get("status", run.get("status", "")) or "")
    failure = run.get("failure", state.get("failure"))
    if status not in {"failed", "interrupted"} or not isinstance(failure, dict):
        return ""
    reason = str(failure.get("reason") or "")
    resume = overview.get("resume") if isinstance(overview.get("resume"), dict) else {}
    note = (
        "Reviewer requested one bounded implementation correction loop."
        if reason == "REVIEW_REVISE" else
        "Automatic correction budget exhausted."
        if reason == "REVIEW_LOOP_EXHAUSTED" else ""
    )
    action = ""
    if resume.get("resumable"):
        label = _e(resume.get("label"))
        checkpoint = resume.get("phase") or "—"
        checkpoint_detail = (
            f'<p class="small mono">RESUME FROM {_e(checkpoint)}'
            f' · tree={_e(resume.get("expected_tree") or "—")}'
            f' · cycle={_e(resume.get("cycle") or "—")}'
            f' · step={_e(resume.get("step_id") or "—")}</p>'
        )
        button = (
            f'<form action="/runs/{_e(run.get("run_id"))}/resume" method="post">'
            f'<input type="hidden" name="_token" value="{_e(token)}">'
            f'<button class="resume" type="submit">{label}</button></form>'
            if token else f'<p><strong>{label}</strong></p>'
        )
        action = f'<p class="label">NEXT ACTION</p>{checkpoint_detail}{button}'
    elif resume.get("reason"):
        action = f'<p class="danger small">RESUME REFUSED: {_e(resume.get("reason"))}</p>'
    detail = failure.get("detail")
    return (
        '<div class="card fail failure-card"><p class="label">FAILED</p>'
        f'<p class="failure-title"><strong>{_e(_FAILURE_MESSAGES.get(reason, reason or "Run failed"))}</strong></p>'
        f'{action}'
        f'<p class="danger small"><strong>FAILED: {_e(reason)}</strong>'
        f'{"<br>" + _e(note) if note else ""}{"<br>" + _e(detail) if detail is not None else ""}</p></div>'
    )


def _run_card(run: dict[str, Any], token: str | None, overview: dict[str, Any], is_v2: bool) -> str:
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    status = str(state.get("status", run.get("status", "")) or "")
    run_id = run.get("run_id")
    totals = overview.get("token_totals") if isinstance(overview.get("token_totals"), dict) else {}
    tokens = "".join(
        f'<dt>{label}</dt><dd id="live-tokens-{key}">{_usage_pair(totals.get(key))}</dd>'
        for key, label in (("planner", "Planner"), ("luna", "Luna"), ("claude", "Claude"), ("reviewer", "Reviewer"))
    )
    style = "failed" if status in {"failed", "blocked", "plan_rejected", "interrupted"} else "success" if status in {"committed", "approved", "published"} else ""
    failure = run.get("failure", state.get("failure"))
    live_reason = failure.get("reason") if isinstance(failure, dict) else ""
    polls = run_page_polls(run)
    return (
        '<header class="sticky run-card">'
        f'<h1>Run <span class="mono">{_e(run_id)}</span></h1>'
        '<div class="card-grid">'
        f'<div><p class="label">RUN</p><p class="value mono">{_e(_short_id(run_id))}</p></div>'
        f'<div><p class="label">STATUS</p><p class="value"><span id="live-status" class="badge {style}">{_e(status.upper() or "—")}</span></p>'
        f'<p class="muted small">updated <span id="live-updated" class="mono">{_e(run.get("updated_at"))}</span></p></div>'
        f'<div><p class="label">CURRENT</p><p class="value" id="live-current">{_e(overview.get("current_label") or "—")}</p></div>'
        f'<div><p class="label">NEXT</p><p class="value" id="live-next">{_e(overview.get("next_label") or "—")}</p></div>'
        f'<div><p class="label">EXECUTION</p><p class="value">{_e(overview.get("execution_label") or "—")}</p></div>'
        f'<div><p class="label">PUBLISH TARGET</p><p class="value">{_e(overview.get("publish_target") or "—")}</p>'
        f'<p class="muted small">{_e(overview.get("publish_target_detail") or "")}</p></div>'
        f'<div><p class="label">TOKENS</p><dl class="tokens">{tokens}</dl></div>'
        '</div>'
        f'{_final_summary(run) if is_v2 else ""}'
        f'<p id="live-failure" class="danger"{"" if polls and live_reason else " hidden"}>{_e(live_reason)}</p>'
        f'{_failure_card(run, token, overview)}'
        + ('<button type="button" id="refresh-details" hidden>Actualiser les détails</button>' if polls else "")
        + '</header>'
    )


def _live_events_card(polls: bool) -> str:
    if not polls:
        return ""
    slots = "".join('<li data-slot hidden></li>' for _ in range(20))
    return (
        '<div class="card live"><h3>Live events</h3>'
        f'<ul id="live-events" class="events"><li id="live-events-empty" class="muted">No live event yet.</li>{slots}</ul></div>'
    )


def render_run(run: dict[str, Any], token: str | None = None, *, config: HarnessConfig | None = None, nonce: str | None = None, refresh_seconds: int | None = None) -> str:
    """Status/action-oriented run page.

    Sections, in order: run status and next action, execution pipeline,
    approval / model selection, current cycle, checks, token usage, plan,
    changed files, raw artifacts / diagnostics.  ``refresh_seconds`` is kept
    for compatibility and ignored: the page never reloads itself.
    """

    del refresh_seconds
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    status = state.get("status", run.get("status")); run_id = run.get("run_id")
    plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}; failure = run.get("failure", state.get("failure"))
    approval = run.get("approval") if isinstance(run.get("approval"), dict) else {}
    overview = run.get("overview") if isinstance(run.get("overview"), dict) else {}
    can_decide = status == AWAITING_APPROVAL_STATUS and bool(token) and not approval.get("recorded")
    recommendation = state.get("recommendation") if isinstance(state.get("recommendation"), dict) else {}
    impl_selected = recommendation.get("implementer_profile") if recommendation.get("status") == "READY" else config.ui.default_implementer_profile if config else None
    review_selected = recommendation.get("reviewer_profile") if recommendation.get("status") == "READY" else config.ui.default_reviewer_profile if config else None
    recommendation_note = ""
    if recommendation.get("status") == "READY" and config is not None:
        names = {profile.id: profile.display_name for profile in profiles_for_config(config).values()}
        recommendation_note = f'<p>Recommended implementer: {_e(names.get(impl_selected, impl_selected))}<br>Recommended reviewer: {_e(names.get(review_selected, review_selected))}</p><p><strong>Rationale:</strong> {_e(recommendation.get("rationale"))}</p>'
    approval_forms = ""
    is_v2 = state.get("planning_protocol") == "v2"
    if can_decide and is_v2:
        approval_forms = _v2_approval_form(run_id, token or "", state, config, run.get("step_artifacts"))
    elif can_decide:
        approval_forms = f'''<section class="card"><h2>Plan approval</h2>
{recommendation_note}
<form action="/runs/{_e(run_id)}/approval" method="post"><input type="hidden" name="_token" value="{_e(token)}"><input type="hidden" name="decision" value="APPROVE"><label for="implementer-profile">Implementer profile</label><select id="implementer-profile" name="implementer_profile" required>{_profile_options(config, "implementer", impl_selected)}</select><label for="reviewer-profile">Reviewer profile</label><select id="reviewer-profile" name="reviewer_profile" required>{_profile_options(config, "reviewer", review_selected)}</select><br><button class="approve" type="submit">APPROVE</button></form>
<form action="/runs/{_e(run_id)}/approval" method="post"><input type="hidden" name="_token" value="{_e(token)}"><input type="hidden" name="decision" value="REJECT"><button class="reject" type="submit">REJECT</button></form></section>'''
    elif approval.get("recorded"):
        approval_forms = f'<section class="card"><h2>Plan approval</h2><p>Décision enregistrée : {_e(approval.get("decision"))}</p></section>'
    diagnostics = run.get("agent_diagnostics") if isinstance(run.get("agent_diagnostics"), dict) else {}; result = diagnostics.get("result") if isinstance(diagnostics.get("result"), dict) else {}; usage = diagnostics.get("usage") if isinstance(diagnostics.get("usage"), dict) else {}
    candidate = run.get("candidate") if isinstance(run.get("candidate"), dict) else {}; changed_files = candidate.get("changed_files") if isinstance(candidate.get("changed_files"), list) else []
    polls = run_page_polls(run)
    failed = status in {"failed", "interrupted"}
    agent_section = (
        (_cycle_sections(run) if isinstance(run.get("cycle_artifacts"), list) and run.get("cycle_artifacts") else _v2_steps(state, run.get("step_artifacts")))
        if is_v2 else
        f'<details open{_section_open(run, ("AGENT_", "CODEX_AUTH_FAILURE"))}><summary>Agent diagnostics</summary><dl><dt>exit_code</dt><dd>{_e(result.get("exit_code"))}</dd><dt>timed_out</dt><dd>{_e(result.get("timed_out"))}</dd><dt>input_tokens</dt><dd>{_e(usage.get("input_tokens"))}</dd><dt>output_tokens</dt><dd>{_e(usage.get("output_tokens"))}</dd></dl><h3>Final report</h3><pre>{_e(diagnostics.get("final_tail"))}</pre><h3>stderr</h3><pre>{_e(diagnostics.get("stderr_tail"))}</pre></details>'
    )
    plan_open = " open" if _section_open(run, ("LLM_FAILURE", "PLAN_", "PLANNER")) else ""
    body = f'''<main id="run" data-run-id="{_e(run_id)}" data-status="{_e(status)}"><p><a href="/">← Tous les runs</a></p>
{_run_card(run, token, overview, is_v2)}
{_publish_section(state)}
{_pipeline_section(overview)}
{_run_configuration(state, config)}
{approval_forms}
<section><h2>EXECUTION</h2>{_execution_card_v2(state, run, config) if is_v2 else _execution_card(state, config)}</section>
<section class="current-cycle"><h2>CURRENT CYCLE</h2>{_agent_auth_failure_notice(run, config)}{_live_events_card(polls)}{agent_section}</section>
<section><h2>CHECKS</h2><details open{_section_open(run, ("CHECK_", "DETERMINISTIC_GATE"))}><summary>Check results</summary>{_check_cards(run.get("checks"))}</details></section>
<section><h2>REVIEW</h2><details open{_section_open(run, ("REVIEW_",))}><summary>Reviewer result</summary>{_review(run.get("review"))}</details></section>
{_usage_section(run)}
<section><h2>PLAN</h2><details{plan_open}><summary>Canonical implementation contract</summary><pre>{_e(plan.get("contract"))}</pre></details><details><summary>planner.raw.md</summary><pre>{_e(plan.get("raw"))}</pre></details><details><summary>SPEC</summary><pre>{_e(run.get("spec"))}</pre></details></section>
<section><h2>DIFF / FILES</h2><p>Changed files</p><ul>{"".join(f'<li class="mono">{_e(path)}</li>' for path in changed_files) or '<li class="muted">Aucun fichier changé.</li>'}</ul><details><summary>Diff</summary><pre>{_e(candidate.get("diff_tail"))}</pre></details></section>
<section><h2>RAW ARTIFACTS / DIAGNOSTICS</h2>
<details{" open" if failed else ""}><summary>Failure diagnostics</summary><p><strong>Failure:</strong> {_failure(failure)}</p><ul>{"".join(f'<li>{_e(item)}</li>' for item in (run.get("progress_tail") or [])) or '<li class="muted">Aucun événement.</li>'}</ul></details>
<details><summary>reviewer.raw.md</summary><pre>{_e(run.get("reviewer_raw"))}</pre></details>
<details{_section_open(run, ("WORKSPACE_SETUP_",))}><summary>Workspace setup</summary>{_setup_cards(run.get("workspace_setup"))}</details>
<details><summary>Timeline</summary><ul class="timeline">{_timeline_items(status)}</ul></details></section></main>'''
    return _page(f"Run {run_id}", body, nonce=nonce, script=polls)

__all__ = ["AWAITING_APPROVAL_STATUS", "RUN_PAGE_DYNAMIC_IDS", "STATE_POLL_MS", "TERMINAL_STATUSES", "refresh_seconds_for_run", "render_index", "render_new_run", "render_run", "run_page_polls"]
