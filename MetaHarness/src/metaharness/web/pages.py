"""Server-rendered, JavaScript-free pages for the local MetaHarness UI."""

from __future__ import annotations

from html import escape
from typing import Any

from ..models import HarnessConfig
from ..profiles import profiles_for_config, safe_profile_metadata


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _page(
    title: str,
    body: str,
    *,
    nonce: str | None = None,
    refresh_seconds: int | None = None,
) -> str:
    nonce_attribute = f' nonce="{_e(nonce)}"' if nonce else ""
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
  </style>
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
    ("validating", "VALIDATING"), ("revising", "REVISING"), ("reviewing", "REVIEWING"),
    ("approved", "APPROVED"), ("committed", "COMMITTED"),
)
_ORDER = {value: index for index, (value, _label) in enumerate(_TIMELINE)}
_TERMINAL_LABELS = {"blocked": "BLOCKED", "plan_rejected": "REJECTED", "failed": "FAILED", "interrupted": "INTERRUPTED"}
TERMINAL_STATUSES = frozenset({"committed", *_TERMINAL_LABELS})
AWAITING_APPROVAL_STATUS = "awaiting_plan_approval"
# Kept as harmless compatibility constants for callers that used the former
# polling template. The page itself contains no script and does not poll.
RUN_PAGE_DYNAMIC_IDS: tuple[str, ...] = ()
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
    style = "failed" if value in {"failed", "blocked", "plan_rejected", "interrupted"} else "success" if value in {"committed", "approved"} else ""
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
    profiles = [safe_profile_metadata(profile) for profile in profiles_for_config(config).values() if any(role.value == "planner" for role in profile.roles)]
    options = "".join(f'<option value="{_e(p["id"])}"{" selected" if p["id"] == config.ui.default_planner_profile else ""}>{_e(p["display_name"])}</option>' for p in profiles)
    body = f'''<main><p><a href="/">← Tous les runs</a></p><h1>New Run</h1>
<dl><dt>Repository</dt><dd class="mono">{_e(config.repo)}</dd><dt>Base ref</dt><dd class="mono">{_e(config.base_ref)}</dd></dl>
<form action="/runs" method="post" accept-charset="UTF-8"><input type="hidden" name="_token" value="{_e(token)}">
<label for="spec">SPEC</label><textarea id="spec" name="spec" rows="20" required></textarea>
<label for="run-id">Run ID (optional)</label><input id="run-id" name="run_id" type="text" autocomplete="off" value="">
<label for="planner-profile">Planner</label><select id="planner-profile" name="planner_profile" required>{options}</select><br><button type="submit">CREATE RUN</button></form></main>'''
    return _page("New Run", body, nonce=nonce)


def _timeline_items(status: Any) -> str:
    current = str(status or "")
    current_order = _ORDER.get(current, -1)
    items = []
    for value, label in _TIMELINE:
        classes = "current" if value == current else "done" if current_order >= 0 and _ORDER.get(value, 99) < current_order else ""
        items.append(f'<li class="{classes}">{label}</li>')
    if current in _TERMINAL_LABELS:
        items.append(f'<li class="current danger">{_TERMINAL_LABELS[current]}</li>')
    return "".join(items)


def _failure_reason(run: dict[str, Any]) -> str:
    failure = run.get("failure")
    return str(failure.get("reason", "")) if isinstance(failure, dict) else str(failure or "")


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
    for role, title in (("planner", "Planner"), ("implementer", "Implementer"), ("reviser", "Reviser"), ("reviewer", "Reviewer")):
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


def _profile_options(config: HarnessConfig | None, role: str, selected: Any) -> str:
    if config is None:
        return ""
    result = []
    for profile in profiles_for_config(config).values():
        item = safe_profile_metadata(profile)
        if role in item["roles"]:
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
    rows: list[str] = []
    for item in steps:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        step_id = item["id"]
        selected = item.get("recommended_profile") or item.get("profile_id")
        rows.append(
            f'<section class="card"><h3>{_e(step_id)} — {_e(item.get("title"))}</h3>'
            f'<p>Recommended implementer: <span class="mono">{_e(selected)}</span></p>'
            f'<label for="step-profile-{_e(step_id)}">Implementer</label>'
            f'<select id="step-profile-{_e(step_id)}" name="step_profile__{_e(step_id)}" required>'
            f'{_profile_options(config, "implementer", selected)}</select>'
            f'{_contract_block(artifact_map.get(step_id, {}))}</section>'
        )
    reviewer = planner.get("reviewer_recommendation")
    return f'''<section class="card"><h2>Execution plan</h2>
<p>Execution mode: <strong>{_e(planner.get("execution_mode"))}</strong></p>
<p>Steps: {_e(len(rows))}</p>
<form action="/runs/{_e(run_id)}/approval" method="post"><input type="hidden" name="_token" value="{_e(token)}"><input type="hidden" name="decision" value="APPROVE">
{"".join(rows)}<label for="reviewer-profile">Reviewer</label><select id="reviewer-profile" name="reviewer_profile" required>{_profile_options(config, "reviewer", reviewer)}</select><br><button class="approve" type="submit">APPROVE PLAN</button></form>
<form action="/runs/{_e(run_id)}/approval" method="post"><input type="hidden" name="_token" value="{_e(token)}"><input type="hidden" name="decision" value="REJECT"><button class="reject" type="submit">REJECT PLAN</button></form></section>'''


def _v2_steps(state: dict[str, Any], artifacts: Any = None) -> str:
    steps = state.get("steps") if isinstance(state.get("steps"), list) else []
    if not steps:
        return '<p class="muted">No staged steps.</p>'
    cards: list[str] = []
    artifact_map = _artifact_map(artifacts)
    for item in steps:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "waiting"))
        icon = _STEP_ICONS.get(status, "…")
        artifact = artifact_map.get(item.get("id"), {})
        usage = artifact.get("usage") if isinstance(artifact.get("usage"), dict) else item.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        warning = f" · {_HIGH_CONTEXT_WARNING}" if artifact.get("high_context") else ""
        events = artifact.get("events") if isinstance(artifact.get("events"), list) else []
        event_items = "".join(f"<li>{_e(event)}</li>" for event in events) or '<li class="muted">No event yet.</li>'
        cards.append(
            f'<details class="card step {_e(status)}"{" open" if status in {"running", "failed"} else ""}>'
            f'<summary>{_e(item.get("id"))} {icon} — {_e(item.get("title"))} · '
            f'{_e(usage.get("input_tokens", 0))} input / {_e(usage.get("output_tokens", 0))} output{warning}</summary>'
            f'<p>profile: <span class="mono">{_e(item.get("profile_id"))}</span></p>'
            f'<p>status: {_e(status)}</p>'
            f'<h4>Recent events</h4><ul class="events">{event_items}</ul>'
            f'<details><summary>contract</summary><pre>{_e(artifact.get("contract"))}</pre></details>'
            f'<details><summary>final report</summary><pre>{_e(artifact.get("final"))}</pre></details>'
            f'<details><summary>stderr</summary><pre>{_e(artifact.get("stderr"))}</pre></details></details>'
        )
    return "".join(cards)


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
    rows = "".join(
        f'<tr><th>{label}</th><td>{_usage_pair(value)}</td><td class="muted">{_usage_detail(value)}</td></tr>'
        for label, value in (
            ("Planner", usage.get("planner")),
            ("Luna", implementer.get("total")),
            ("Reviewer", usage.get("reviewer")),
        )
    )
    step_rows = []
    for step in implementer.get("steps") if isinstance(implementer.get("steps"), list) else []:
        if not isinstance(step, dict):
            continue
        step_usage = step.get("usage") if isinstance(step.get("usage"), dict) else {}
        high = isinstance(step_usage.get("input_tokens"), int) and step_usage["input_tokens"] > _HIGH_WORKER_INPUT_TOKENS
        step_rows.append(
            f'<tr><td class="mono">{_e(step.get("id"))}</td><td>{_usage_pair(step_usage)}</td>'
            f'<td class="muted">{_usage_detail(step_usage)}{" · " + _HIGH_CONTEXT_WARNING if high else ""}</td></tr>'
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
    approved = selection if isinstance(selection, dict) and selection.get("schema_version") == 3 else {}
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
    return f'<div class="grid">{planner_card}{reviewer_card}</div>{steps_table}'


def render_run(run: dict[str, Any], token: str | None = None, *, config: HarnessConfig | None = None, nonce: str | None = None, refresh_seconds: int | None = None) -> str:
    state = run.get("state") if isinstance(run.get("state"), dict) else {}
    status = state.get("status", run.get("status")); run_id = run.get("run_id")
    plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}; failure = run.get("failure", state.get("failure"))
    approval = run.get("approval") if isinstance(run.get("approval"), dict) else {}
    can_decide = status == AWAITING_APPROVAL_STATUS and bool(token) and not approval.get("recorded")
    execution = state.get("execution") if isinstance(state.get("execution"), dict) else {}
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
    refresh = refresh_seconds if refresh_seconds is not None else refresh_seconds_for_run(run)
    failure_top = f'<p class="danger"><strong>FAILED: {_e(failure.get("reason") if isinstance(failure, dict) else failure)}</strong><br>{_e(failure.get("detail") if isinstance(failure, dict) else "")}</p>' if status == "failed" and failure else ""
    body = f'''<main><p><a href="/">← Tous les runs</a></p>
<header class="sticky"><h1>Run <span class="mono">{_e(run_id)}</span></h1><p>{_status_badge(status)} · updated_at <span class="mono">{_e(run.get("updated_at"))}</span></p>{failure_top}</header>
{_usage_section(run)}
{approval_forms}
<section><h2>PLAN</h2><details open{_section_open(run, ("LLM_FAILURE", "PLAN_", "PLANNER"))}><summary>Canonical implementation contract</summary><pre>{_e(plan.get("contract"))}</pre></details><details><summary>planner.raw.md</summary><pre>{_e(plan.get("raw"))}</pre></details><details><summary>SPEC</summary><pre>{_e(run.get("spec"))}</pre></details></section>
<section><h2>EXECUTION</h2>{_execution_card_v2(state, run, config) if is_v2 else _execution_card(state, config)}</section>
<section><h2>AGENT</h2>{_agent_auth_failure_notice(run, config)}{_v2_steps(state, run.get("step_artifacts")) if is_v2 else f'<details open{_section_open(run, ("AGENT_", "CODEX_AUTH_FAILURE"))}><summary>Agent diagnostics</summary><dl><dt>exit_code</dt><dd>{_e(result.get("exit_code"))}</dd><dt>timed_out</dt><dd>{_e(result.get("timed_out"))}</dd><dt>input_tokens</dt><dd>{_e(usage.get("input_tokens"))}</dd><dt>output_tokens</dt><dd>{_e(usage.get("output_tokens"))}</dd></dl><h3>Final report</h3><pre>{_e(diagnostics.get("final_tail"))}</pre><h3>stderr</h3><pre>{_e(diagnostics.get("stderr_tail"))}</pre></details>'}</section>
<section><h2>CHECKS</h2><details open{_section_open(run, ("CHECK_", "DETERMINISTIC_GATE"))}><summary>Check results</summary>{_check_cards(run.get("checks"))}</details></section>
<section><h2>REVIEW</h2><details open{_section_open(run, ("REVIEW_",))}><summary>Reviewer result</summary>{_review(run.get("review"))}</details><details><summary>reviewer.raw.md</summary><pre>{_e(run.get("reviewer_raw"))}</pre></details></section>
<section><h2>Setup</h2><details open{_section_open(run, ("WORKSPACE_SETUP_",))}><summary>Workspace setup</summary>{_setup_cards(run.get("workspace_setup"))}</details></section>
<section><h2>DIFF / FILES</h2><p>Changed files</p><ul>{"".join(f'<li class="mono">{_e(path)}</li>' for path in changed_files) or '<li class="muted">Aucun fichier changé.</li>'}</ul><details><summary>Diff</summary><pre>{_e(candidate.get("diff_tail"))}</pre></details></section>
<section><h2>DIAGNOSTICS</h2><p><strong>Failure:</strong> {_failure(failure)}</p><ul>{"".join(f'<li>{_e(item)}</li>' for item in (run.get("progress_tail") or [])) or '<li class="muted">Aucun événement.</li>'}</ul></section>
<section><h2>Timeline</h2><ul class="timeline">{_timeline_items(status)}</ul></section></main>'''
    return _page(f"Run {run_id}", body, nonce=nonce, refresh_seconds=refresh)


__all__ = ["AWAITING_APPROVAL_STATUS", "RUN_PAGE_DYNAMIC_IDS", "STATE_POLL_MS", "TERMINAL_STATUSES", "refresh_seconds_for_run", "render_index", "render_new_run", "render_run"]
