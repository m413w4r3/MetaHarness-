# MetaHarness ↔ Nimbalyst — manual E2E procedure

Prerequisites: `metaharness` on `PATH`, `npm run build` done, extension installed
from this folder, backend module `metaharness-runtime` enabled.

| # | Action | Expected |
|---|--------|----------|
| 1 | Settings → MetaHarness: Executable `metaharness`, Config `/abs/path/examples/autowork.toml`, Port `8765`, Auto start on. Save. | Settings persisted per project. |
| 2 | Open the MetaHarness panel. | Backend spawns `metaharness web --control-token-file <dataDir>/metaharness-control.token`; badge `Connected`. `ss -ltn \| grep 8765` shows `127.0.0.1` only. |
| 2b | Settings → TEST CONNECTION. | Effective config (repo, base ref, planning, approval, checks) shown. No `argv`, env or key values. |
| 3 | Settings → RUN DOCTOR. | One row per check (`id`, PASS/FAIL). A failing check still renders rows (report `ok:false`), never raw secrets. |
| 4 | ＋ New Run, SPEC `Add a one-line comment to README.md`. | Profile selects come from `/api/v1/model-profiles`; defaults = `defaults.*`. Submit → run detail opens at once (POST 202 `{ok,run_id,location,accepted}`). |
| 5 | Logs tab during planning. | Planner events appear progressively; `progress?offset=` strictly increases; no line appears twice. |
| 6 | With `require_plan_approval=true`: status `awaiting_plan_approval`. | Approval form: one profile select per step, reviewer/reviser/repair selects. APPROVE → body carries `step_profile__<id>`; status leaves the approval gate. |
| 7 | Implementation. | Steps tab: successive steps with profile and commit; Checks tab: gate results; repairs if any. |
| 8 | Review tab. | Candidate SHA, verdict, reviewer raw output, semantic revision report if enabled. |
| 9 | Walk tabs Overview, Plan, Steps, Checks, Review, Diff, Usage, Logs, Diagnostics, Results. | Each renders data or an explicit empty state. Logs of a terminal run are read to the end once, then polling stops. |
| 10 | Interrupt a run (stop the service mid-implementation), restart panel. | `Resume` button only when `overview.resume.resumable`; resume keeps the same run id; Logs restart from offset 0 without duplicates. |
| 11 | Security (terminal): see commands below. | All refusals as listed. |

```sh
T=$(cat <dataDir>/metaharness-control.token)            # never paste it anywhere
curl -s -o /dev/null -w '%{http_code}\n' -X POST -H 'X-MetaHarness-Token: bad' -d '{}' 127.0.0.1:8765/api/v1/runs   # 403
curl -s -w ' %{http_code}\n' -H 'Host: evil.com' 127.0.0.1:8765/api/v1/runs                                          # 403 host not allowed
curl -s -w ' %{http_code}\n' -H 'Origin: http://evil.com' -H "X-MetaHarness-Token: $T" -X POST -d '{}' 127.0.0.1:8765/api/v1/runs  # 403
curl -s -w ' %{http_code}\n' -X POST -d '{}' "127.0.0.1:8765/api/v1/runs?token=$T"                                   # 403 (URL token ignored)
curl -s -m2 http://$(hostname -I | awk '{print $1}'):8765/api/v1/health || echo refused                             # refused
curl -s -o /dev/null -w '%{http_code}\n' "127.0.0.1:8765/api/v1/runs/<id>/artifact?name=../../etc/passwd"           # 404
```

Process lifecycle: start a `metaharness web` yourself on 8765 first → panel attaches
(`serverOwned:false`), no second process, and disabling the extension leaves yours
running; mutations then fail with `TOKEN_REJECTED` unless it uses the extension's
token file. A non-MetaHarness listener on 8765 → `PORT_IN_USE`.
