# MetaHarness repository rules

- Python >= 3.12; prefer the standard library.
- RunStateStore owns atomic state writes.
- Never persist or print secrets. api_key_env is a variable name, never its value.
- Production orchestration stays provider/model neutral.
- Agents must not commit, move HEAD, create branches, publish, or widen scope.
- Prefer targeted reads and targeted tests. Never dump large files or full logs when a bounded excerpt is sufficient.
- On failure, inspect the smallest useful diagnostic excerpt.
- Run the relevant adversarial module when parser, subprocess, security, Git or resume boundaries change; do not run the entire slow adversarial suite unnecessarily.

## Output discipline

Work first; narrate minimally.

Final response:
- changed paths;
- checks actually run and status;
- one blocker/residual risk when present.

Do not restate the task.
Do not summarize unchanged code.
Do not explain routine edits.
Do not provide tutorials.

Target <= 8 lines and <= 1200 characters.
