#!/usr/bin/env python3
"""
Orchestrateur de prompts Codex + Claude Code : sessions indépendantes,
séquentielles, suivies, avec commit git après chaque tâche.

Sous-commandes
--------------
  run <manifeste>   Exécute les tâches
  status / watch    État du run (temps réel)
  logs <task_id>    Événements + réponse finale d'une tâche
  tasks <manifeste> Valide le manifeste sans rien exécuter

Manifeste (.md)
---------------
    # ~/perso/AutoWork
    @defaults provider=codex model=gpt-5.6-luna effort=high
    - prompts/p1.md : gpt-5.6-luna - high
    - prompts/p2.md : claude-sonnet-5 - high | provider=claude ctx=p1
    - prompts/p3.md : opus - xhigh | provider=claude permission=acceptEdits

Le format .json / .jsonl reste accepté.
Stdlib uniquement. Nécessite `git` et au moins un CLI parmi `codex` / `claude`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

RUNS_ROOT = Path(os.environ.get("AGENT_RUNS_DIR",
                                    os.environ.get("CODEX_RUNS_DIR", "runs")))
LATEST = RUNS_ROOT / "latest"

C_CYAN, C_GREEN, C_RED, C_YEL, C_DIM, C_OFF = (
    "\033[1;36m", "\033[32m", "\033[1;31m", "\033[33m", "\033[2m", "\033[0m"
)
PROVIDERS = ("codex", "claude")
CODEX_EFFORTS = ("low", "medium", "high", "xhigh")
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max", "ultracode")
EFFORTS = tuple(dict.fromkeys(CODEX_EFFORTS + CLAUDE_EFFORTS))
CLAUDE_MODEL_ALIASES = {"sonnet", "opus", "haiku", "fable"}
CLAUDE_PERMISSION_MODES = {
    "default", "manual", "acceptEdits", "plan", "auto", "dontAsk",
    "bypassPermissions",
}
CLAUDE_SANDBOX_TO_PERMISSION = {
    "read-only": "plan",
    "workspace-write": "acceptEdits",
    "danger-full-access": "bypassPermissions",
}


# --------------------------------------------------------------------------- #
# Modèle de tâche
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    id: str
    prompt_file: str | None = None
    prompt: str | None = None
    title: str | None = None
    provider: str = "codex"              # codex | claude
    model: str | None = None
    effort: str | None = None
    sandbox: str = "workspace-write"     # abstraction commune; voir mapping Claude ci-dessous
    cwd: str | None = None
    profile: str | None = None            # Codex uniquement
    schema: str | None = None
    context_from: str | None = None
    timeout: int = 0
    retries: int = 1
    commit: bool = True

    # Claude Code (optionnels)
    permission_mode: str | None = None
    max_turns: int = 0
    max_budget_usd: float | None = None
    allowed_tools: str | None = None
    disallowed_tools: str | None = None
    fallback_model: str | None = None
    bare: bool = False

    # `-c key=value` de Codex. Conservé pour rétrocompatibilité.
    extra_config: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.title or self.id

    def resolve_prompt(self, run_dir: Path) -> str:
        parts: list[str] = []
        if self.context_from:
            prev = run_dir / f"{self.context_from}.out.md"
            if prev.exists():
                parts.append(
                    f"## Contexte issu de l'étape « {self.context_from} »\n\n"
                    f"{prev.read_text(encoding='utf-8').strip()}\n\n---\n"
                )
            else:
                print(f"  ! ctx={self.context_from} : sortie absente, ignorée", file=sys.stderr)
        if self.prompt_file:
            p = Path(self.prompt_file).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"Tâche {self.id} : prompt introuvable ({p})")
            parts.append(p.read_text(encoding="utf-8"))
        elif self.prompt:
            parts.append(self.prompt)
        else:
            raise ValueError(f"Tâche {self.id} : aucun prompt")
        return "\n".join(parts)

    def build_argv(self, out_file: Path) -> list[str]:
        if self.provider == "claude":
            return self._build_claude_argv()
        return self._build_codex_argv(out_file)

    def _build_codex_argv(self, out_file: Path) -> list[str]:
        # `codex exec` est non-interactif par nature. Tous les chemins sont
        # absolus car `-C` déplace le répertoire de travail.
        argv = ["codex", "exec", "--json"]
        for opt in ("--color", "--skip-git-repo-check"):
            if exec_supports(opt):
                argv += [opt, "never"] if opt == "--color" else [opt]
        argv += [
            "--sandbox", self.sandbox,
            "--output-last-message", str(Path(out_file).expanduser().absolute()),
        ]
        if self.model:
            argv += ["-m", self.model]
        if self.effort:
            argv += ["-c", f'model_reasoning_effort="{self.effort}"']
        for k, v in self.extra_config.items():
            argv += ["-c", f"{k}={v}"]
        if self.cwd:
            argv += ["-C", str(Path(self.cwd).expanduser().absolute())]
        if self.profile:
            argv += ["-p", self.profile]
        if self.schema:
            argv += ["--output-schema", str(Path(self.schema).expanduser().absolute())]
        return argv + ["-"]

    def _build_claude_argv(self) -> list[str]:
        # Claude Code lit le prompt sur stdin quand aucun prompt n'est donné en
        # argument. `stream-json` nous permet de réutiliser le suivi temps réel.
        argv = ["claude", "-p", "--output-format", "stream-json"]
        if self.model:
            argv += ["--model", self.model]
        if self.effort:
            argv += ["--effort", self.effort]

        permission = self.permission_mode or CLAUDE_SANDBOX_TO_PERMISSION.get(self.sandbox)
        if permission:
            argv += ["--permission-mode", permission]
        if self.max_turns:
            argv += ["--max-turns", str(self.max_turns)]
        if self.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(self.max_budget_usd)]
        if self.allowed_tools:
            argv += ["--allowedTools", self.allowed_tools]
        if self.disallowed_tools:
            argv += ["--disallowedTools", self.disallowed_tools]
        if self.fallback_model:
            argv += ["--fallback-model", self.fallback_model]
        if self.bare:
            argv += ["--bare"]
        if self.schema:
            schema_path = Path(self.schema).expanduser()
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            argv += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
        return argv

    def subprocess_cwd(self) -> str | None:
        # Codex gère le cwd avec `-C`; Claude Code doit être lancé depuis le projet.
        if self.provider != "claude" or not self.cwd:
            return None
        return str(Path(self.cwd).expanduser().absolute())


# --------------------------------------------------------------------------- #
# Parseur du manifeste Markdown
# --------------------------------------------------------------------------- #
RE_SECTION = re.compile(r"^#\s+(?P<cwd>[^\s#].*?)\s*$")
RE_DEFAULTS = re.compile(r"^@defaults\s+(?P<kv>.+)$", re.I)
RE_ITEM = re.compile(r"^[-*]\s+(?P<rest>.+)$")
RE_DASH = re.compile(r"\s+-\s+")            # séparateur « modèle - effort »
RE_TITLE = re.compile(r"^#\s+(.+)$", re.M)  # premier titre du fichier prompt


def _strip_comment(line: str) -> str:
    """Retire un commentaire de fin de ligne (« # … ») sans casser les chemins."""
    out, started = [], False
    for i, ch in enumerate(line):
        if ch == "#" and started and (i == 0 or line[i - 1].isspace()):
            break
        if not ch.isspace():
            started = True
        out.append(ch)
    return "".join(out).strip()


def _parse_kv(blob: str) -> dict[str, str]:
    return dict(tok.split("=", 1) for tok in blob.split() if "=" in tok)


def _parse_spec(spec: str) -> tuple[str | None, str | None]:
    """« modèle - effort » -> (modèle, effort). Tolère l'un des deux absent."""
    spec = spec.strip()
    if not spec:
        return None, None
    bits = RE_DASH.split(spec, maxsplit=1)
    if len(bits) == 2:
        return (bits[0].strip() or None), (bits[1].strip() or None)
    token = bits[0].strip().lstrip("-").strip()
    if token in EFFORTS:
        return None, token
    return (token or None), None


SANDBOX_ALIASES = {
    "ro": "read-only", "read-only": "read-only", "readonly": "read-only",
    "rw": "workspace-write", "write": "workspace-write",
    "full": "danger-full-access", "danger": "danger-full-access",
}


def infer_provider(model: str | None, explicit: str | None = None) -> str:
    """Déduit Claude pour ses alias/modèles; Codex reste le défaut historique."""
    if explicit:
        p = explicit.lower()
        aliases = {"anthropic": "claude", "claude-code": "claude", "openai": "codex"}
        p = aliases.get(p, p)
        if p not in PROVIDERS:
            raise ValueError(f"Provider inconnu « {explicit} » — attendu : {', '.join(PROVIDERS)}")
        return p
    if model and (model.lower() in CLAUDE_MODEL_ALIASES or model.lower().startswith("claude-")):
        return "claude"
    return "codex"

_EXEC_HELP: str | None = None


def exec_supports(flag: str) -> bool:
    """Le jeu de flags de `codex exec` varie selon la version. On lit l'aide une
    fois (instantané, sans appel API) plutôt que de supposer. Si l'aide est
    indisponible, on considère le flag supporté."""
    global _EXEC_HELP
    if _EXEC_HELP is None:
        try:
            r = subprocess.run(["codex", "exec", "--help"], capture_output=True,
                               text=True, timeout=20)
            _EXEC_HELP = (r.stdout or "") + (r.stderr or "")
        except (OSError, subprocess.SubprocessError):
            _EXEC_HELP = ""
    return (not _EXEC_HELP) or (flag in _EXEC_HELP)


def _apply_flag(task: Task, flag: str, base: Path, where: str) -> None:
    if flag in SANDBOX_ALIASES:
        task.sandbox = SANDBOX_ALIASES[flag]
        return
    if flag == "nocommit":
        task.commit = False
        return
    if flag == "bare":
        task.bare = True
        return
    if "=" not in flag:
        raise ValueError(f"{where} — option inconnue « {flag} »")
    k, v = flag.split("=", 1)
    k = k.replace("-", "_")
    match k:
        case "ctx":
            task.context_from = v
        case "timeout":
            task.timeout = int(v)
        case "retries":
            task.retries = int(v)
        case "profile":
            task.profile = v
        case "schema":
            task.schema = str((base / v).expanduser())
        case "id":
            task.id = v
        case "provider" | "backend" | "engine":
            task.provider = infer_provider(task.model, v)
        case "model":
            task.model = v
        case "effort":
            task.effort = v
        case "sandbox":
            task.sandbox = SANDBOX_ALIASES.get(v, v)
        case "permission" | "permission_mode":
            task.permission_mode = v
        case "max_turns":
            task.max_turns = int(v)
        case "budget" | "max_budget_usd":
            task.max_budget_usd = float(v)
        case "allowed_tools":
            task.allowed_tools = v
        case "disallowed_tools":
            task.disallowed_tools = v
        case "fallback_model":
            task.fallback_model = v
        case _:
            # Les options libres historiques sont des `-c` Codex. Pour Claude,
            # on refuse ensuite en preflight au lieu de les ignorer silencieusement.
            task.extra_config[k] = v


def parse_markdown_manifest(path: Path) -> list[Task]:
    base = path.parent
    tasks: list[Task] = []
    cwd: str | None = None
    defaults: dict[str, str] = {}
    seen: set[str] = set()

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#!"):
            continue
        where = f"{path}:{lineno}"

        if (m := RE_SECTION.match(line)):
            candidate = _strip_comment(m.group("cwd"))
            # « # Grammaire » n'est pas un chemin : seules les sections qui
            # ressemblent à un répertoire changent le cwd.
            if candidate.startswith(("~", "/", ".")) or Path(candidate).expanduser().is_dir():
                cwd, defaults = candidate, {}
            continue

        if (m := RE_DEFAULTS.match(line)):
            defaults.update(_parse_kv(_strip_comment(m.group("kv"))))
            continue

        if not (m := RE_ITEM.match(line)):
            continue
        rest = _strip_comment(m.group("rest"))
        if not rest:
            continue

        body, _, flagblob = rest.partition("|")
        pathpart, _, spec = body.partition(":")
        model, effort = _parse_spec(spec)

        pf = (base / pathpart.strip()).expanduser()
        selected_model = model or defaults.get("model")
        explicit_provider = (defaults.get("provider") or defaults.get("backend")
                             or defaults.get("engine"))
        task = Task(
            id=pf.stem,
            prompt_file=str(pf),
            cwd=cwd,
            provider=infer_provider(selected_model, explicit_provider),
            model=selected_model,
            effort=effort or defaults.get("effort"),
            sandbox=SANDBOX_ALIASES.get(defaults.get("sandbox", ""),
                                        defaults.get("sandbox", "workspace-write")),
            profile=defaults.get("profile"),
            permission_mode=defaults.get("permission") or defaults.get("permission_mode"),
            max_turns=int(defaults.get("max_turns", "0")),
            max_budget_usd=(float(defaults["max_budget_usd"])
                            if "max_budget_usd" in defaults else
                            float(defaults["budget"]) if "budget" in defaults else None),
            allowed_tools=defaults.get("allowed_tools"),
            disallowed_tools=defaults.get("disallowed_tools"),
            fallback_model=defaults.get("fallback_model"),
            bare=defaults.get("bare", "false").lower() in ("1", "true", "yes", "on"),
        )
        for flag in flagblob.split():
            _apply_flag(task, flag, base, where)

        if task.id in seen:
            n = 2
            while f"{task.id}-{n}" in seen:
                n += 1
            task.id = f"{task.id}-{n}"
        seen.add(task.id)

        if pf.exists() and (t := RE_TITLE.search(pf.read_text(encoding="utf-8"))):
            task.title = t.group(1).strip()

        tasks.append(task)

    if not tasks:
        raise ValueError(f"{path} : aucune tâche trouvée")
    return tasks


def parse_json_manifest(path: Path) -> list[Task]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        entries = [json.loads(l) for l in raw.splitlines()
                   if l.strip() and not l.lstrip().startswith("#")]
    else:
        data = json.loads(raw)
        entries = data["tasks"] if isinstance(data, dict) else data
    known = set(Task.__dataclass_fields__)
    out = []
    for i, e in enumerate(entries, 1):
        e.setdefault("id", f"task-{i:02d}")
        if "provider" not in e:
            e["provider"] = infer_provider(e.get("model"))
        if (unknown := set(e) - known):
            raise ValueError(f"Tâche {e['id']} : champs inconnus {sorted(unknown)}")
        out.append(Task(**e))
    return out


def load_manifest(path: Path) -> list[Task]:
    if not path.exists():
        sys.exit(f"Manifeste introuvable : {path}")
    # Chemin absolu obligatoire : les tâches tournent avec -C <cwd>, donc tout
    # chemin relatif serait résolu depuis le répertoire du projet, pas le nôtre.
    path = path.resolve()
    if path.suffix.lower() in (".json", ".jsonl"):
        return parse_json_manifest(path)
    return parse_markdown_manifest(path)


# --------------------------------------------------------------------------- #
# Git
# --------------------------------------------------------------------------- #
def git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)


def git_root(cwd: Path) -> Path | None:
    r = git(cwd, "rev-parse", "--show-toplevel")
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None


def git_dirty(cwd: Path) -> list[str]:
    r = git(cwd, "status", "--porcelain")
    return [l for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else []


def git_head(cwd: Path) -> str | None:
    r = git(cwd, "rev-parse", "--short", "HEAD")
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def git_commit(cwd: Path, task: Task, run_id: str) -> tuple[str | None, str]:
    """Commite ce qui a changé sous `cwd`. Retourne (sha court, note d'état)."""
    if not git_root(cwd):
        return None, "pas un dépôt git"
    if not git_dirty(cwd):
        return None, "aucun changement"
    if (add := git(cwd, "add", "-A", ".")).returncode != 0:
        return None, f"git add a échoué : {add.stderr.strip()[:120]}"
    subject = task.label if len(task.label) <= 72 else task.label[:69] + "..."
    body = (
        f"Généré par agent_runner ({run_id}).\n\n"
        f"Tâche   : {task.id}\n"
        f"Backend : {task.provider}\n"
        f"Prompt  : {task.prompt_file}\n"
        f"Modèle  : {task.model or 'défaut'}\n"
        f"Effort  : {task.effort or 'défaut'}\n"
        f"Sandbox : {task.sandbox}\n"
    )
    if (c := git(cwd, "commit", "-m", subject, "-m", body)).returncode != 0:
        return None, f"git commit a échoué : {c.stderr.strip()[:120]}"
    return git_head(cwd), "commit créé"


# --------------------------------------------------------------------------- #
# État persistant
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, run_dir: Path, manifest: str = "", tasks: list[Task] | None = None):
        self.path = run_dir / "state.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            for t in tasks or []:
                self.data["tasks"].setdefault(t.id, self._blank(t))
        else:
            self.data = {
                "run_id": run_dir.name, "manifest": manifest,
                "runner_pid": os.getpid(), "started_at": time.time(),
                "finished_at": None,
                "tasks": {t.id: self._blank(t) for t in (tasks or [])},
            }
        self.data["runner_pid"] = os.getpid()
        self.flush()

    @staticmethod
    def _blank(t: Task) -> dict:
        return {"status": "pending", "title": t.label, "attempt": 0,
                "started_at": None, "ended_at": None, "exit_code": None,
                "activity": None, "provider": t.provider, "model": t.model,
                "effort": t.effort, "cwd": t.cwd, "commit": None}

    def flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def update(self, tid: str, **f) -> None:
        self.data["tasks"].setdefault(tid, {}).update(f)
        self.flush()

    def get(self, tid: str) -> dict:
        return self.data["tasks"].get(tid, {})


# --------------------------------------------------------------------------- #
# Flux d'événements
# --------------------------------------------------------------------------- #
def extract_usage(ev: dict) -> dict | None:
    """Récupère la consommation de tokens, où qu'elle se trouve dans l'événement.
    Le schéma varie selon les versions, donc on cherche largement."""
    candidates = [ev, ev.get("msg"), ev.get("info"), ev.get("item"), ev.get("message")]
    for node in candidates:
        if not isinstance(node, dict):
            continue
        for key in ("total_token_usage", "token_usage", "usage", "last_token_usage"):
            u = node.get(key)
            if isinstance(u, dict) and any(
                k in u for k in ("input_tokens", "output_tokens", "total_tokens")
            ):
                return {k: v for k, v in u.items() if isinstance(v, int)}
    return None


def fmt_tokens(u: dict | None) -> str:
    if not u:
        return "-"
    total = u.get("total_tokens")
    if total is None:
        total = u.get("input_tokens", 0) + u.get("output_tokens", 0)
    return f"{total:,}".replace(",", " ")


def summarize_event(ev: dict) -> str | None:
    etype = ev.get("type") or ev.get("msg", {}).get("type") or "?"

    # Claude Code / Agent SDK stream-json.
    if etype == "assistant" and isinstance(ev.get("message"), dict):
        content = ev["message"].get("content") or []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = block.get("name", "outil")
                inp = block.get("input") or {}
                if name == "Bash" and isinstance(inp, dict) and inp.get("command"):
                    return f"$ {str(inp['command']).replace(chr(10), ' ')[:110]}"
                if name in ("Edit", "Write") and isinstance(inp, dict):
                    path = inp.get("file_path") or inp.get("path") or ""
                    return f"édition : {path}" if path else f"outil : {name}"
                return f"outil : {name}"
            if block.get("type") == "text" and block.get("text"):
                return f"message : {str(block['text']).replace(chr(10), ' ')[:110]}"
        return "réflexion…"
    if etype == "result":
        if ev.get("subtype") == "success":
            return "résultat final"
        return f"ERREUR Claude : {ev.get('subtype', 'échec')}"
    if etype == "system" and ev.get("subtype") == "init":
        return "session Claude initialisée"

    # Codex JSONL.
    item = ev.get("item") or ev.get("msg", {}).get("item") or {}
    itype = item.get("type", "")
    if "command" in item:
        return f"$ {str(item['command']).replace(chr(10), ' ')[:110]}"
    if itype in ("file_change", "patch_apply", "apply_patch"):
        paths = item.get("paths") or item.get("files") or []
        return f"édition : {', '.join(map(str, paths))[:110]}"
    if itype in ("reasoning", "agent_reasoning"):
        return "réflexion…"
    if itype in ("agent_message", "assistant_message"):
        return f"message : {str(item.get('text', '')).replace(chr(10), ' ')[:110]}"
    if "error" in str(etype).lower():
        return f"ERREUR : {json.dumps(ev, ensure_ascii=False)[:150]}"
    if str(etype).endswith(("started", "completed", "begin", "end")):
        return str(etype)
    return None


def tail(path: Path, n: int = 15) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def stream_task(task: Task, run_dir: Path, state: State, verbose: bool) -> int:
    out_file = run_dir / f"{task.id}.out.md"
    ev_file = run_dir / f"{task.id}.events.jsonl"
    err_file = run_dir / f"{task.id}.stderr.log"
    prompt = task.resolve_prompt(run_dir)
    (run_dir / f"{task.id}.prompt.txt").write_text(prompt, encoding="utf-8")

    argv = task.build_argv(out_file)
    state.update(task.id, status="running", started_at=time.time(),
                 argv=shlex.join(argv), activity="démarrage…", child_pid=None)

    deadline = time.time() + task.timeout if task.timeout else None
    timed_out = False
    claude_result: dict | None = None

    # stderr est toujours capturé : les deux CLI y écrivent notamment leurs
    # erreurs d'arguments, de modèle et d'authentification.
    with err_file.open("w", encoding="utf-8") as errfh:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errfh,
            cwd=task.subprocess_cwd(), text=True, encoding="utf-8", bufsize=1)
        state.update(task.id, child_pid=proc.pid)
        assert proc.stdin and proc.stdout
        proc.stdin.write(prompt)
        proc.stdin.close()

        with ev_file.open("w", encoding="utf-8") as fh:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                if deadline and time.time() > deadline:
                    timed_out = True
                    proc.send_signal(signal.SIGINT)
                    state.update(task.id, activity="timeout — interruption")
                    break
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (u := extract_usage(ev)):
                    state.update(task.id, tokens=u)
                if task.provider == "claude" and ev.get("type") == "result":
                    claude_result = ev
                    if isinstance(ev.get("total_cost_usd"), (int, float)):
                        state.update(task.id, cost_usd=ev["total_cost_usd"])
                    if ev.get("session_id"):
                        state.update(task.id, session_id=ev["session_id"])
                if (s := summarize_event(ev)):
                    state.update(task.id, activity=s, activity_at=time.time())
                    if verbose:
                        print(f"    {C_DIM}· {s}{C_OFF}")

        code = proc.wait()

    if task.provider == "claude" and claude_result is not None:
        if claude_result.get("subtype") == "success":
            final = claude_result.get("structured_output")
            if final is None:
                final = claude_result.get("result", "")
            if isinstance(final, (dict, list)):
                text = json.dumps(final, indent=2, ensure_ascii=False) + "\n"
            else:
                text = str(final or "")
            out_file.write_text(text, encoding="utf-8")
        elif code == 0:
            # Le binaire peut avoir terminé proprement alors que la boucle agent a
            # atteint max_turns / budget / validation de schéma. Pour le runner,
            # c'est bien un échec de tâche.
            code = 1

    if code != 0:
        state.update(task.id, stderr_tail=tail(err_file, 5))
    return 124 if timed_out else code


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def preflight(tasks: list[Task], allow_dirty: bool, do_commit: bool) -> None:
    providers = {t.provider for t in tasks}
    unknown = providers - set(PROVIDERS)
    if unknown:
        sys.exit(f"Provider(s) inconnu(s) : {', '.join(sorted(unknown))}")

    if "codex" in providers:
        if not shutil.which("codex"):
            sys.exit("CLI `codex` introuvable dans le PATH.")
        try:
            ok = subprocess.run(["codex", "login", "status"], capture_output=True,
                                timeout=20).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        if not ok:
            sys.exit("Codex n'est pas authentifié : lance `codex login`.")

    if "claude" in providers:
        if not shutil.which("claude"):
            sys.exit("CLI `claude` introuvable dans le PATH.")
        try:
            ok = subprocess.run(["claude", "auth", "status"], capture_output=True,
                                timeout=20).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        if not ok:
            sys.exit("Claude Code n'est pas authentifié : lance `claude auth login`.")

    for t in tasks:
        if t.prompt_file and not Path(t.prompt_file).expanduser().exists():
            sys.exit(f"Prompt introuvable : {t.prompt_file}  (tâche {t.id})")
        if t.cwd and not Path(t.cwd).expanduser().is_dir():
            sys.exit(f"Répertoire inexistant : {t.cwd}  (tâche {t.id})")
        efforts = CLAUDE_EFFORTS if t.provider == "claude" else CODEX_EFFORTS
        if t.effort and t.effort not in efforts:
            sys.exit(f"Effort invalide « {t.effort} » pour {t.provider} (tâche {t.id}) — "
                     f"attendu : {', '.join(efforts)}")
        if t.provider == "claude":
            if t.profile:
                sys.exit(f"Tâche {t.id} : `profile=` est spécifique à Codex.")
            if t.extra_config:
                sys.exit(f"Tâche {t.id} : options Codex -c incompatibles avec Claude : "
                         f"{', '.join(sorted(t.extra_config))}")
            if t.permission_mode and t.permission_mode not in CLAUDE_PERMISSION_MODES:
                sys.exit(f"Tâche {t.id} : permission Claude inconnue « {t.permission_mode} »")
            if t.schema and not Path(t.schema).expanduser().is_file():
                sys.exit(f"Schéma introuvable : {t.schema}  (tâche {t.id})")

    if not do_commit or allow_dirty:
        return
    checked: set[Path] = set()
    for t in tasks:
        if not (t.commit and t.cwd):
            continue
        cwd = Path(t.cwd).expanduser()
        if cwd in checked:
            continue
        checked.add(cwd)
        if git_root(cwd) and (dirty := git_dirty(cwd)):
            print(f"{C_RED}Dépôt sale : {cwd}{C_OFF}")
            for l in dirty[:10]:
                print(f"  {l}")
            sys.exit("Commite ou remise ton travail, ou relance avec --allow-dirty.")


def cmd_run(args) -> int:
    manifest = Path(args.manifest)
    tasks = load_manifest(manifest)
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t.id in wanted]
        if not tasks:
            sys.exit(f"Aucune tâche pour --only {args.only}")

    do_commit = not args.no_commit
    preflight(tasks, args.allow_dirty, do_commit)

    if args.resume and LATEST.exists():
        run_dir = LATEST.resolve()
        print(f"Reprise du run {run_dir.name}")
    else:
        run_dir = (RUNS_ROOT / time.strftime("%Y%m%d-%H%M%S")).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        if LATEST.is_symlink() or LATEST.exists():
            LATEST.unlink()
        LATEST.symlink_to(run_dir, target_is_directory=True)

    state = State(run_dir, manifest=str(manifest), tasks=tasks)
    print(f"Run : {run_dir}  ({len(tasks)} tâches, commit={'oui' if do_commit else 'non'})\n")

    failures = 0
    for i, task in enumerate(tasks, 1):
        if state.get(task.id).get("status") == "ok" and not args.force:
            print(f"[{i}/{len(tasks)}] {task.id} — déjà réussie, ignorée")
            continue

        print(f"{C_CYAN}[{i}/{len(tasks)}] {task.label}{C_OFF}")
        print(f"{C_DIM}   id={task.id} provider={task.provider} "
              f"model={task.model or 'défaut'} effort={task.effort or 'défaut'} "
              f"sandbox={task.sandbox} cwd={task.cwd or '.'}{C_OFF}")

        code, attempt, start = 1, 0, time.time()
        while True:
            attempt += 1
            state.update(task.id, attempt=attempt)
            start = time.time()
            try:
                code = stream_task(task, run_dir, state, args.verbose)
            except KeyboardInterrupt:
                state.update(task.id, status="interrupted", ended_at=time.time())
                state.data["finished_at"] = time.time()
                state.flush()
                sys.exit("\nInterrompu par l'utilisateur.")
            except Exception as exc:
                state.update(task.id, status="failed", exit_code=-1,
                             ended_at=time.time(), activity=str(exc))
                print(f"  {C_RED}{exc}{C_OFF}")
                code = -1
                break
            if code == 0:
                break
            err = tail(run_dir / f"{task.id}.stderr.log", 15)
            if err:
                print(f"  {C_RED}stderr :{C_OFF}")
                for l in err.splitlines():
                    print(f"    {C_DIM}{l}{C_OFF}")
            # Code 2 = erreur d'usage du CLI (modèle inconnu, flag invalide).
            # Réessayer à l'identique ne peut pas aider.
            if code == 2:
                hint = ("`codex debug models`" if task.provider == "codex"
                        else "`claude --help` / le nom du modèle")
                print(f"  {C_YEL}! code 2 = erreur d'arguments — pas de nouvelle "
                      f"tentative. Vérifie avec {hint}.{C_OFF}")
                break
            if attempt > task.retries:
                break
            print(f"  {C_YEL}! échec (code {code}) — tentative {attempt + 1} dans 15 s{C_OFF}")
            time.sleep(15)

        elapsed = round(time.time() - start, 1)

        if code == 0:
            state.update(task.id, status="ok", exit_code=0,
                         ended_at=time.time(), activity="terminée")
            print(f"  {C_GREEN}OK{C_OFF} en {elapsed}s · "
                  f"{fmt_tokens(state.get(task.id).get('tokens'))} tokens "
                  f"→ {run_dir / f'{task.id}.out.md'}")
            if do_commit and task.commit and task.cwd:
                sha, note = git_commit(Path(task.cwd).expanduser(), task, run_dir.name)
                state.update(task.id, commit=sha, commit_note=note)
                print(f"  {C_DIM}git : {note}{f' ({sha})' if sha else ''}{C_OFF}")
        else:
            failures += 1
            state.update(task.id, status="failed", exit_code=code, ended_at=time.time())
            print(f"  {C_RED}ÉCHEC{C_OFF} (code {code}) — "
                  f"{run_dir / f'{task.id}.stderr.log'}")
            print(f"  {C_DIM}rejouer à la main : "
                  f"jq -r '.tasks[\"{task.id}\"].argv' {run_dir / 'state.json'}{C_OFF}")
            if task.cwd:
                cwd = Path(task.cwd).expanduser()
                if git_root(cwd) and git_dirty(cwd):
                    print(f"  {C_YEL}! modifications non commitées laissées dans "
                          f"{task.cwd} — inspecte-les avant de reprendre{C_OFF}")
            if not args.keep_going:
                break

        if i < len(tasks) and args.pause:
            time.sleep(args.pause)

    state.data["finished_at"] = time.time()
    state.flush()
    print()
    render_status(run_dir)
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# status / logs / tasks
# --------------------------------------------------------------------------- #
def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def render_status(run_dir: Path) -> None:
    sf = run_dir / "state.json"
    if not sf.exists():
        print(f"Aucun état dans {run_dir}")
        return
    d = json.loads(sf.read_text(encoding="utf-8"))
    alive = not d.get("finished_at") and _alive(d.get("runner_pid"))
    print(f"Run {d['run_id']}  ·  {d.get('manifest', '?')}  ·  runner "
          f"{'actif' if alive else 'terminé'} (pid {d.get('runner_pid')})")
    print(f"{'ID':<16} {'BACKEND':<7} {'ÉTAT':<11} {'DURÉE':>7} {'TOKENS':>9} "
          f"{'COÛT':>9} {'EFF':<9} {'COMMIT':<9} ACTIVITÉ")
    print("-" * 132)
    labels = {"ok": (C_GREEN, "ok"), "failed": (C_RED, "échec"),
              "running": (C_YEL, "en cours"), "pending": ("", "en attente"),
              "interrupted": ("", "interrompue"), "orpheline": (C_RED, "orpheline")}
    grand = {"input_tokens": 0, "output_tokens": 0}
    grand_total = 0
    for tid, t in d["tasks"].items():
        st = t.get("status", "pending")
        if st == "running" and not _alive(t.get("child_pid")) and not alive:
            st = "orpheline"
        col, txt = labels.get(st, ("", st))
        dur = (f"{(t.get('ended_at') or time.time()) - t['started_at']:.0f}s"
               if t.get("started_at") else "-")
        u = t.get("tokens") or {}
        for k in grand:
            grand[k] += u.get(k, 0)
        grand_total += u.get("total_tokens",
                             u.get("input_tokens", 0) + u.get("output_tokens", 0))
        cost = t.get("cost_usd")
        cost_s = f"${cost:.4f}" if isinstance(cost, (int, float)) else "-"
        print(f"{tid:<16} {(t.get('provider') or 'codex'):<7} {col}{txt:<11}{C_OFF} "
              f"{dur:>7} {fmt_tokens(u):>9} {cost_s:>9} "
              f"{(t.get('effort') or '-'):<9} {(t.get('commit') or '-'):<9} "
              f"{(t.get('activity') or '')[:36]}")
    if grand_total or grand["input_tokens"]:
        print(f"\nTotal : {grand_total:,} tokens".replace(",", " ")
              + (f"  (entrée {grand['input_tokens']:,} / sortie {grand['output_tokens']:,})"
                 .replace(",", " ") if grand["input_tokens"] else ""))


def cmd_status(args) -> int:
    run_dir = Path(args.run) if args.run else LATEST
    if not run_dir.exists():
        sys.exit("Aucun run trouvé.")
    if getattr(args, "watch", False):
        try:
            while True:
                os.system("clear")
                render_status(run_dir.resolve())
                print("\n(Ctrl-C pour quitter)")
                time.sleep(2)
        except KeyboardInterrupt:
            pass
    else:
        render_status(run_dir.resolve())
    return 0


def cmd_logs(args) -> int:
    run_dir = (Path(args.run) if args.run else LATEST).resolve()
    ev = run_dir / f"{args.task_id}.events.jsonl"
    if not ev.exists():
        sys.exit(f"Pas d'événements pour {args.task_id} dans {run_dir}")
    for line in ev.read_text(encoding="utf-8").splitlines():
        try:
            if (s := summarize_event(json.loads(line))):
                print(s)
        except json.JSONDecodeError:
            continue
    out = run_dir / f"{args.task_id}.out.md"
    if out.exists():
        print("\n--- réponse finale ---\n")
        print(out.read_text(encoding="utf-8"))
    return 0


def cmd_tasks(args) -> int:
    for i, t in enumerate(load_manifest(Path(args.manifest)), 1):
        print(f"{C_CYAN}{i}. {t.label}{C_OFF}")
        print(f"   id={t.id}  provider={t.provider}  cwd={t.cwd or '.'} "
              f"commit={'oui' if t.commit else 'non'}"
              f"{'  ctx=' + t.context_from if t.context_from else ''}")
        print(f"   {C_DIM}{' '.join(t.build_argv(Path(f'runs/<run>/{t.id}.out.md')))}{C_OFF}\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Orchestrateur de prompts Codex + Claude Code")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="exécuter un manifeste")
    r.add_argument("manifest")
    r.add_argument("--only", help="ids séparés par des virgules")
    r.add_argument("--resume", action="store_true", help="reprendre le dernier run")
    r.add_argument("--force", action="store_true", help="réexécuter même les tâches OK")
    r.add_argument("--keep-going", action="store_true", help="continuer après un échec")
    r.add_argument("--no-commit", action="store_true", help="désactive tous les commits")
    r.add_argument("--allow-dirty", action="store_true", help="ne pas exiger un dépôt propre")
    r.add_argument("--pause", type=int, default=5, help="secondes entre deux tâches")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="état du run")
    s.add_argument("--run")
    s.add_argument("--watch", action="store_true")
    s.set_defaults(func=cmd_status)

    w = sub.add_parser("watch", help="état rafraîchi en continu")
    w.add_argument("--run")
    w.set_defaults(func=lambda a: cmd_status(argparse.Namespace(run=a.run, watch=True)))

    l = sub.add_parser("logs", help="événements d'une tâche")
    l.add_argument("task_id")
    l.add_argument("--run")
    l.set_defaults(func=cmd_logs)

    t = sub.add_parser("tasks", help="valider le manifeste sans exécuter")
    t.add_argument("manifest")
    t.set_defaults(func=cmd_tasks)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())